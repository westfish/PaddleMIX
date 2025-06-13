import math, paddle, numpy as np
from typing import Any, Dict, Optional, Union
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, is_torch_version

# ---------- 多项式缩放 ----------
_POLY = [4.98651651e02, -2.83781631e02, 5.58554382e01,
         -3.82021401e00, 2.64230861e-01]
def _poly(x):
    a,b,c,d,e = _POLY
    return (((a*x+b)*x+c)*x+d)*x+e

def AdaSkipFluxForward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        pooled_projections: paddle.Tensor = None,
        timestep: paddle.Tensor = None,
        img_ids: paddle.Tensor = None,
        txt_ids: paddle.Tensor = None,
        guidance: paddle.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[paddle.Tensor, Transformer2DModelOutput]:

    # ---------- 0. 总开关和时间范围控制 ----------
    if not getattr(self, "adaskip_enabled", True):
        return self._orig_forward(
            hidden_states, encoder_hidden_states, pooled_projections, timestep,
            img_ids, txt_ids, guidance, joint_attention_kwargs,
            controlnet_block_samples, controlnet_single_block_samples,
            return_dict, controlnet_blocks_repeat
        )
    
    # 检查是否在启用AdaSkip的时间范围内
    step_start = getattr(self, "step_start", 0)      # 默认从0开始
    step_end = getattr(self, "step_end", 1000)       # 默认到1000结束
    current_timestep = timestep.item() if hasattr(timestep, 'item') else float(timestep)
    is_within_time_range = step_start <= current_timestep <= step_end

    joint_attention_kwargs = joint_attention_kwargs.copy() if joint_attention_kwargs else {}
    lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.cast(hidden_states.dtype) * 1000
    guidance = None if guidance is None else guidance.cast(hidden_states.dtype) * 1000
    temb = self.time_text_embed(timestep, pooled_projections) if guidance is None \
           else self.time_text_embed(timestep, guidance, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3: txt_ids = txt_ids[0]
    if img_ids.ndim == 3: img_ids = img_ids[0]
    ids = paddle.concat((txt_ids, img_ids), 0)
    rotary = self.pos_embed(ids)

    if "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_emb = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        joint_attention_kwargs["ip_hidden_states"] = self.encoder_hid_proj(ip_emb)

    # ---------- 1. 缓存区 ----------
    n_blk = len(self.transformer_blocks)
    n_single_blk = len(self.single_transformer_blocks)
    if not hasattr(self, "adaskip_cache"):
        self.adaskip_cache = dict(
            prev_probe=[None]*n_blk,
            prev_out=[None]*n_blk,
            prev_prev=[None]*n_blk,
            prev_enc=[None]*n_blk,
            last_upd=[0]*n_blk,           # 上次更新的 step，初始化为0而不是-1
            # single_transformer_blocks 缓存
            single_prev_probe=[None]*n_single_blk,
            single_prev_out=[None]*n_single_blk,
            single_prev_prev=[None]*n_single_blk,
            single_last_upd=[0]*n_single_blk,  # 初始化为0而不是-1
            # 历史信息用于归一化
            history_scores=[],              # 保存历史scores用于归一化
            history_single_scores=[],       # 保存历史single_scores用于归一化
            max_history_size=50,            # 最多保存50个历史值
            warmup_steps=5,                 # warmup前5步强制执行
            # 统计信息
            transformer_exec_count=0,
            transformer_skip_count=0,
            single_exec_count=0,
            single_skip_count=0,
            step=0
        )
    cache = self.adaskip_cache
    cache["step"] += 1
    step, T = cache["step"], getattr(self, "num_steps", 50)

    # ---------- 2. Probe & 分数 (transformer_blocks) ----------
    scores, probes = [], []
    for i, blk in enumerate(self.transformer_blocks):
        probe, *_ = blk.norm1(hidden_states, emb=temb)
        probes.append(probe.detach())
        if cache["prev_probe"][i] is None:
            score = paddle.full([1], 1.0, dtype=hidden_states.dtype)
        else:
            rel = paddle.abs(probe - cache["prev_probe"][i]).mean() / \
                  (paddle.abs(cache["prev_probe"][i]).mean() + 1e-6)
            score = _poly(rel).unsqueeze(0)
        scores.append(score)
    scores_t = paddle.concat(scores, axis=0)                 # (n_blk,)
    
    # 收集历史scores用于归一化
    max_history_size = cache["max_history_size"]
    history_scores = cache["history_scores"]
    
    # 添加当前scores到历史
    current_scores_cpu = [s.cpu().item() for s in scores_t]
    history_scores.extend(current_scores_cpu)
    
    # 维护固定大小的历史队列
    if len(history_scores) > max_history_size:
        history_scores = history_scores[-max_history_size:]
        cache["history_scores"] = history_scores

    # ---------- 3. 跳跃判定 ----------
    delta0   = getattr(self, "adaskip_delta0", 0.25)  # 0-1范围，0=全算，1=全跳
    max_skip = getattr(self, "adaskip_max_skip", 2)
    warmup_steps = cache["warmup_steps"]
    
    exec_mask = []
    for i in range(n_blk):
        if not is_within_time_range:
            # 如果不在时间范围内，强制执行所有blocks但仍更新缓存
            exec_mask.append(True)
        else:
            # 严格的delta0控制：保证0→0%跳跃，1→100%跳跃，连续性
            # 首先检查是否需要初始化缓存（第一次必须执行）
            need_init = cache["prev_out"][i] is None
            is_warmup = step <= warmup_steps
            
            if need_init or is_warmup:
                # 初始化时或warmup期间必须执行，建立缓存和历史数据
                should_execute = True
            elif delta0 <= 0:
                # delta0=0: 数学保证100%执行（0%跳跃）
                should_execute = True
            elif delta0 >= 1:
                # delta0=1: 数学保证0%执行（100%跳跃）
                should_execute = False
            else:
                # 0<delta0<1: 基于历史信息的连续控制
                raw_score = scores_t[i].item()
                
                # 基于历史数据进行归一化
                if len(history_scores) < 10:
                    # 历史数据不足，先强制执行
                    should_execute = True
                else:
                    # 使用历史数据进行z-score归一化
                    history_mean = sum(history_scores) / len(history_scores)
                    history_var = sum((x - history_mean) ** 2 for x in history_scores) / len(history_scores)
                    history_std = math.sqrt(history_var + 1e-8)  # 避免除零
                    
                    # z-score归一化
                    z_score = (raw_score - history_mean) / history_std
                    # 将z-score转换为[0,1]概率（使用累积分布函数近似）
                    normalized_score = 1 / (1 + math.exp(-z_score))
                    
                    execution_threshold = 1 - delta0  # delta0↑ → threshold↓ → 更难执行
                    should_execute = normalized_score > execution_threshold
                    
                    # max_skip安全网（仅在delta0<1时生效，防止过度跳跃导致质量问题）
                    if not should_execute:
                        cond_step = step - cache["last_upd"][i] >= max_skip
                        should_execute = cond_step
            
            exec_mask.append(should_execute)

    # ---------- 4. Block 循环 ----------
    for i, blk in enumerate(self.transformer_blocks):
        if exec_mask[i]:
            # ---- 真算 ----
            cache["transformer_exec_count"] += 1
            if self.training and self.gradient_checkpointing:
                def wrap(m):
                    return lambda *inp: m(*inp, joint_attention_kwargs=joint_attention_kwargs)
                ckpt_kw = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = paddle.utils.checkpoint.checkpoint(
                    wrap(blk),
                    hidden_states, encoder_hidden_states, temb, rotary, **ckpt_kw
                )
            else:
                encoder_hidden_states, hidden_states = blk(
                    hidden_states, encoder_hidden_states, temb,
                    image_rotary_emb=rotary, joint_attention_kwargs=joint_attention_kwargs
                )
            # ControlNet
            if controlnet_block_samples is not None:
                interval = int(np.ceil(n_blk / len(controlnet_block_samples)))
                idx = i % len(controlnet_block_samples) if controlnet_blocks_repeat else i // interval
                hidden_states += controlnet_block_samples[idx]

            cache["prev_prev"][i] = cache["prev_out"][i]
            cache["prev_out"][i]  = hidden_states.detach()
            cache["prev_probe"][i]= probes[i]
            cache["prev_enc"][i]  = encoder_hidden_states.detach()
            cache["last_upd"][i]  = step
        else:
            # ---- 跳过：复用 / 预测 ----
            cache["transformer_skip_count"] += 1
            hidden_states = cache["prev_out"][i]
            encoder_hidden_states = cache["prev_enc"][i]
            if cache["prev_prev"][i] is not None:
                hidden_states = hidden_states + 0.5 * (hidden_states - cache["prev_prev"][i])
            if controlnet_block_samples is not None:
                interval = int(np.ceil(n_blk / len(controlnet_block_samples)))
                idx = i % len(controlnet_block_samples) if controlnet_blocks_repeat else i // interval
                hidden_states += controlnet_block_samples[idx]

    # ---------- 5. single_transformer_blocks 加速优化 ----------
    hidden_states = paddle.concat([encoder_hidden_states, hidden_states], 1)
    
    # 5.1 为 single_transformer_blocks 计算 probe & 分数
    single_scores, single_probes = [], []
    for i, blk in enumerate(self.single_transformer_blocks):
        probe, *_ = blk.norm(hidden_states, emb=temb)  # single block 的 norm 返回 tuple，需要解包
        single_probes.append(probe.detach())
        if cache["single_prev_probe"][i] is None:
            score = paddle.full([1], 1.0, dtype=hidden_states.dtype)
        else:
            rel = paddle.abs(probe - cache["single_prev_probe"][i]).mean() / \
                  (paddle.abs(cache["single_prev_probe"][i]).mean() + 1e-6)
            score = _poly(rel).unsqueeze(0)
        single_scores.append(score)
    single_scores_t = paddle.concat(single_scores, axis=0)   # (n_single_blk,)
    
    # 收集single_scores的历史信息
    history_single_scores = cache["history_single_scores"]
    current_single_scores_cpu = [s.cpu().item() for s in single_scores_t]
    history_single_scores.extend(current_single_scores_cpu)
    
    # 维护固定大小的历史队列
    if len(history_single_scores) > max_history_size:
        history_single_scores = history_single_scores[-max_history_size:]
        cache["history_single_scores"] = history_single_scores
    
    # 5.2 跳跃判定
    single_exec_mask = []
    for i in range(n_single_blk):
        if not is_within_time_range:
            # 如果不在时间范围内，强制执行所有single blocks但仍更新缓存
            single_exec_mask.append(True)
        else:
            # 严格的delta0控制：与transformer_blocks保持一致
            # 首先检查是否需要初始化缓存（第一次必须执行）
            need_init = cache["single_prev_out"][i] is None
            is_warmup = step <= warmup_steps
            
            if need_init or is_warmup:
                # 初始化时或warmup期间必须执行，建立缓存和历史数据
                should_execute = True
            elif delta0 <= 0:
                # delta0=0: 数学保证100%执行（0%跳跃）
                should_execute = True
            elif delta0 >= 1:
                # delta0=1: 数学保证0%执行（100%跳跃）
                should_execute = False
            else:
                # 0<delta0<1: 基于历史信息的连续控制
                raw_score = single_scores_t[i].item()
                
                # 基于历史数据进行归一化
                if len(history_single_scores) < 10:
                    # 历史数据不足，先强制执行
                    should_execute = True
                else:
                    # 使用历史数据进行z-score归一化
                    history_mean = sum(history_single_scores) / len(history_single_scores)
                    history_var = sum((x - history_mean) ** 2 for x in history_single_scores) / len(history_single_scores)
                    history_std = math.sqrt(history_var + 1e-8)  # 避免除零
                    
                    # z-score归一化
                    z_score = (raw_score - history_mean) / history_std
                    # 将z-score转换为[0,1]概率（使用累积分布函数近似）
                    normalized_score = 1 / (1 + math.exp(-z_score))
                    
                    execution_threshold = 1 - delta0  # delta0↑ → threshold↓ → 更难执行
                    should_execute = normalized_score > execution_threshold
                    
                    # max_skip安全网（仅在delta0<1时生效）
                    if not should_execute:
                        cond_step = step - cache["single_last_upd"][i] >= max_skip
                        should_execute = cond_step
            
            single_exec_mask.append(should_execute)
    
    # 5.3 single_transformer_blocks 执行循环
    for i, blk in enumerate(self.single_transformer_blocks):
        if single_exec_mask[i]:
            # ---- 真算 ----
            cache["single_exec_count"] += 1
            hidden_states = blk(hidden_states, temb=temb,
                                image_rotary_emb=rotary,
                                joint_attention_kwargs=joint_attention_kwargs)
            # 更新缓存
            cache["single_prev_prev"][i] = cache["single_prev_out"][i]
            cache["single_prev_out"][i] = hidden_states.detach()
            cache["single_prev_probe"][i] = single_probes[i]
            cache["single_last_upd"][i] = step
        else:
            # ---- 跳过：复用 / 预测 ----
            cache["single_skip_count"] += 1
            hidden_states = cache["single_prev_out"][i]
            if cache["single_prev_prev"][i] is not None:
                hidden_states = hidden_states + 0.5 * (hidden_states - cache["single_prev_prev"][i])
    
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    out = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    return Transformer2DModelOutput(sample=out) if return_dict else (out,)


if __name__ == "__main__":
    import time
    import paddle
    from ppdiffusers import FluxPipeline
    from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

    # 1. 基本参数
    prompt = "A surreal landscape painted in vibrant watercolor"
    num_inference_steps = 50                    # 推理步数

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16
    )
    pipe.set_progress_bar_config(disable=True)


    # # -------- A. 运行原生前向 ----------
    generator = paddle.Generator().manual_seed(42)
    start = time.time()
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        max_sequence_length=512,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    image.save('origin.png')
    t_orig = time.time() - start
    print(f"[原生] 生成 1 张图耗时：{t_orig:.2f} s")


    # -------- B. 打补丁并启用 AdaSkip ----------
    # 保存原始forward方法
    FluxTransformer2DModel._orig_forward = FluxTransformer2DModel.forward
    FluxTransformer2DModel.forward = AdaSkipFluxForward                    # 猴补
    tr   = pipe.transformer
    tr.num_steps = 50
    tr.adaskip_enabled = True          # 主开关
    
    # --------- 时间范围控制 ----------
    tr.step_start = 50                # 从timestep=50开始启用AdaSkip  
    tr.step_end = 950                 # 到timestep=950结束AdaSkip（更大范围测试）
    
    # --------- 档位选择 (delta0: 0=全算, 1=全跳) ----------
    # Conservative (保守型，少跳跃，高质量)
    # tr.adaskip_delta0 = 0.2; tr.adaskip_max_skip = 1
    # Balanced (平衡型，适中跳跃)
    # tr.adaskip_delta0 = 0.4; tr.adaskip_max_skip = 2
    # Aggressive (激进型，多跳跃，高速度) - 测试历史归一化
    tr.adaskip_delta0 = 0.6; tr.adaskip_max_skip = 3
    # Test extreme case (最大跳跃)
    # tr.adaskip_delta0 = 1.0; tr.adaskip_max_skip = 5  # 测试极端情况

    # -------- C. Warmup阶段：收集历史数据 ----------
    print("\n[Warmup] 开始历史数据收集阶段...")
    warmup_steps = getattr(tr, 'adaskip_cache', {}).get('warmup_steps', 5) if hasattr(tr, 'adaskip_cache') else 5
    
    # 使用delta0=0进行warmup，确保收集到足够的历史数据
    original_delta0 = tr.adaskip_delta0
    tr.adaskip_delta0 = 0.0  # warmup期间强制执行所有blocks
    
    generator = paddle.Generator().manual_seed(42)
    start_warmup = time.time()
    
    # 运行少量步骤进行warmup
    print(f"[Warmup] 运行{warmup_steps + 5}步收集历史数据...")
    warmup_image = pipe(
        prompt=prompt,
        height=512,        # 使用较小尺寸加速warmup
        width=512,
        guidance_scale=3.5,
        max_sequence_length=512,
        num_inference_steps=warmup_steps + 5,  # 比warmup_steps多几步
        generator=generator,
    ).images[0]
    
    t_warmup = time.time() - start_warmup
    print(f"[Warmup] 完成，耗时：{t_warmup:.2f} s")
    
    # 恢复原始delta0设置
    tr.adaskip_delta0 = original_delta0
    
    # 显示warmup后的历史信息
    if hasattr(tr, 'adaskip_cache'):
        cache = tr.adaskip_cache
        history_scores = cache.get('history_scores', [])
        history_single_scores = cache.get('history_single_scores', [])
        print(f"[Warmup] 收集到 transformer_scores: {len(history_scores)}个, single_scores: {len(history_single_scores)}个")
        
        # 重置统计计数器，正式测试时重新计算
        cache["transformer_exec_count"] = 0
        cache["transformer_skip_count"] = 0  
        cache["single_exec_count"] = 0
        cache["single_skip_count"] = 0
        
        # 清除probe缓存（因为图像尺寸变化导致形状不匹配）
        # 保留历史scores数据，但清除尺寸相关的缓存
        n_blk = len(tr.transformer_blocks)
        n_single_blk = len(tr.single_transformer_blocks)
        cache["prev_probe"] = [None]*n_blk
        cache["prev_out"] = [None]*n_blk
        cache["prev_prev"] = [None]*n_blk
        cache["prev_enc"] = [None]*n_blk
        cache["last_upd"] = [0]*n_blk
        cache["single_prev_probe"] = [None]*n_single_blk
        cache["single_prev_out"] = [None]*n_single_blk
        cache["single_prev_prev"] = [None]*n_single_blk
        cache["single_last_upd"] = [0]*n_single_blk
        cache["step"] = 0  # 重置step计数器
        
        print(f"[Warmup] 已清除probe缓存（尺寸变化），保留历史归一化数据")
    
    print(f"[Warmup] 历史数据收集完成，开始正式性能测试（delta0={tr.adaskip_delta0}）...")

    # -------- D. 正式性能测试 ----------
    generator = paddle.Generator().manual_seed(42)
    start = time.time()
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        max_sequence_length=512,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]
    t_adaskip = time.time() - start
    image.save('adaskip.png')
    print(f"[AdaSkip] 生成 1 张图耗时：{t_adaskip:.2f} s")
    
    # 统计跳跃信息
    if hasattr(tr, 'adaskip_cache'):
        cache = tr.adaskip_cache
        
        # 显示时间范围配置
        step_start = getattr(tr, "step_start", 0)
        step_end = getattr(tr, "step_end", 1000)
        print(f"[配置] AdaSkip时间范围: {step_start} ≤ timestep ≤ {step_end}")
        delta0_val = getattr(tr, 'adaskip_delta0', 0.25)
        print(f"[配置] 参数设置: delta0={delta0_val} (0=全算,1=全跳), max_skip={getattr(tr, 'adaskip_max_skip', 2)}")
        print(f"[说明] 基于历史信息归一化，warmup前{cache.get('warmup_steps', 5)}步强制执行")
        print(f"[调试] 总步数: {cache.get('step', 0)}, transformer_blocks: {len(tr.transformer_blocks)}, single_blocks: {len(tr.single_transformer_blocks)}")
        
        # 显示历史信息统计
        history_scores = cache.get('history_scores', [])
        history_single_scores = cache.get('history_single_scores', [])
        if history_scores:
            history_mean = sum(history_scores) / len(history_scores)
            print(f"[历史] transformer_scores: {len(history_scores)}个样本, 均值={history_mean:.3f}")
        if history_single_scores:
            single_history_mean = sum(history_single_scores) / len(history_single_scores) 
            print(f"[历史] single_scores: {len(history_single_scores)}个样本, 均值={single_history_mean:.3f}")
        
        # 计算跳跃统计
        transformer_exec = cache.get("transformer_exec_count", 0)
        transformer_skip = cache.get("transformer_skip_count", 0)
        single_exec = cache.get("single_exec_count", 0) 
        single_skip = cache.get("single_skip_count", 0)
        
        total_exec = transformer_exec + single_exec
        total_skip = transformer_skip + single_skip
        total_operations = total_exec + total_skip
        
        if total_operations > 0:
            skip_rate = total_skip / total_operations * 100
            transformer_skip_rate = transformer_skip / (transformer_exec + transformer_skip) * 100 if (transformer_exec + transformer_skip) > 0 else 0
            single_skip_rate = single_skip / (single_exec + single_skip) * 100 if (single_exec + single_skip) > 0 else 0
            
            print(f"[统计] transformer_blocks - 执行: {transformer_exec}, 跳跃: {transformer_skip} (跳跃率: {transformer_skip_rate:.1f}%)")
            print(f"[统计] single_blocks - 执行: {single_exec}, 跳跃: {single_skip} (跳跃率: {single_skip_rate:.1f}%)")
            print(f"[统计] 总体跳跃率: {skip_rate:.1f}% ({total_skip}/{total_operations})")
        else:
            print("[统计] 暂无统计数据")

    # 4. 最终结论
    speedup = t_orig / t_adaskip if t_adaskip > 0 else float("inf")
    print(f"\n[结论] ============ 性能对比 ============")
    print(f"[结论] 原生forward耗时: {t_orig:.2f} s")
    print(f"[结论] Warmup阶段耗时: {t_warmup:.2f} s") 
    print(f"[结论] AdaSkip正式耗时: {t_adaskip:.2f} s")
    print(f"[结论] 纯加速比 ≈ {speedup:.2f}×")
    total_time = t_warmup + t_adaskip
    effective_speedup = t_orig / total_time if total_time > 0 else float("inf")
    print(f"[结论] 包含warmup的有效加速比 ≈ {effective_speedup:.2f}×")
    print(f"[结论] =====================================")
    
    # 5. 算法总结
    print(f"\n[算法] AdaSkip基于历史信息的归一化控制:")
    print(f"[算法] • Warmup阶段: 收集{warmup_steps + 5}步历史数据")
    print(f"[算法] • 历史队列: 最多保存50个scores样本")
    print(f"[算法] • Z-Score归一化: 基于历史均值和标准差")
    print(f"[算法] • 连续控制: delta0∈[0,1] → 跳跃率[0%,100%]")
    print(f"[算法] • 安全网: max_skip={getattr(tr, 'adaskip_max_skip', 3)}步强制执行")