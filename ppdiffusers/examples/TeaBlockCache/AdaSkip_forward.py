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

    # ---------- 0. 总开关 ----------
    if not getattr(self, "adaskip_enabled", True):
        return self._orig_forward(
            hidden_states, encoder_hidden_states, pooled_projections, timestep,
            img_ids, txt_ids, guidance, joint_attention_kwargs,
            controlnet_block_samples, controlnet_single_block_samples,
            return_dict, controlnet_blocks_repeat
        )

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
            last_upd=[-1]*n_blk,           # 上次更新的 step
            # single_transformer_blocks 缓存
            single_prev_probe=[None]*n_single_blk,
            single_prev_out=[None]*n_single_blk,
            single_prev_prev=[None]*n_single_blk,
            single_last_upd=[-1]*n_single_blk,
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

    # ---------- 3. 跳跃判定 ----------
    delta0   = getattr(self, "adaskip_delta0", 0.25)
    delta_t  = delta0 * math.exp(step / T * 1.2)
    max_skip = getattr(self, "adaskip_max_skip", 2)

    exec_mask = []
    for i in range(n_blk):
        # 条件1：diff > delta  → 必算
        cond_diff = scores_t[i].item() > delta_t
        # 条件2：超过 max_skip 步强制算
        cond_step = step - cache["last_upd"][i] >= max_skip
        exec_mask.append(cond_diff or cond_step)

    # ---------- 4. Block 循环 ----------
    for i, blk in enumerate(self.transformer_blocks):
        if exec_mask[i]:
            # ---- 真算 ----
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
    
    # 5.2 跳跃判定
    single_exec_mask = []
    for i in range(n_single_blk):
        # 条件1：diff > delta  → 必算
        cond_diff = single_scores_t[i].item() > delta_t
        # 条件2：超过 max_skip 步强制算
        cond_step = step - cache["single_last_upd"][i] >= max_skip
        single_exec_mask.append(cond_diff or cond_step)
    
    # 5.3 single_transformer_blocks 执行循环
    for i, blk in enumerate(self.single_transformer_blocks):
        if single_exec_mask[i]:
            # ---- 真算 ----
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
    FluxTransformer2DModel.forward = AdaSkipFluxForward                    # 猴补
    tr   = pipe.transformer
    tr.num_steps = 50
    tr.adaskip_enabled = True          # 主开关

    # --------- 档位选择 ----------
    # HQ (≈1.5×)
    # tr.adaskip_delta0 = 0.25; tr.adaskip_max_skip = 1
    # Balanced (≈1.9×)
    # tr.adaskip_delta0 = 0.30; tr.adaskip_max_skip = 2
    # Speed (≈2.3×)
    tr.adaskip_delta0 = 0.9; tr.adaskip_max_skip = 3

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
        total_blocks = len(tr.transformer_blocks) + len(tr.single_transformer_blocks)
        total_executions = cache["step"] * total_blocks
        
        # 计算 transformer_blocks 的跳跃数
        transformer_skips = 0
        for i in range(len(tr.transformer_blocks)):
            if cache["last_upd"][i] < cache["step"]:
                transformer_skips += cache["step"] - cache["last_upd"][i] - 1
        
        # 计算 single_transformer_blocks 的跳跃数  
        single_skips = 0
        for i in range(len(tr.single_transformer_blocks)):
            if cache["single_last_upd"][i] < cache["step"]:
                single_skips += cache["step"] - cache["single_last_upd"][i] - 1
        
        total_skips = transformer_skips + single_skips
        skip_rate = total_skips / total_executions * 100
        print(f"[统计] transformer_blocks 跳跃: {transformer_skips}, single_blocks 跳跃: {single_skips}")
        print(f"[统计] 总跳跃率: {skip_rate:.1f}%")

    # 4. 简单结论
    speedup = t_orig / t_adaskip if t_adaskip > 0 else float("inf")
    print(f"速度提升倍数 ≈ {speedup:.2f}×")