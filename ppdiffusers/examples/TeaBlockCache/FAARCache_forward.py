import paddle
from typing import Any, Dict, Optional, Union
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import logger, scale_lora_layers, unscale_lora_layers, USE_PEFT_BACKEND

def FAARCacheForward(
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
    # ---------- 0. LoRA 缩放 & 嵌入预处理 ----------
    joint_attention_kwargs = joint_attention_kwargs.copy() if joint_attention_kwargs else {}
    lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    if USE_PEFT_BACKEND: scale_lora_layers(self, lora_scale)
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.cast(hidden_states.dtype) * 1000
    guidance = None if guidance is None else guidance.cast(hidden_states.dtype) * 1000
    temb = self.time_text_embed(timestep, pooled_projections) if guidance is None \
           else self.time_text_embed(timestep, guidance, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # ---------- 1. 计算频域签名 ----------
    # DCT‑II(行列)后取前 1×1 低频分量；等价于全局均值池化
    sig_t = hidden_states.mean(axis=(-2, -1))          # shape (B, C)

    # 初始化缓存区
    if not hasattr(self, "faar_cache"):
        self.faar_cache = {
            "prev_sig": sig_t.clone(),
            "alpha": paddle.ones_like(sig_t),          # AR‑1 系数 α
            "beta": paddle.zeros_like(sig_t),          # AR‑1 截距 β
            "prev_feat": hidden_states.clone(),        # 上一真算步特征
            "alpha_h": paddle.ones([1], dtype=hidden_states.dtype),
            "beta_h": paddle.zeros([1], dtype=hidden_states.dtype),
            "cnt": 0
        }
    cache = self.faar_cache
    cache["cnt"] += 1

    # ---------- 2. AR‑1 预测并判定 ----------
    sig_hat = cache["alpha"] * cache["prev_sig"] + cache["beta"]
    rel_err = (paddle.abs(sig_t - sig_hat).mean() /
               paddle.abs(sig_t).mean())
    should_calc = (rel_err.item() >= getattr(self, "faar_thresh", 0.08)
                   or cache["cnt"] == 1
                   or cache["cnt"] == self.num_steps)

    if should_calc:
        # ---------- 3a. 常规计算 ----------
        # 全部 transformer_blocks 正常跑（可选梯度检查点）
        for i, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=self.pos_embed(paddle.concat((txt_ids, img_ids), 0)),
            )
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], 1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states=hidden_states,
                                  temb=temb,
                                  image_rotary_emb=self.pos_embed(paddle.concat((txt_ids, img_ids), 0)))
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        # ---------- 3b. 在线更新 AR 系数 ----------
        # 对签名 α, β 使用简单矩估
        sig_prev = cache["prev_sig"]
        delta = sig_t - sig_prev
        denom = (sig_prev ** 2).mean() + 1e-6
        new_alpha = (delta * sig_prev).mean() / denom
        cache["alpha"] = 0.9 * cache["alpha"] + 0.1 * new_alpha
        cache["beta"] = 0.9 * cache["beta"] + 0.1 * (sig_t.mean() - cache["alpha"].mean() * sig_prev.mean())

        # 对特征 α_h, β_h (缩放到标量层面，减少显存)
        feat_prev_mean = cache["prev_feat"].mean()
        feat_cur_mean = hidden_states.mean()
        cache["alpha_h"] = 0.9 * cache["alpha_h"] + 0.1 * (feat_cur_mean / (feat_prev_mean + 1e-6))
        cache["beta_h"]  = 0.9 * cache["beta_h"] + 0.1 * (feat_cur_mean - cache["alpha_h"] * feat_prev_mean)

        # 更新缓存
        cache["prev_sig"] = sig_t.clone()
        cache["prev_feat"] = hidden_states.clone()
    else:
        # ---------- 4. 跳步 & 自回归特征预测 ----------
        hidden_states = cache["alpha_h"] * cache["prev_feat"] + cache["beta_h"]

    # ---------- 5. 输出映射 ----------
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND: unscale_lora_layers(self, lora_scale)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)


if __name__ == "__main__":
    import time
    import paddle
    from ppdiffusers import FluxPipeline
    from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

    # 1. 基本参数
    prompt = "A surreal landscape painted in vibrant watercolor"
    num_inference_steps = 50                    # 推理步数

    # 2. 构建原始 Pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16
    )
    pipe.set_progress_bar_config(disable=True)

    # 保留原始 forward 以便后续还原
    orig_forward = FluxTransformer2DModel.forward

    generator = paddle.Generator().manual_seed(42)

    # # -------- A. 运行原生前向 ----------
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

    # -------- B. 打补丁并启用 FAAR‑Cache ----------
    FluxTransformer2DModel.forward = FAARCacheForward
    tr = pipe.transformer
    tr.faar_thresh = 0.3       # 误差阈值，可调
    tr.num_steps = num_inference_steps

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
    image.save('faar.png')
    t_faar = time.time() - start
    print(f"[FAAR‑Cache] 生成 1 张图耗时：{t_faar:.2f} s")

    # 3. 还原 forward，以免影响后续实验
    FluxTransformer2DModel.forward = orig_forward

    # 4. 简单结论
    speedup = t_orig / t_faar if t_faar > 0 else float("inf")
    print(f"速度提升倍数 ≈ {speedup:.2f}×")