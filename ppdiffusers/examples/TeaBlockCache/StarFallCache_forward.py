import math, paddle
from typing import Any, Dict, Optional, Union
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

def StarFallCacheForward(
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

    # ---------- 0. 前置嵌入 ----------
    joint_attention_kwargs = joint_attention_kwargs.copy() if joint_attention_kwargs else {}
    lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    if USE_PEFT_BACKEND: scale_lora_layers(self, lora_scale)
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.cast(hidden_states.dtype) * 1000
    guidance = None if guidance is None else guidance.cast(hidden_states.dtype) * 1000
    temb = self.time_text_embed(timestep, pooled_projections) if guidance is None \
         else self.time_text_embed(timestep, guidance, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    image_rotary_emb = self.pos_embed(paddle.concat((txt_ids, img_ids), 0))

    # ---------- 1. 缓存初始化 ----------
    if not hasattr(self, "sf_cache"):
        H, W = hidden_states.shape[-2:]
        tile_size = getattr(self, "sf_tile", 16)
        n_tiles = (H // tile_size) * (W // tile_size)
        self.sf_cache = dict(
            prev_probe=None, prev_feat=hidden_states.clone(),
            alpha=paddle.ones([n_tiles,1,1,1], dtype=hidden_states.dtype),
            beta =paddle.zeros([n_tiles,1,1,1], dtype=hidden_states.dtype),
            cnt=0
        )
    C = hidden_states.shape[1]
    cache = self.sf_cache
    cache["cnt"] += 1

    # ---------- 2. 全局跳步判定 ----------
    g_sig = hidden_states.mean(axis=(-2,-1))                      # (B,C)
    if cache["prev_probe"] is None:
        rel_g = paddle.to_tensor(1.)
    else:
        rel_g = (paddle.abs(g_sig - cache["prev_probe"]).mean() /
                 paddle.abs(cache["prev_probe"]).mean())
    should_skip_all = rel_g.item() < getattr(self, "sf_global_thresh", 0.06) \
                      and cache["cnt"] not in (1, self.num_steps)

    # ---------- 3. 若全跳 → AR‑全图预测 ----------
    if should_skip_all:
        hidden_states = cache["alpha_g"] * cache["prev_feat"] + cache["beta_g"]
    else:
        # ---- 3.1 Tile Probe & 选 K tiles ----
        tile_sz = getattr(self, "sf_tile", 16)
        H, W = hidden_states.shape[-2:]
        hidden_reshaped = hidden_states.reshape(
            [hidden_states.shape[0], C, H//tile_sz, tile_sz, W//tile_sz, tile_sz]
        ).transpose([0,2,4,1,3,5])          # (B,Th,Tw,C,ts,ts)
        tile_mean = hidden_reshaped.mean(axis=(-1,-2))            # (B,Th,Tw,C)

        tile_delta = paddle.abs(tile_mean - cache["prev_probe"]) \
                     if cache["prev_probe"] is not None else paddle.ones_like(tile_mean)
        score = tile_delta.mean(axis=-1)                          # (B,Th,Tw)

        # flatten & 取 Top‑K
        k = getattr(self, "sf_topk", 256)
        score_flat = score.flatten()
        topk_idx = paddle.topk(score_flat, k=min(k, score_flat.shape[0])).indices
        mask_flat = paddle.zeros_like(score_flat, dtype='bool')
        mask_flat[topk_idx] = True
        tile_mask = mask_flat.reshape([H//tile_sz, W//tile_sz])   # True = 需计算

        # ---- 3.2 构建像素 mask → gather ----
        full_mask = tile_mask.unsqueeze(-1).unsqueeze(-1).expand([-1,-1,tile_sz,tile_sz])
        full_mask = full_mask.reshape([H//tile_sz, W//tile_sz, tile_sz, tile_sz])\
                               .transpose([0,2,1,3]).reshape([H,W])  # (H,W)
        full_mask = full_mask.astype('bool')

        def gather_tokens(x):
            # x:(B,C,H,W) → (B,C,Nsel)
            return x[..., full_mask].reshape([x.shape[0], C, -1])

        def scatter_tokens(sel, x):
            out = cache_buf.clone()
            out[..., full_mask] = sel.reshape([x.shape[0], C, -1])
            return out

        cache_buf = hidden_states.clone()   # 用于 scatter

        sel_tokens = gather_tokens(hidden_states)                 # (B,C,Nsel)

        # ---- 3.3 运行 Block 仅算选定 token ----
        for blk in self.transformer_blocks:
            sel_tokens = blk.forward_sparse(                # <== 需在 Block 内提供稀疏接口
                sel_tokens, encoder_hidden_states, temb,
                image_rotary_emb=image_rotary_emb, mask=full_mask
            )
        # scatter 回到整图
        hidden_states = scatter_tokens(sel_tokens, hidden_states)

        # ---- 3.4 单流 Block 全算（量少，可忽略）----
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], 1)
        for blk in self.single_transformer_blocks:
            hidden_states = blk(hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        # ---- 3.5 更新 per‑tile AR 系数 ----
        new_tile_mean = hidden_reshaped.mean(axis=(-1,-2))        # (B,Th,Tw,C)
        alpha = (new_tile_mean / (cache["prev_probe"] + 1e-6)).mean(axis=-1, keepdim=True)
        beta  = new_tile_mean.mean(axis=-1,keepdim=True) - alpha*cache["prev_probe"].mean(axis=-1,keepdim=True)
        cache["alpha"] = 0.9*cache["alpha"] + 0.1*alpha
        cache["beta"]  = 0.9*cache["beta"]  + 0.1*beta
        cache["prev_probe"] = new_tile_mean.detach()
        cache["prev_feat"]  = hidden_states.detach()
        # 全图 AR 系数也同步更新
        cache["alpha_g"] = cache["alpha"].mean()
        cache["beta_g"]  = cache["beta"].mean()

    # ---------- 4. Head ----------
    hidden_states = self.norm_out(hidden_states, temb)
    out = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND: unscale_lora_layers(self, lora_scale)
    return Transformer2DModelOutput(sample=out) if return_dict else (out,)


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
    FluxTransformer2DModel.forward = StarFallCacheForward
    tr = pipe.transformer

    tr.sf_tile = 16          # 每 Tile 16×16
    tr.sf_topk = 256         # 每步最多 256 Tiles 真算
    tr.sf_global_thresh = 0.06
    tr.num_steps = 50

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
    image.save('sfcache.png')
    t_faar = time.time() - start
    print(f"[StarFall‑Cache] 生成 1 张图耗时：{t_faar:.2f} s")

    # 3. 还原 forward，以免影响后续实验
    FluxTransformer2DModel.forward = orig_forward

    # 4. 简单结论
    speedup = t_orig / t_faar if t_faar > 0 else float("inf")
    print(f"速度提升倍数 ≈ {speedup:.2f}×")