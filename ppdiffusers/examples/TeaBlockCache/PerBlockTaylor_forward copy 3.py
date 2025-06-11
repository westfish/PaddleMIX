# -*- coding: utf-8 -*-
"""
Per‑Block Taylor Prediction (Stable Edition)
===========================================
• 仅当 EXT_AVAILABLE 且激活历史≥2 时才做预测  
• 引入 max_skip(默认3)：限制连续跳帧数量  
• fallback 模式下永不预测，只更新缓存  
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import paddle
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers

# ───── 外部 Taylor 库 ───────────────────────────────────────────────────────
try:
    from cache_functions import cache_init_step
    from taylorseer_utils import step_taylor_formula, step_derivative_approximation
    EXT_AVAILABLE = True
except ImportError:
    EXT_AVAILABLE = False

# ───── 安全回退：只更新，不预测 ─────────────────────────────────────────────
def _fallback_cache_init():
    cache_dic = {"cache": {"hidden": {}}, "max_order": 3, "first_enhance": 2}
    current = {"step": 0, "activated_steps": [], "skip_streak": 0}
    return cache_dic, current

def _fallback_taylor_update(cache_dic: Dict, current: Dict, feat: paddle.Tensor):
    cache_dic["cache"]["hidden"][0] = feat
    if "prev" in cache_dic:
        cache_dic["cache"]["hidden"][1] = feat - cache_dic["prev"]
    cache_dic["prev"] = feat.clone()

# 设置一个全局默认：连续跳帧最多 3 步
DEFAULT_MAX_SKIP = 3

# ───── 主前向 ──────────────────────────────────────────────────────────────
def PerBlockTaylorPredictionForward(
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

    joint_attention_kwargs = {} if joint_attention_kwargs is None else joint_attention_kwargs.copy()
    lora_scale = joint_attention_kwargs.pop("scale", 1.0) if joint_attention_kwargs else 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    # ---- Embed & prepare ---------------------------------------------------
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    guidance = None if guidance is None else guidance.to(hidden_states.dtype) * 1000
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    image_rotary_emb = self.pos_embed(paddle.concat((txt_ids, img_ids), axis=0))

    if "ip_adapter_image_embeds" in joint_attention_kwargs:
        joint_attention_kwargs["ip_hidden_states"] = self.encoder_hid_proj(
            joint_attention_kwargs.pop("ip_adapter_image_embeds")
        )

    # ---- 状态表 -----------------------------------------------------------
    if not hasattr(self, "block_states"):
        self.block_states, self.single_block_states = {}, {}
    if not hasattr(self, "cnt"):
        self.cnt = 0
    self.cnt += 1
    force_compute = (self.cnt == 1 or self.cnt == self.num_steps)

    # ======================================================================
    # transformer_blocks
    # ======================================================================
    for idx, block in enumerate(self.transformer_blocks):
        st = self.block_states.setdefault(
            idx,
            {"acc_dist": 0.0, "prev_mod_inp": None, "cache_dic": None, "current": None},
        )
        if st["cache_dic"] is None:
            if EXT_AVAILABLE:
                st["cache_dic"], st["current"] = cache_init_step(self)
            else:
                st["cache_dic"], st["current"] = _fallback_cache_init()

        max_skip = getattr(self, "block_max_skip", DEFAULT_MAX_SKIP)   # 可外部配置

        should_compute = force_compute

        # ---- 启发式距离 ---------------------------------------------------
        if not force_compute and idx >= self.block_cache_start:
            mod_inp = block.norm1(hidden_states, emb=temb)
            mod_inp = mod_inp[0] if isinstance(mod_inp, tuple) else mod_inp
            if st["prev_mod_inp"] is not None:
                C = max(1, mod_inp.shape[-1] // 8)
                rel = (
                    (mod_inp[:, :, :C] - st["prev_mod_inp"][:, :, :C]).abs().mean()
                    / st["prev_mod_inp"][:, :, :C].abs().mean()
                )
                st["acc_dist"] += float(rel)
                if st["acc_dist"] >= self.block_rel_l1_thresh:
                    st["acc_dist"] = 0.0
                    should_compute = True
                else:
                    should_compute = False
            st["prev_mod_inp"] = mod_inp.clone()

        # ---- 🔧 ❶ 预测分支：需满足三条件 ----------------------------------
        use_taylor = (
            not should_compute
            and EXT_AVAILABLE
            and len(st["current"]["activated_steps"]) >= 2
            and st["current"].get("skip_streak", 0) < max_skip
        )

        if use_taylor:
            try:
                pred = step_taylor_formula(st["cache_dic"], st["current"])
            except Exception:
                pred = None
            if pred is not None and paddle.isfinite(pred).all():
                hidden_states = pred
                st["current"]["step"] += 1
                st["current"]["skip_streak"] = st["current"].get("skip_streak", 0) + 1  # 连跳 +1
                continue
            # 预测失败 → 执行真实计算

        # ---- 真实计算 -----------------------------------------------------
        st["current"]["skip_streak"] = 0                            # 重置跳帧计数
        st["current"]["activated_steps"].append(st["current"]["step"])
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )

        if controlnet_block_samples is not None:
            inter = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
            hidden_states += (
                controlnet_block_samples[idx % len(controlnet_block_samples)]
                if controlnet_blocks_repeat
                else controlnet_block_samples[idx // inter]
            )

        # ---- 缓存更新 ----------------------------------------------------
        if EXT_AVAILABLE and len(st["current"]["activated_steps"]) >= 2:
            step_derivative_approximation(st["cache_dic"], st["current"], hidden_states)
        else:
            _fallback_taylor_update(st["cache_dic"], st["current"], hidden_states)
        st["current"]["step"] += 1

    # ======================================================================
    # single_transformer_blocks
    # ======================================================================
    enc_len = encoder_hidden_states.shape[1]
    combined_hidden = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

    for idx, block in enumerate(self.single_transformer_blocks):
        st = self.single_block_states.setdefault(
            idx,
            {"acc_dist": 0.0, "prev_mod_inp": None, "cache_dic": None, "current": None},
        )
        if st["cache_dic"] is None:
            if EXT_AVAILABLE:
                st["cache_dic"], st["current"] = cache_init_step(self)
            else:
                st["cache_dic"], st["current"] = _fallback_cache_init()

        max_skip = getattr(self, "single_max_skip", DEFAULT_MAX_SKIP)
        should_compute = force_compute

        if not force_compute and idx >= self.single_block_cache_start:
            mod_inp = block.norm(combined_hidden, emb=temb)
            mod_inp = mod_inp[0] if isinstance(mod_inp, tuple) else mod_inp
            if st["prev_mod_inp"] is not None:
                C = max(1, mod_inp.shape[-1] // 8)
                rel = (
                    (mod_inp[:, :, :C] - st["prev_mod_inp"][:, :, :C]).abs().mean()
                    / st["prev_mod_inp"][:, :, :C].abs().mean()
                )
                st["acc_dist"] += float(rel)
                if st["acc_dist"] >= self.single_block_rel_l1_thresh:
                    st["acc_dist"] = 0.0
                    should_compute = True
                else:
                    should_compute = False
            st["prev_mod_inp"] = mod_inp.clone()

        use_taylor = (
            not should_compute
            and EXT_AVAILABLE
            and len(st["current"]["activated_steps"]) >= 2
            and st["current"].get("skip_streak", 0) < max_skip
        )

        if use_taylor:
            try:
                pred = step_taylor_formula(st["cache_dic"], st["current"])
            except Exception:
                pred = None
            if pred is not None and paddle.isfinite(pred).all():
                combined_hidden = pred
                st["current"]["step"] += 1
                st["current"]["skip_streak"] = st["current"].get("skip_streak", 0) + 1
                continue

        st["current"]["skip_streak"] = 0
        st["current"]["activated_steps"].append(st["current"]["step"])
        combined_hidden = block(hidden_states=combined_hidden, temb=temb, image_rotary_emb=image_rotary_emb)

        if controlnet_single_block_samples is not None:
            inter = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
            combined_hidden += controlnet_single_block_samples[idx // inter]

        if EXT_AVAILABLE and len(st["current"]["activated_steps"]) >= 2:
            step_derivative_approximation(st["cache_dic"], st["current"], combined_hidden)
        else:
            _fallback_taylor_update(st["cache_dic"], st["current"], combined_hidden)
        st["current"]["step"] += 1

    hidden_states = combined_hidden[:, enc_len:, ...]

    # ---- 输出 ------------------------------------------------------------
    self.cnt = 0 if self.cnt == self.num_steps else self.cnt
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)

# 使用示例
# model.forward = PerBlockTaylorPredictionForward.__get__(model, model.__class__)
