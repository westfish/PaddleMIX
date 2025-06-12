from typing import Any, Dict, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logger,
    scale_lora_layers,
    unscale_lora_layers,
)

def TeaBlockCacheForward(
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
    """
    优化版 TeaBlockCacheForward：
    - 统一主‑block 与 single‑block 逻辑，减少重复代码
    - 使用 paddle.polyval 实现多项式，省去手写五次运算
    - 避免频繁 clone，推理阶段使用 detach
    - register_buffer（若可用）保存多项式系数，自动迁移设备
    """

    # --------------------------- 工具函数 --------------------------- #
    def _init_poly_coeffs(dtype: paddle.dtype):
        coeffs = [
            4.98651651e02,
            -2.83781631e02,
            5.58554382e01,
            -3.82021401e00,
            2.64230861e-01,
        ]
        # 仅首次创建；若当前类继承自 nn.Layer，优先注册为 buffer
        if not hasattr(self, "_poly_coeffs_tensor"):
            try:
                self.register_buffer(
                    "_poly_coeffs_tensor",
                    paddle.to_tensor(coeffs, dtype=dtype),
                    persistable=False,
                )
            except AttributeError:
                # 非 Layer 或 register_buffer 不可用时退化为普通属性
                self._poly_coeffs_tensor = paddle.to_tensor(coeffs, dtype=dtype)

    def _polyval(x: paddle.Tensor) -> paddle.Tensor:
        return paddle.polyval(self._poly_coeffs_tensor, x)

    def _should_compute(
        state_dict: Dict[str, Any],
        mod_inp: paddle.Tensor,
        rel_thresh: float,
        force_compute: bool,
    ) -> bool:
        """
        判断是否需要重新计算当前 block，并更新 state_dict。
        state_dict 字段：
          accum      : 已累计的距离
          previous   : 上一次的 modulated input
          cached_out : 上一次输出（主‑block / single‑block 通用）
          cached_enc : 主‑block 专用 encoder cache
        """
        if force_compute:
            state_dict["previous"] = mod_inp.detach()
            return True

        if state_dict["previous"] is None:
            state_dict["previous"] = mod_inp.detach()
            return True

        # 取前 1/8 通道做启发式
        C = max(1, mod_inp.shape[-1] // 8)
        diff = (mod_inp[:, :, :C] - state_dict["previous"][:, :, :C]).abs().sum()
        base = state_dict["previous"][:, :, :C].abs().sum() + 1e-6
        rel_change = diff / base

        state_dict["accum"] += _polyval(rel_change)
        if state_dict["accum"] >= rel_thresh:
            state_dict["accum"] = 0.0
            state_dict["previous"] = mod_inp.detach()
            return True
        return False

    # --------------------------- 前置处理 --------------------------- #
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs and joint_attention_kwargs.get("scale") is not None:
            logger.warning("未启用 PEFT 时 `scale` 参数无效。")

    hidden_states = self.x_embedder(hidden_states)

    _init_poly_coeffs(hidden_states.dtype)

    timestep = timestep.astype(hidden_states.dtype) * 1000
    guidance = None if guidance is None else guidance.astype(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 兼容旧版三维 id
    if txt_ids.ndim == 3:
        logger.warning("`txt_ids` 三维张量已弃用，请去掉 batch 维度。")
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning("`img_ids` 三维张量已弃用，请去掉 batch 维度。")
        img_ids = img_ids[0]

    ids = paddle.concat((txt_ids, img_ids), axis=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_emb = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        joint_attention_kwargs["ip_hidden_states"] = self.encoder_hid_proj(ip_emb)

    # --------------------------- 状态初始化 --------------------------- #
    within_range = self.step_start <= timestep <= self.step_end

    if timestep.item() == 1000 or getattr(self, "cnt", 0) == 0:
        self.block_heuristic_states = {}
        self.single_block_heuristic_states = {}
        self.cnt = 0

    self.cnt += 1
    force_compute_global = self.cnt in (1, self.num_steps)

    # --------------------------- 核心循环 --------------------------- #
    def _process(
        blocks,
        states_dict,
        cache_start,
        rel_thresh,
        with_encoder: bool,
        control_samples,
    ):
        nonlocal hidden_states, encoder_hidden_states

        for idx, block in enumerate(blocks):
            if idx not in states_dict:
                states_dict[idx] = dict(accum=0.0, previous=None, cached_out=None, cached_enc=None)

            st = states_dict[idx]
            norm_fn = block.norm1 if with_encoder else block.norm
            mod_inp = norm_fn(hidden_states, emb=temb)
            if isinstance(mod_inp, (tuple, list)):
                mod_inp = mod_inp[0]

            compute_now = _should_compute(
                st,
                mod_inp,
                rel_thresh,
                force_compute_global or not (within_range and idx >= cache_start),
            )

            if compute_now:
                if self.training and self.gradient_checkpointing:
                    # 与 torch.checkpoint 类似的多卡重计算
                    hidden_tmp = paddle.distributed.fleet.utils.recompute(
                        block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states if with_encoder else None,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    hidden_tmp = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states if with_encoder else None,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                if with_encoder:
                    encoder_hidden_states, hidden_states = hidden_tmp
                else:
                    hidden_states = hidden_tmp

                # 写缓存（只在允许缓存的层内）
                if within_range and idx >= cache_start:
                    st["cached_out"] = hidden_states.detach()
                    if with_encoder:
                        st["cached_enc"] = encoder_hidden_states.detach()

                # controlnet 残差
                if control_samples is not None:
                    inter = int(np.ceil(len(blocks) / len(control_samples)))
                    if with_encoder:
                        if controlnet_blocks_repeat:
                            hidden_states += control_samples[idx % len(control_samples)]
                        else:
                            hidden_states += control_samples[idx // inter]
                    else:
                        hidden_states[:, encoder_hidden_states.shape[1] :] += control_samples[
                            idx // inter
                        ]
            else:
                # 直接读取缓存
                hidden_states = st["cached_out"]
                if with_encoder:
                    encoder_hidden_states = st["cached_enc"]

    # 主‑blocks
    _process(
        self.transformer_blocks,
        getattr(self, "block_heuristic_states", {}),
        self.block_cache_start,
        self.block_rel_l1_thresh,
        with_encoder=True,
        control_samples=controlnet_block_samples,
    )

    # 拼接 encoder 与 image hidden states
    hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

    # single‑blocks
    _process(
        self.single_transformer_blocks,
        getattr(self, "single_block_heuristic_states", {}),
        self.single_block_cache_start,
        self.single_block_rel_l1_thresh,
        with_encoder=False,
        control_samples=controlnet_single_block_samples,
    )

    # 仅保留图像部分
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    # 计数复位
    if self.cnt == self.num_steps:
        self.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)