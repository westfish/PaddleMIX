from typing import Any, Dict, Optional, Tuple, Union

import time  # ----------------------- NEW -----------------------
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
    """A hybrid caching strategy that combines time and block dimension partitioning with heuristic caching.

    This method extends TeaCache's heuristic approach to work on a per-block basis, allowing for more fine-grained
    control over computation.  **This version contains lightweight profiling code that prints the time spent in
    each major stage, as well as the total extra cost introduced by the heuristic-caching logic.**
    """

    # -------------------------------------------------------------------------
    #                               PROFILING SET-UP
    # -------------------------------------------------------------------------
    total_start = time.perf_counter()
    heuristic_overhead = 0.0  # cumulative time spent in heuristic logic (change detection, etc.)

    # ------------------------ PEFT / LoRA scale handling ----------------------
    stage_start = time.perf_counter()
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    print(f"[TeaBlockCacheForward] LoRA-scaling prep: {time.perf_counter() - stage_start:.4f}s")

    # ------------------------- Initial embeddings ---------------------------
    stage_start = time.perf_counter()
    hidden_states = self.x_embedder(hidden_states)
    print(f"[TeaBlockCacheForward] x_embedder: {time.perf_counter() - stage_start:.4f}s")

    timestep = timestep.to(hidden_states.dtype) * 1000
    guidance = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None

    stage_start = time.perf_counter()
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    print(f"[TeaBlockCacheForward] time_text_embed: {time.perf_counter() - stage_start:.4f}s")

    stage_start = time.perf_counter()
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    print(f"[TeaBlockCacheForward] context_embedder: {time.perf_counter() - stage_start:.4f}s")

    # -------------------- Positional / rotary embedding ----------------------
    stage_start = time.perf_counter()
    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated. "
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated. "
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]
    ids = paddle.concat((txt_ids, img_ids), axis=0)
    image_rotary_emb = self.pos_embed(ids)
    print(f"[TeaBlockCacheForward] pos_embed: {time.perf_counter() - stage_start:.4f}s")

    # -------------------- IP-Adapter (if provided) ---------------------------
    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        stage_start = time.perf_counter()
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
        print(f"[TeaBlockCacheForward] ip_adapter proj: {time.perf_counter() - stage_start:.4f}s")

    # --------------------- Heuristic / cache bookkeeping --------------------
    # Initialise memo dicts only once per new generation-run.
    if not hasattr(self, "block_heuristic_states"):
        self.block_heuristic_states = {}
        self.single_block_heuristic_states = {}

    is_within_time_range = self.step_start <= timestep <= self.step_end

    # reset counters at the beginning of generation
    if timestep == 1000 or self.cnt == 0:
        self.block_heuristic_states.clear()
        self.single_block_heuristic_states.clear()
        self.cnt = 0

    self.cnt += 1  # global step counter
    force_compute = self.cnt in (1, self.num_steps)

    # --------------------- Main transformer blocks loop ---------------------
    for index_block, block in enumerate(self.transformer_blocks):
        block_t0 = time.perf_counter()
        # --------------------------------------------------------------------
        # 1. HEURISTIC decision & state maintenance
        # --------------------------------------------------------------------
        heur_t0 = time.perf_counter()
        if index_block not in self.block_heuristic_states:
            self.block_heuristic_states[index_block] = {
                "accumulated_distance": 0.0,
                "previous_modulated_input": None,
                "cached_output": None,
                "cached_encoder_output": None,
            }
        block_state = self.block_heuristic_states[index_block]

        should_compute_block = force_compute

        if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
            # ---- compute modulated input (cheap, reused later) ----
            inp = hidden_states
            norm_result = inp  # keep as-is; original norm call commented out
            modulated_inp = norm_result[0] if isinstance(norm_result, tuple) else norm_result

            if block_state["previous_modulated_input"] is not None:
                rel_change = (
                    (modulated_inp - block_state["previous_modulated_input"]).abs().mean()
                    / block_state["previous_modulated_input"].abs().mean()
                ).cpu().item()
                coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
                rescale_func = np.poly1d(coefficients)
                block_state["accumulated_distance"] += rescale_func(rel_change)
                if block_state["accumulated_distance"] < self.block_rel_l1_thresh:
                    should_compute_block = False
                else:
                    block_state["accumulated_distance"] = 0
            block_state["previous_modulated_input"] = modulated_inp
        else:
            # still need to update previous input for future comparisons
            if is_within_time_range and index_block >= self.block_cache_start:
                block_state["previous_modulated_input"] = hidden_states

        heuristic_overhead += time.perf_counter() - heur_t0
        # --------------------------------------------------------------------
        # 2. FORWARD pass (or cache reuse)
        # --------------------------------------------------------------------
        if should_compute_block:
            fwd_t0 = time.perf_counter()
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        return module(*inputs, return_dict=return_dict) if return_dict is not None else module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = paddle.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            # cache outputs for potential reuse
            if is_within_time_range and index_block >= self.block_cache_start:
                block_state["cached_output"] = hidden_states
                block_state["cached_encoder_output"] = encoder_hidden_states

            # control-net residual (only if block executed)
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                idx = index_block % len(controlnet_block_samples) if controlnet_blocks_repeat else index_block // interval_control
                hidden_states = hidden_states + controlnet_block_samples[idx]

            print(
                f"[TeaBlockCacheForward] block[{index_block:02d}] forward: {time.perf_counter() - fwd_t0:.4f}s"
            )
        else:
            # cache reuse path
            if block_state["cached_output"] is not None:
                hidden_states = block_state["cached_output"]
                encoder_hidden_states = block_state["cached_encoder_output"]
            print(
                f"[TeaBlockCacheForward] block[{index_block:02d}] reused cache (no fwd)"
            )
        print(f"                 └── total (heur+maybe fwd): {time.perf_counter() - block_t0:.4f}s")

    # -------- Concatenate encoder & image states before single blocks --------
    hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

    # ------------------- Single transformer blocks loop ---------------------
    for index_block, block in enumerate(self.single_transformer_blocks):
        block_t0 = time.perf_counter()
        heur_t0 = time.perf_counter()
        if index_block not in self.single_block_heuristic_states:
            self.single_block_heuristic_states[index_block] = {
                "accumulated_distance": 0.0,
                "previous_modulated_input": None,
                "cached_output": None,
            }
        block_state = self.single_block_heuristic_states[index_block]
        should_compute_block = force_compute

        if not force_compute and is_within_time_range and index_block >= self.single_block_cache_start:
            inp = hidden_states
            modulated_inp = inp  # no separate norm call for brevity
            if block_state["previous_modulated_input"] is not None:
                rel_change = (
                    (modulated_inp - block_state["previous_modulated_input"]).abs().mean()
                    / block_state["previous_modulated_input"].abs().mean()
                ).cpu().item()
                coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
                rescale_func = np.poly1d(coefficients)
                block_state["accumulated_distance"] += rescale_func(rel_change)
                if block_state["accumulated_distance"] < self.single_block_rel_l1_thresh:
                    should_compute_block = False
                else:
                    block_state["accumulated_distance"] = 0
            block_state["previous_modulated_input"] = modulated_inp
        else:
            if is_within_time_range and index_block >= self.single_block_cache_start:
                block_state["previous_modulated_input"] = hidden_states

        heuristic_overhead += time.perf_counter() - heur_t0

        if should_compute_block:
            fwd_t0 = time.perf_counter()
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        return module(*inputs, return_dict=return_dict) if return_dict is not None else module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = paddle.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # cache for reuse
            if is_within_time_range and index_block >= self.single_block_cache_start:
                block_state["cached_output"] = hidden_states

            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] += controlnet_single_block_samples[
                    index_block // interval_control
                ]
            print(
                f"[TeaBlockCacheForward] single_block[{index_block:02d}] forward: {time.perf_counter() - fwd_t0:.4f}s"
            )
        else:
            if block_state["cached_output"] is not None:
                hidden_states = block_state["cached_output"]
            print(
                f"[TeaBlockCacheForward] single_block[{index_block:02d}] reused cache (no fwd)"
            )
        print(f"                 └── total: {time.perf_counter() - block_t0:.4f}s")

    # -------------- Strip encoder portion & final projections ---------------
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    stage_start = time.perf_counter()
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    print(f"[TeaBlockCacheForward] norm_out + proj_out: {time.perf_counter() - stage_start:.4f}s")

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    # end-of-generation counter reset
    if self.cnt == self.num_steps:
        self.cnt = 0

    # -------------------------- PROFILING SUMMARY ---------------------------
    total_elapsed = time.perf_counter() - total_start
    print(
        f"[TeaBlockCacheForward] === SUMMARY ===\n"
        f"    heuristic_overhead: {heuristic_overhead:.4f}s\n"
        f"    total_forward:      {total_elapsed:.4f}s\n"
    )

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
