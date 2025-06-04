from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, logger, scale_lora_layers, unscale_lora_layers

# === 新增 ===
import time


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

        # === 新增：整体计时 ===
        _t_forward_start = time.perf_counter()
        _timings: Dict[str, float] = {
            'transformer_blocks_heuristic_total': 0.0,
            'single_blocks_heuristic_total':     0.0,
        }

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

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = paddle.concat((txt_ids, img_ids), axis=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        if not hasattr(self, 'block_heuristic_states'):
            self.block_heuristic_states = {}
            self.single_block_heuristic_states = {}

        is_within_time_range = self.step_start <= timestep <= self.step_end

        if timestep == 1000 or self.cnt == 0:
            self.block_heuristic_states = {}
            self.single_block_heuristic_states = {}
            self.cnt = 0

        self.cnt += 1
        force_compute = (self.cnt == 1 or self.cnt == self.num_steps)

        # === 新增：transformer_blocks 总循环计时 ===
        _t_tb_loop_start = time.perf_counter()

        for index_block, block in enumerate(self.transformer_blocks):
            # === 新增：单 block 起始时间（统计总 loop 用） ===
            _t_block_start = time.perf_counter()
            _t_heuristic_start = time.perf_counter()

            if index_block not in self.block_heuristic_states:
                self.block_heuristic_states[index_block] = {
                    'accumulated_distance': 0,
                    'previous_modulated_input': None,
                    'cached_output': None,
                    'cached_encoder_output': None,
                    'should_compute': True
                }

            block_state = self.block_heuristic_states[index_block]
            should_compute_block = force_compute

            # === 新增：heuristic 段计时 ===

            if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
                inp = hidden_states.clone()
                temb_ = temb.clone()
                norm_result = inp
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp = norm_result[0]
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result

                if block_state['previous_modulated_input'] is not None:
                    rel_change = (
                        (modulated_inp - block_state['previous_modulated_input']).abs().mean()
                        / block_state['previous_modulated_input'].abs().mean()
                    ).cpu().item()
                    coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
                    rescale_func = np.poly1d(coefficients)
                    block_state['accumulated_distance'] += rescale_func(rel_change)

                    if block_state['accumulated_distance'] < self.block_rel_l1_thresh:
                        should_compute_block = False
                    else:
                        block_state['accumulated_distance'] = 0
                        should_compute_block = True
                else:
                    should_compute_block = True

                block_state['previous_modulated_input'] = modulated_inp.clone()
            else:
                should_compute_block = True
                if is_within_time_range and index_block >= self.block_cache_start:
                    inp = hidden_states.clone()
                    temb_ = temb.clone()
                    norm_result = inp
                    if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                        modulated_inp = norm_result[0]
                    elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                        modulated_inp = norm_result[0]
                    else:
                        modulated_inp = norm_result
                    block_state['previous_modulated_input'] = modulated_inp.clone()

            # === 新增：累计 heuristic 耗时 ===
            _timings['transformer_blocks_heuristic_total'] += time.perf_counter() - _t_heuristic_start

            if should_compute_block:
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
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

                if is_within_time_range and index_block >= self.block_cache_start:
                    block_state['cached_output'] = hidden_states.clone()
                    block_state['cached_encoder_output'] = encoder_hidden_states.clone()

                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            else:
                if (block_state['cached_output'] is not None and
                    block_state['cached_encoder_output'] is not None):
                    hidden_states = block_state['cached_output']
                    encoder_hidden_states = block_state['cached_encoder_output']

            # === 新增：累计 transformer_blocks loop 耗时（单 block 加到总计） ===
            # 注意这里只记录总循环时间，在循环结束后统一计算

        _timings['transformer_blocks_loop_total'] = time.perf_counter() - _t_tb_loop_start

        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        # === 新增：single_transformer_blocks 总循环计时 ===
        _t_stb_loop_start = time.perf_counter()

        for index_block, block in enumerate(self.single_transformer_blocks):
            _t_block_start = time.perf_counter()

            _t_heuristic_start = time.perf_counter()
            if index_block not in self.single_block_heuristic_states:
                self.single_block_heuristic_states[index_block] = {
                    'accumulated_distance': 0,
                    'previous_modulated_input': None,
                    'cached_output': None,
                    'should_compute': True
                }

            block_state = self.single_block_heuristic_states[index_block]
            should_compute_block = force_compute

            

            if not force_compute and is_within_time_range and index_block >= self.single_block_cache_start:
                inp = hidden_states.clone()
                temb_ = temb.clone()
                norm_result = block.norm(inp, emb=temb_)
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp = norm_result[0]
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result

                if block_state['previous_modulated_input'] is not None:
                    rel_change = (
                        (modulated_inp - block_state['previous_modulated_input']).abs().mean()
                        / block_state['previous_modulated_input'].abs().mean()
                    ).cpu().item()
                    coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
                    rescale_func = np.poly1d(coefficients)
                    block_state['accumulated_distance'] += rescale_func(rel_change)

                    if block_state['accumulated_distance'] < self.single_block_rel_l1_thresh:
                        should_compute_block = False
                    else:
                        block_state['accumulated_distance'] = 0
                        should_compute_block = True
                else:
                    should_compute_block = True

                block_state['previous_modulated_input'] = modulated_inp.clone()
            else:
                should_compute_block = True
                if is_within_time_range and index_block >= self.single_block_cache_start:
                    inp = hidden_states.clone()
                    temb_ = temb.clone()
                    norm_result = block.norm(inp, emb=temb_)
                    if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                        modulated_inp = norm_result[0]
                    elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                        modulated_inp = norm_result[0]
                    else:
                        modulated_inp = norm_result
                    block_state['previous_modulated_input'] = modulated_inp.clone()

            # === 新增：累计 heuristic 耗时 ===
            _timings['single_blocks_heuristic_total'] += time.perf_counter() - _t_heuristic_start

            if should_compute_block:
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
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

                if is_within_time_range and index_block >= self.single_block_cache_start:
                    block_state['cached_output'] = hidden_states.clone()

                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )
            else:
                if block_state['cached_output'] is not None:
                    hidden_states = block_state['cached_output']

        _timings['single_blocks_loop_total'] = time.perf_counter() - _t_stb_loop_start

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        if self.cnt == self.num_steps:
            self.cnt = 0

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        # === 新增：总 heuristic & 总 forward ===
        _timings['heuristic_total'] = (
            _timings['transformer_blocks_heuristic_total'] +
            _timings['single_blocks_heuristic_total']
        )
        _timings['total_forward'] = time.perf_counter() - _t_forward_start

        # === 新增：打印统计结果 ===
        print("[TeaBlockCacheForward Timings]")
        for k, v in _timings.items():
            print(f"  {k}: {v:.6f}s")

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)