from typing import Any, Dict, Optional, Tuple, Union
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import  Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, logger, scale_lora_layers, unscale_lora_layers


def step_taylor_formula(cache_dic: Dict, current: Dict) -> paddle.Tensor: 
    """
    Compute Taylor expansion prediction.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    output = 0

    for i in range(len(cache_dic['cache']['hidden'])):
        output += (1 / math.factorial(i)) * cache_dic['cache']['hidden'][i] * (x ** i)
    
    return output


def step_derivative_approximation(cache_dic: Dict, current: Dict, feature: paddle.Tensor):
    """
    Compute derivative approximation for Taylor expansion.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Current feature tensor
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache']['hidden'].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache']['hidden'][i]) / difference_distance
        else:
            break
    
    cache_dic['cache']['hidden'] = updated_taylor_factors


def TeaBlockCacheTaylorForward(
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
        A hybrid caching strategy that combines TeaBlockCache per-block heuristics 
        with Taylor expansion cache reuse mechanism.
        
        This method integrates:
        1. Per-block heuristic caching from TeaBlockCache
        2. Taylor expansion prediction for improved cache reuse
        3. Time and block dimension partitioning
        """
        if not hasattr(self, "_poly_coeffs_tensor"):
            self._poly_coeffs_tensor = paddle.to_tensor(
                [4.98651651e02, -2.83781631e02, 5.58554382e01,
                -3.82021401e00, 2.64230861e-01],
                dtype=hidden_states.dtype,
            )
            
        # Initialize Taylor cache system
        if not hasattr(self, 'taylor_cache_system'):
            self.taylor_cache_system = {
                'max_order': 3,  # Maximum Taylor expansion order
                'first_enhance': 2,  # First step to start using derivatives
                'cache': {'hidden': {}},
                'activated_steps': [],
                'step_counter': 0
            }
            
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
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

        # Initialize per-block heuristic states if not exists
        if not hasattr(self, 'block_heuristic_states'):
            self.block_heuristic_states = {}
            self.single_block_heuristic_states = {}
            
        # Check if we're in cache-enabled time range
        is_within_time_range = self.step_start <= timestep <= self.step_end
        
        # Reset states at the beginning of generation
        if timestep == 1000 or self.cnt == 0:
            self.block_heuristic_states = {}
            self.single_block_heuristic_states = {}
            self.taylor_cache_system = {
                'max_order': 3,
                'first_enhance': 2,
                'cache': {'hidden': {}},
                'activated_steps': [],
                'step_counter': 0
            }
            self.cnt = 0

        # Update Taylor cache system step counter
        self.taylor_cache_system['step_counter'] = self.cnt
        
        # Global step counter
        self.cnt += 1
        
        # Force computation at first and last steps
        force_compute = (self.cnt == 1 or self.cnt == self.num_steps)

        # Taylor expansion for global hidden states prediction
        use_taylor_prediction = False
        if (len(self.taylor_cache_system['activated_steps']) >= 2 and 
            not force_compute and is_within_time_range):
            
            current_info = {
                'step': self.cnt,
                'activated_steps': self.taylor_cache_system['activated_steps']
            }
            
            # Try to use Taylor prediction for the entire hidden states
            if len(self.taylor_cache_system['cache']['hidden']) > 0:
                try:
                    predicted_hidden_states = step_taylor_formula(self.taylor_cache_system, current_info)
                    # Use predicted states if they have reasonable magnitude
                    if paddle.isfinite(predicted_hidden_states).all():
                        hidden_states = predicted_hidden_states
                        use_taylor_prediction = True
                except:
                    use_taylor_prediction = False

        # Process transformer blocks with per-block heuristics
        for index_block, block in enumerate(self.transformer_blocks):
            # Initialize block state if not exists
            if index_block not in self.block_heuristic_states:
                self.block_heuristic_states[index_block] = {
                    'accumulated_distance': 0,
                    'previous_modulated_input': None,
                    'cached_output': None,
                    'cached_encoder_output': None,
                    'should_compute': True,
                    'taylor_cache': {'cache': {'hidden': {}}, 'activated_steps': []}
                }
            
            block_state = self.block_heuristic_states[index_block]
            
            # Determine if this block should be computed
            should_compute_block = force_compute or use_taylor_prediction
            
            if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
                # Calculate modulated input for more accurate change detection
                inp = hidden_states
                temb_ = temb
                norm_result = block.norm1(inp, emb=temb_)
                
                # Handle different return formats safely
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result
                
                # Apply heuristic for this specific block using modulated input
                if block_state['previous_modulated_input'] is not None and not use_taylor_prediction:
                    hidden_dim = modulated_inp.shape[-1]
                    C = max(1, hidden_dim // 8) 
                    mod_head = modulated_inp[:, :, :C]
                    prev_head = block_state['previous_modulated_input'][:, :, :C]
                    rel_change = paddle.linalg.norm(mod_head - prev_head, 1) / paddle.linalg.norm(prev_head, 1)

                    coeffs = self._poly_coeffs_tensor 
                    rescale = (((coeffs[0] * rel_change + coeffs[1]) *
                                rel_change + coeffs[2]) *
                                rel_change + coeffs[3]) * rel_change + coeffs[4]
                    block_state['accumulated_distance'] += rescale

                    if block_state['accumulated_distance'] < self.block_rel_l1_thresh:
                        should_compute_block = False
                    else:
                        block_state['accumulated_distance'] = 0
                        should_compute_block = True
                else:
                    should_compute_block = True
                
                # Update previous modulated input
                block_state['previous_modulated_input'] = modulated_inp.clone()
            else:
                should_compute_block = True
                if is_within_time_range and index_block >= self.block_cache_start:
                    # Still compute modulated input for future comparisons
                    inp = hidden_states
                    temb_ = temb
                    norm_result = block.norm1(inp, emb=temb_)
                    
                    if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                    elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                        modulated_inp = norm_result[0]
                    else:
                        modulated_inp = norm_result
                    block_state['previous_modulated_input'] = modulated_inp.clone()

            if should_compute_block:
                # Compute the block
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
                
                # Cache the outputs
                if is_within_time_range and index_block >= self.block_cache_start:
                    block_state['cached_output'] = hidden_states.clone()
                    block_state['cached_encoder_output'] = encoder_hidden_states.clone()
                
                # controlnet residual
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
                # Use cached outputs
                if (block_state['cached_output'] is not None and 
                    block_state['cached_encoder_output'] is not None):
                    hidden_states = block_state['cached_output']
                    encoder_hidden_states = block_state['cached_encoder_output']

        # Update Taylor expansion cache after processing transformer blocks
        if not use_taylor_prediction and is_within_time_range:
            self.taylor_cache_system['activated_steps'].append(self.cnt)
            current_info = {
                'step': self.cnt,
                'activated_steps': self.taylor_cache_system['activated_steps']
            }
            
            if len(self.taylor_cache_system['activated_steps']) >= 2:
                step_derivative_approximation(self.taylor_cache_system, current_info, hidden_states.clone())

        # Concatenate encoder and image hidden states
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        # Process single transformer blocks with per-block heuristics
        for index_block, block in enumerate(self.single_transformer_blocks):
            # Initialize block state if not exists
            if index_block not in self.single_block_heuristic_states:
                self.single_block_heuristic_states[index_block] = {
                    'accumulated_distance': 0,
                    'previous_modulated_input': None,
                    'cached_output': None,
                    'should_compute': True
                }
            
            block_state = self.single_block_heuristic_states[index_block]
            
            # Determine if this block should be computed
            should_compute_block = force_compute or use_taylor_prediction
            
            if not force_compute and is_within_time_range and index_block >= self.single_block_cache_start:
                # Calculate modulated input for single blocks
                inp = hidden_states
                temb_ = temb
                norm_result = block.norm(inp, emb=temb_)
                
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result
                
                # Apply heuristic for this specific block using modulated input
                if block_state['previous_modulated_input'] is not None and not use_taylor_prediction:
                    hidden_dim = modulated_inp.shape[-1]
                    C = max(1, hidden_dim // 8) 
                    mod_head = modulated_inp[:, :, :C]
                    prev_head = block_state['previous_modulated_input'][:, :, :C]
                    rel_change = paddle.linalg.norm(mod_head - prev_head, 1) / paddle.linalg.norm(prev_head, 1)
                    
                    # Apply rescaling function
                    coeffs = self._poly_coeffs_tensor
                    rescale = (((coeffs[0] * rel_change + coeffs[1]) *
                                rel_change + coeffs[2]) *
                                rel_change + coeffs[3]) * rel_change + coeffs[4]
                    block_state['accumulated_distance'] += rescale
                    
                    # Check if accumulated change exceeds threshold
                    if block_state['accumulated_distance'] < self.single_block_rel_l1_thresh:
                        should_compute_block = False
                    else:
                        block_state['accumulated_distance'] = 0
                        should_compute_block = True
                else:
                    should_compute_block = True
                
                # Update previous modulated input
                block_state['previous_modulated_input'] = modulated_inp.clone()
            else:
                should_compute_block = True
                if is_within_time_range and index_block >= self.single_block_cache_start:
                    # Still compute modulated input for future comparisons
                    inp = hidden_states
                    temb_ = temb
                    norm_result = block.norm(inp, emb=temb_)
                    
                    if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                    elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                        modulated_inp = norm_result[0]
                    else:
                        modulated_inp = norm_result
                    block_state['previous_modulated_input'] = modulated_inp.clone()

            if should_compute_block:
                # Compute the block
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
                
                # Cache the output
                if is_within_time_range and index_block >= self.single_block_cache_start:
                    block_state['cached_output'] = hidden_states.clone()
                
                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )
            else:
                # Use cached output
                if block_state['cached_output'] is not None:
                    hidden_states = block_state['cached_output']

        # Extract only the image hidden states
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # Reset counter if we've reached the end
        if self.cnt == self.num_steps:
            self.cnt = 0

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output) 