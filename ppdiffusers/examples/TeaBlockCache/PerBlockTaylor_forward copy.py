from typing import Any, Dict, Optional, Tuple, Union
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, logger, scale_lora_layers, unscale_lora_layers

# =================================================================
# 1. 泰勒预测引擎：核心数学逻辑
# =================================================================

def fallback_cache_init_step(max_order=2, first_enhance=1):
    """为单个块的泰勒预测系统初始化缓存结构。"""
    cache_dic = {'cache': {'hidden': {}}, 'max_order': max_order, 'first_enhance': first_enhance}
    current = {'step': 0, 'activated_steps': []}
    return cache_dic, current

def fallback_step_taylor_formula(cache_dic: Dict, current: Dict) -> Optional[paddle.Tensor]:
    """使用缓存的各阶导数（系数）通过泰勒公式进行预测。"""
    if len(current['activated_steps']) < 1 or len(cache_dic['cache']['hidden']) == 0:
        return None
    if current['step'] < cache_dic.get('first_enhance', 1):
        return None
    
    try:
        # 如果只有一个计算点，直接返回该点的值（0阶预测）
        if len(current['activated_steps']) < 2:
            return cache_dic['cache']['hidden'].get(0)

        x = current['step'] - current['activated_steps'][-1]
        output = cache_dic['cache']['hidden'][0]
        max_order = cache_dic.get('max_order', 2)
        
        for i in range(1, min(max_order, len(cache_dic['cache']['hidden']))):
            if i in cache_dic['cache']['hidden']:
                term = cache_dic['cache']['hidden'][i] * (x ** i)
                term = term / math.factorial(i)
                output = output + term
        
        if not paddle.isfinite(output).all():
            return cache_dic['cache']['hidden'].get(0)
        return output
    except Exception:
        return cache_dic['cache']['hidden'].get(0, None)

def fallback_step_derivative_approximation(cache_dic: Dict, feature: paddle.Tensor):
    """在一个块的完整计算后，用新输出更新其独立的泰勒缓存。"""
    try:
        max_order = cache_dic.get('max_order', 2)
        prev_val = feature
        for i in range(max_order):
            if i in cache_dic['cache']['hidden']:
                new_deriv = prev_val - cache_dic['cache']['hidden'][i]
                cache_dic['cache']['hidden'][i] = prev_val
                prev_val = new_deriv
            else:
                cache_dic['cache']['hidden'][i] = prev_val
                break
    except Exception as e:
        logger.warning(f"Error in derivative approximation: {e}, resetting cache.")
        cache_dic['cache']['hidden'] = {0: feature}

def apply_polynomial_rescale(rel_change: float) -> float:
    """应用多项式rescale函数"""
    coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
    rescale_func = np.poly1d(coefficients)
    return float(rescale_func(rel_change))

# =================================================================
# 2. 逐块泰勒预测策略的 Forward 函数
# =================================================================

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
        """
        逐块泰勒预测的缓存策略。
        保留了逐块启发式决策的框架，但将简单的缓存复用替换为更智能的泰勒预测。
        """
        # --- 多项式系数初始化 ---
        if not hasattr(self, "_poly_coeffs_tensor"):
            self._poly_coeffs_tensor = paddle.to_tensor(
                [4.98651651e02, -2.83781631e02, 5.58554382e01,
                -3.82021401e00, 2.64230861e-01],
                dtype=hidden_states.dtype,
            )
        
        # --- LoRA 处理 ---
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

        # --- 标准输入处理 ---
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

        # --- 泰勒缓存系统初始化与重置 ---
        if not hasattr(self, 'block_taylor_caches'):
            self.block_taylor_caches = {} # 存储每个块独立的泰勒缓存
            self.single_block_taylor_caches = {}

        is_within_time_range = self.step_start <= timestep <= self.step_end
        
        # Reset states at the beginning of generation
        if timestep == 1000 or self.cnt == 0:
            self.block_taylor_caches = {}
            self.single_block_taylor_caches = {}
            self.cnt = 0

        # Global step counter
        self.cnt += 1
        
        # Force computation at first and last steps
        force_compute = (self.cnt == 1 or self.cnt == self.num_steps)

        # --- PART 1: 对 `transformer_blocks` 应用逐块泰勒预测 ---
        for index_block, block in enumerate(self.transformer_blocks):
            # 1.1 初始化每个块独立的泰勒缓存系统
            if index_block not in self.block_taylor_caches:
                hs_cache, hs_curr = fallback_cache_init_step()
                enc_hs_cache, enc_hs_curr = fallback_cache_init_step()
                self.block_taylor_caches[index_block] = {
                    'accumulated_distance': 0.0,
                    'previous_modulated_input': None,
                    'taylor_hs': {'cache_dic': hs_cache, 'current': hs_curr},
                    'taylor_enc_hs': {'cache_dic': enc_hs_cache, 'current': enc_hs_curr},
                }
            
            block_cache = self.block_taylor_caches[index_block]

            # 1.2 启发式决策 (这部分逻辑保持不变)
            should_compute_block = force_compute
            if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
                try:
                    # Calculate modulated input (like TeaCache) for more accurate change detection
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
                    if block_cache['previous_modulated_input'] is not None:
                        hidden_dim = modulated_inp.shape[-1]
                        C = max(1, hidden_dim // 8) 
                        mod_head = modulated_inp[:, :, :C]
                        prev_head = block_cache['previous_modulated_input'][:, :, :C]
                        rel_change = paddle.linalg.norm(mod_head - prev_head, 1) / paddle.linalg.norm(prev_head, 1)

                        coeffs = self._poly_coeffs_tensor 
                        rescale = (((coeffs[0] * rel_change + coeffs[1]) *
                                    rel_change + coeffs[2]) *
                                    rel_change + coeffs[3]) * rel_change + coeffs[4]
                        block_cache['accumulated_distance'] += rescale

                        if block_cache['accumulated_distance'] < self.block_rel_l1_thresh:
                            should_compute_block = False
                        else:
                            block_cache['accumulated_distance'] = 0
                            should_compute_block = True
                    else:
                        should_compute_block = True
                    
                    # Update previous modulated input
                    block_cache['previous_modulated_input'] = modulated_inp.clone()
                except:
                    should_compute_block = True
            
            # 1.3 根据决策进行计算或预测
            if should_compute_block:
                # --- 计算路径 ---
                # 标记此步为计算步
                block_cache['taylor_hs']['current']['activated_steps'].append(block_cache['taylor_hs']['current']['step'])
                block_cache['taylor_enc_hs']['current']['activated_steps'].append(block_cache['taylor_enc_hs']['current']['step'])
                
                # 执行计算
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    new_encoder_hidden, new_hidden = paddle.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    new_encoder_hidden, new_hidden = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                # 用新结果更新此块的泰勒缓存
                fallback_step_derivative_approximation(block_cache['taylor_hs']['cache_dic'], new_hidden)
                fallback_step_derivative_approximation(block_cache['taylor_enc_hs']['cache_dic'], new_encoder_hidden)
                
                hidden_states, encoder_hidden_states = new_hidden, new_encoder_hidden
                
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
                # --- 预测路径 (核心修改点) ---
                predicted_hs = fallback_step_taylor_formula(block_cache['taylor_hs']['cache_dic'], block_cache['taylor_hs']['current'])
                predicted_enc_hs = fallback_step_taylor_formula(block_cache['taylor_enc_hs']['cache_dic'], block_cache['taylor_enc_hs']['current'])

                # 如果预测成功，则使用预测值；否则安全回退到上一次的计算值（0阶项）
                hidden_states = predicted_hs if predicted_hs is not None else block_cache['taylor_hs']['cache_dic']['cache']['hidden'][0]
                encoder_hidden_states = predicted_enc_hs if predicted_enc_hs is not None else block_cache['taylor_enc_hs']['cache_dic']['cache']['hidden'][0]
            
            # 统一更新此块内部的步数计数器
            block_cache['taylor_hs']['current']['step'] += 1
            block_cache['taylor_enc_hs']['current']['step'] += 1

        # Concatenate encoder and image hidden states
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        # --- PART 2: 对 `single_transformer_blocks` 应用逐块泰勒预测 ---
        for index_block, block in enumerate(self.single_transformer_blocks):
            # 2.1 初始化每个块独立的泰勒缓存系统
            if index_block not in self.single_block_taylor_caches:
                hs_cache, hs_curr = fallback_cache_init_step()
                self.single_block_taylor_caches[index_block] = {
                    'accumulated_distance': 0.0,
                    'previous_modulated_input': None,
                    'taylor_hs': {'cache_dic': hs_cache, 'current': hs_curr}
                }
            
            block_cache = self.single_block_taylor_caches[index_block]

            # 2.2 启发式决策 (逻辑保持不变)
            should_compute_block = force_compute
            if not force_compute and is_within_time_range and index_block >= self.single_block_cache_start:
                try:
                    # Calculate modulated input for single blocks (they have norm layer too)
                    inp = hidden_states
                    temb_ = temb
                    norm_result = block.norm(inp, emb=temb_)
                    # Handle different return formats safely
                    if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                    elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                        modulated_inp = norm_result[0]
                    else:
                        modulated_inp = norm_result
                    
                    # Apply heuristic for this specific block using modulated input
                    if block_cache['previous_modulated_input'] is not None:
                        # Calculate change in modulated input for this block
                        hidden_dim = modulated_inp.shape[-1]
                        C = max(1, hidden_dim // 8) 
                        mod_head = modulated_inp[:, :, :C]
                        prev_head = block_cache['previous_modulated_input'][:, :, :C]
                        rel_change = paddle.linalg.norm(mod_head - prev_head, 1) / paddle.linalg.norm(prev_head, 1)
                        
                        # Apply rescaling function
                        coeffs = self._poly_coeffs_tensor
                        rescale = (((coeffs[0] * rel_change + coeffs[1]) *
                                    rel_change + coeffs[2]) *
                                    rel_change + coeffs[3]) * rel_change + coeffs[4]
                        block_cache['accumulated_distance'] += rescale
                        
                        # Check if accumulated change exceeds threshold
                        if block_cache['accumulated_distance'] < self.single_block_rel_l1_thresh:
                            should_compute_block = False
                        else:
                            block_cache['accumulated_distance'] = 0
                            should_compute_block = True
                    else:
                        should_compute_block = True
                    
                    # Update previous modulated input
                    block_cache['previous_modulated_input'] = modulated_inp.clone()
                except:
                    should_compute_block = True

            # 2.3 根据决策进行计算或预测
            if should_compute_block:
                # --- 计算路径 ---
                block_cache['taylor_hs']['current']['activated_steps'].append(block_cache['taylor_hs']['current']['step'])
                
                # 执行计算
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    new_hidden = paddle.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    new_hidden = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                
                # 用新结果更新此块的泰勒缓存
                fallback_step_derivative_approximation(block_cache['taylor_hs']['cache_dic'], new_hidden)
                hidden_states = new_hidden
                
                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )
            else:
                # --- 预测路径 ---
                predicted_hs = fallback_step_taylor_formula(block_cache['taylor_hs']['cache_dic'], block_cache['taylor_hs']['current'])
                hidden_states = predicted_hs if predicted_hs is not None else block_cache['taylor_hs']['cache_dic']['cache']['hidden'][0]
            
            # 统一更新此块内部的步数计数器
            block_cache['taylor_hs']['current']['step'] += 1

        # Extract only the image hidden states
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        # Reset counter if we've reached the end
        if self.cnt == self.num_steps:
            self.cnt = 0

        # --- 后续处理 ---
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output) 