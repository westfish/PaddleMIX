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
    """
    x = current['step'] - current['activated_steps'][-1]
    output = 0

    for i in range(len(cache_dic['cache']['hidden'])):
        output += (1 / math.factorial(i)) * cache_dic['cache']['hidden'][i] * (x ** i)
    
    return output


def step_derivative_approximation(cache_dic: Dict, current: Dict, feature: paddle.Tensor):
    """
    Compute derivative approximation for Taylor expansion.
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


def block_taylor_predict(block_cache: Dict, current_step: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    ✅ 核心修正：Per-block Taylor expansion prediction
    这里真正实现了per-block级别的泰勒级数预测，而不是简单的缓存复用
    """
    if len(block_cache['activated_steps']) < 2:
        return None, None
    
    current_info = {
        'step': current_step,
        'activated_steps': block_cache['activated_steps']
    }
    
    try:
        predicted_hidden = None
        predicted_encoder = None
        
        # 使用泰勒级数预测hidden states
        if 'hidden' in block_cache['cache'] and len(block_cache['cache']['hidden']) > 0:
            predicted_hidden = step_taylor_formula(
                {'cache': {'hidden': block_cache['cache']['hidden']}}, 
                current_info
            )
        
        # 使用泰勒级数预测encoder states  
        if 'encoder' in block_cache['cache'] and len(block_cache['cache']['encoder']) > 0:
            predicted_encoder = step_taylor_formula(
                {'cache': {'hidden': block_cache['cache']['encoder']}}, 
                current_info
            )
        
        # 数值稳定性检查
        if predicted_hidden is not None and not paddle.isfinite(predicted_hidden).all():
            predicted_hidden = None
        if predicted_encoder is not None and not paddle.isfinite(predicted_encoder).all():
            predicted_encoder = None
            
        return predicted_hidden, predicted_encoder
    
    except Exception:
        return None, None


def update_block_taylor_cache(block_cache: Dict, current_step: int, hidden_output: paddle.Tensor, encoder_output: paddle.Tensor = None):
    """
    ✅ 更新per-block的泰勒展开缓存
    """
    if 'cache' not in block_cache:
        block_cache['cache'] = {'hidden': {}, 'encoder': {}}
    if 'activated_steps' not in block_cache:
        block_cache['activated_steps'] = []
    
    block_cache['activated_steps'].append(current_step)
    
    if len(block_cache['activated_steps']) >= 2:
        current_info = {
            'step': current_step,
            'activated_steps': block_cache['activated_steps']
        }
        
        # 更新hidden states的泰勒系数
        hidden_cache_dict = {
            'max_order': 3,
            'first_enhance': 2,
            'cache': {'hidden': block_cache['cache']['hidden']}
        }
        step_derivative_approximation(hidden_cache_dict, current_info, hidden_output)
        block_cache['cache']['hidden'] = hidden_cache_dict['cache']['hidden']
        
        # 更新encoder states的泰勒系数
        if encoder_output is not None:
            encoder_cache_dict = {
                'max_order': 3,
                'first_enhance': 2,
                'cache': {'hidden': block_cache['cache']['encoder']}
            }
            step_derivative_approximation(encoder_cache_dict, current_info, encoder_output)
            block_cache['cache']['encoder'] = encoder_cache_dict['cache']['hidden']


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
        ✅ 修正版本：真正在per-block层面实现泰勒级数预测
        
        主要修正：
        1. 不再使用简单的cached_output复用
        2. 在每个block层面实现泰勒级数预测
        3. 将简单的缓存复用升级为数学严谨的泰勒预测
        """
        
        # [前面的初始化代码保持不变]
        if not hasattr(self, "_poly_coeffs_tensor"):
            self._poly_coeffs_tensor = paddle.to_tensor(
                [4.98651651e02, -2.83781631e02, 5.58554382e01,
                -3.82021401e00, 2.64230861e-01],
                dtype=hidden_states.dtype,
            )
            
        if not hasattr(self, 'taylor_cache_system'):
            self.taylor_cache_system = {
                'max_order': 3, 
                'first_enhance': 2,
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
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
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
            self.taylor_cache_system = {
                'max_order': 3,
                'first_enhance': 2,
                'cache': {'hidden': {}},
                'activated_steps': [],
                'step_counter': 0
            }
            self.cnt = 0

        self.taylor_cache_system['step_counter'] = self.cnt
        self.cnt += 1
        force_compute = (self.cnt == 1 or self.cnt == self.num_steps)

        # 全局泰勒预测（保持不变）
        use_taylor_prediction = False
        if (len(self.taylor_cache_system['activated_steps']) >= 2 and 
            not force_compute and is_within_time_range):
            
            current_info = {
                'step': self.cnt,
                'activated_steps': self.taylor_cache_system['activated_steps']
            }
            
            if len(self.taylor_cache_system['cache']['hidden']) > 0:
                try:
                    predicted_hidden_states = step_taylor_formula(self.taylor_cache_system, current_info)
                    if paddle.isfinite(predicted_hidden_states).all():
                        hidden_states = predicted_hidden_states
                        use_taylor_prediction = True
                except:
                    use_taylor_prediction = False

        # ✅ 核心修正：Transformer blocks 处理
        for index_block, block in enumerate(self.transformer_blocks):
            if index_block not in self.block_heuristic_states:
                self.block_heuristic_states[index_block] = {
                    'accumulated_distance': 0,
                    'previous_modulated_input': None,
                    # ❌ 移除简单缓存：'cached_output': None, 'cached_encoder_output': None,
                    'should_compute': True,
                    # ✅ 添加真正的泰勒缓存
                    'taylor_cache': {
                        'cache': {'hidden': {}, 'encoder': {}}, 
                        'activated_steps': []
                    }
                }
            
            block_state = self.block_heuristic_states[index_block]
            should_compute_block = force_compute or use_taylor_prediction
            
            # 启发式判断逻辑（保持不变）
            if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
                inp = hidden_states
                temb_ = temb
                norm_result = block.norm1(inp, emb=temb_)
                
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result
                
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
                
                block_state['previous_modulated_input'] = modulated_inp.clone()

            if should_compute_block:
                # 实际计算block
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                
                # ✅ 更新泰勒缓存（不是简单缓存）
                if is_within_time_range and index_block >= self.block_cache_start:
                    update_block_taylor_cache(
                        block_state['taylor_cache'],
                        self.cnt,
                        hidden_states.clone(),
                        encoder_hidden_states.clone()
                    )
                
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
                # ✅ 核心修正：使用泰勒级数预测，不是简单的缓存复用
                predicted_hidden, predicted_encoder = block_taylor_predict(
                    block_state['taylor_cache'], 
                    self.cnt
                )
                
                if predicted_hidden is not None and predicted_encoder is not None:
                    # ✅ 使用泰勒预测的结果
                    hidden_states = predicted_hidden
                    encoder_hidden_states = predicted_encoder
                    print(f"✅ Block {index_block}: 使用泰勒级数预测")
                else:
                    # 如果泰勒预测失败，强制计算
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                    print(f"⚠️ Block {index_block}: 泰勒预测失败，强制计算")

        # 更新全局泰勒缓存
        if not use_taylor_prediction and is_within_time_range:
            self.taylor_cache_system['activated_steps'].append(self.cnt)
            current_info = {
                'step': self.cnt,
                'activated_steps': self.taylor_cache_system['activated_steps']
            }
            
            if len(self.taylor_cache_system['activated_steps']) >= 2:
                step_derivative_approximation(self.taylor_cache_system, current_info, hidden_states.clone())

        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        # ✅ Single transformer blocks 同样的修正
        for index_block, block in enumerate(self.single_transformer_blocks):
            if index_block not in self.single_block_heuristic_states:
                self.single_block_heuristic_states[index_block] = {
                    'accumulated_distance': 0,
                    'previous_modulated_input': None,
                    # ❌ 移除简单缓存：'cached_output': None,
                    'should_compute': True,
                    # ✅ 添加泰勒缓存
                    'taylor_cache': {
                        'cache': {'hidden': {}}, 
                        'activated_steps': []
                    }
                }
            
            block_state = self.single_block_heuristic_states[index_block]
            should_compute_block = force_compute or use_taylor_prediction
            
            # 启发式判断（保持不变）
            if not force_compute and is_within_time_range and index_block >= self.single_block_cache_start:
                inp = hidden_states
                temb_ = temb
                norm_result = block.norm(inp, emb=temb_)
                
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result
                
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
                    
                    if block_state['accumulated_distance'] < self.single_block_rel_l1_thresh:
                        should_compute_block = False
                    else:
                        block_state['accumulated_distance'] = 0
                        should_compute_block = True
                
                block_state['previous_modulated_input'] = modulated_inp.clone()

            if should_compute_block:
                # 实际计算single block
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                
                # ✅ 更新single block的泰勒缓存
                if is_within_time_range and index_block >= self.single_block_cache_start:
                    block_state['taylor_cache']['activated_steps'].append(self.cnt)
                    
                    if len(block_state['taylor_cache']['activated_steps']) >= 2:
                        current_info = {
                            'step': self.cnt,
                            'activated_steps': block_state['taylor_cache']['activated_steps']
                        }
                        single_cache_dict = {
                            'max_order': 3,
                            'first_enhance': 2,
                            'cache': {'hidden': block_state['taylor_cache']['cache']['hidden']}
                        }
                        step_derivative_approximation(single_cache_dict, current_info, hidden_states.clone())
                        block_state['taylor_cache']['cache']['hidden'] = single_cache_dict['cache']['hidden']
                
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )
            else:
                # ✅ 使用single block的泰勒预测
                if len(block_state['taylor_cache']['activated_steps']) >= 2:
                    current_info = {
                        'step': self.cnt,
                        'activated_steps': block_state['taylor_cache']['activated_steps']
                    }
                    
                    try:
                        predicted_hidden = step_taylor_formula(
                            {'cache': {'hidden': block_state['taylor_cache']['cache']['hidden']}}, 
                            current_info
                        )
                        
                        if paddle.isfinite(predicted_hidden).all():
                            hidden_states = predicted_hidden
                            print(f"✅ Single Block {index_block}: 使用泰勒级数预测")
                        else:
                            # 泰勒预测失败，强制计算
                            hidden_states = block(
                                hidden_states=hidden_states,
                                temb=temb,
                                image_rotary_emb=image_rotary_emb,
                                joint_attention_kwargs=joint_attention_kwargs,
                            )
                            print(f"⚠️ Single Block {index_block}: 泰勒预测失败，强制计算")
                    except:
                        # 强制计算
                        hidden_states = block(
                            hidden_states=hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                        print(f"❌ Single Block {index_block}: 泰勒预测异常，强制计算")

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        if self.cnt == self.num_steps:
            self.cnt = 0

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output) 