from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, logger, scale_lora_layers, unscale_lora_layers
import math
import collections

class SimplePolynomialRegressor:
    """简单的多项式回归器，不依赖sklearn"""
    
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
        
    def _create_polynomial_features(self, X):
        """创建多项式特征"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        features = [np.ones(X.shape[0])]  # 常数项
        
        for d in range(1, self.degree + 1):
            features.append((X[:, 0] ** d))
        
        return np.column_stack(features)
    
    def fit(self, X, y):
        """拟合多项式"""
        try:
            X_poly = self._create_polynomial_features(X)
            # 使用最小二乘法求解：(X^T X)^(-1) X^T y
            XTX = np.dot(X_poly.T, X_poly)
            XTy = np.dot(X_poly.T, y)
            
            # 添加正则化避免奇异矩阵
            regularization = 1e-6 * np.eye(XTX.shape[0])
            self.coefficients = np.linalg.solve(XTX + regularization, XTy)
            
        except np.linalg.LinAlgError:
            # 如果求解失败，使用简单线性拟合
            if len(X) > 1:
                self.coefficients = np.array([np.mean(y), (y[-1] - y[0]) / (X[-1] - X[0] + 1e-8)])
            else:
                self.coefficients = np.array([np.mean(y), 0])
    
    def predict(self, X):
        """预测"""
        if self.coefficients is None:
            return np.zeros(len(X))
        
        try:
            X_poly = self._create_polynomial_features(X)
            return np.dot(X_poly, self.coefficients)
        except:
            return np.zeros(len(X))

class SimpleInputOutputPredictor:
    """简化的输入输出预测器"""
    
    def __init__(self, max_samples=30):
        self.max_samples = max_samples
        
        # 存储输入变化和输出变化的样本
        self.input_changes = collections.deque(maxlen=max_samples)
        self.output_changes = collections.deque(maxlen=max_samples)
        
        # 不同的预测模型
        self.models = {
            'linear': SimplePolynomialRegressor(degree=1),
            'quadratic': SimplePolynomialRegressor(degree=2),
            'cubic': SimplePolynomialRegressor(degree=3)
        }
        
        # 模型评分
        self.model_scores = {name: {'mse': float('inf'), 'count': 0} 
                           for name in self.models.keys()}
        
        self.current_best_model = 'quadratic'
        
    def add_sample(self, input_change, output_change):
        """添加训练样本"""
        self.input_changes.append(input_change)
        self.output_changes.append(output_change)
        
        # 如果样本足够，更新模型
        if len(self.input_changes) >= 5:
            self._update_models()
    
    def _update_models(self):
        """更新所有预测模型"""
        if len(self.input_changes) < 3:
            return
            
        X = np.array(self.input_changes)
        y = np.array(self.output_changes)
        
        # 更新每个模型并评估
        for model_name, model in self.models.items():
            try:
                # 训练模型
                model.fit(X, y)
                
                # 评估模型（使用最近几个样本）
                if len(X) >= 5:
                    recent_X = X[-5:]
                    recent_y = y[-5:]
                    pred_y = model.predict(recent_X)
                    mse = np.mean((pred_y - recent_y) ** 2)
                    
                    # 更新评分（指数移动平均）
                    if self.model_scores[model_name]['count'] == 0:
                        self.model_scores[model_name]['mse'] = mse
                    else:
                        alpha = 0.3  # 学习率
                        self.model_scores[model_name]['mse'] = (
                            alpha * mse + (1 - alpha) * self.model_scores[model_name]['mse']
                        )
                    
                    self.model_scores[model_name]['count'] += 1
                    
            except Exception:
                # 模型训练失败，给予惩罚
                self.model_scores[model_name]['mse'] = float('inf')
        
        # 选择最佳模型
        self._select_best_model()
    
    def _select_best_model(self):
        """选择当前表现最好的模型"""
        best_score = float('inf')
        best_model = self.current_best_model
        
        for model_name, score_info in self.model_scores.items():
            if score_info['count'] > 0 and score_info['mse'] < best_score:
                best_score = score_info['mse']
                best_model = model_name
        
        self.current_best_model = best_model
    
    def predict_output_change(self, input_change):
        """预测给定输入变化对应的输出变化"""
        if len(self.input_changes) < 3:
            # 样本不足时，使用启发式方法
            return self._heuristic_prediction(input_change)
        
        try:
            model = self.models[self.current_best_model]
            pred = model.predict(np.array([input_change]))[0]
            return max(0, pred)  # 输出变化不能为负
        except:
            return self._heuristic_prediction(input_change)
    
    def _heuristic_prediction(self, input_change):
        """启发式预测（当模型不可用时）"""
        if len(self.input_changes) == 0:
            # 完全没有历史数据，使用保守估计
            return input_change * 0.8
        
        # 使用历史数据的简单比例
        input_list = list(self.input_changes)
        output_list = list(self.output_changes)
        
        if len(input_list) > 0:
            # 计算平均比例
            ratios = [o / (i + 1e-8) for i, o in zip(input_list, output_list)]
            avg_ratio = np.mean(ratios)
            return input_change * avg_ratio
        else:
            return input_change * 0.5
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'current_best': self.current_best_model,
            'scores': {k: v for k, v in self.model_scores.items()},
            'sample_count': len(self.input_changes)
        }

class SimpleAdaptiveThresholdManager:
    """简化的自适应阈值管理"""
    
    def __init__(self, initial_thresh=0.01):
        self.base_threshold = initial_thresh
        self.current_threshold = initial_thresh
        
        # 性能监控
        self.cache_hit_history = collections.deque(maxlen=20)
        self.prediction_error_history = collections.deque(maxlen=10)
        
        # 自适应参数
        self.adaptation_rate = 0.1
        self.target_hit_rate = 0.7
        
    def update(self, was_cached, prediction_error=None):
        """更新阈值"""
        self.cache_hit_history.append(1.0 if was_cached else 0.0)
        
        if prediction_error is not None:
            self.prediction_error_history.append(abs(prediction_error))
        
        # 基于缓存命中率调整
        if len(self.cache_hit_history) >= 8:
            current_hit_rate = np.mean(self.cache_hit_history)
            
            if current_hit_rate < self.target_hit_rate - 0.15:
                # 命中率太低，降低阈值（更容易缓存）
                self.current_threshold *= (1 - self.adaptation_rate)
            elif current_hit_rate > self.target_hit_rate + 0.15:
                # 命中率太高，提高阈值（更严格）
                self.current_threshold *= (1 + self.adaptation_rate)
        
        # 基于预测误差调整
        if len(self.prediction_error_history) >= 5:
            avg_error = np.mean(self.prediction_error_history)
            if avg_error > 0.1:  # 预测误差较大
                self.current_threshold *= 1.1  # 提高阈值，更保守
        
        # 保持合理范围
        self.current_threshold = np.clip(self.current_threshold, 0.001, 0.2)
    
    def get_threshold(self, timestep=None):
        """获取当前阈值"""
        threshold = self.current_threshold
        
        # 时间步权重（早期和晚期更重要）
        if timestep is not None:
            normalized_timestep = timestep / 1000.0
            # U形曲线：开始和结束时更重要
            time_weight = 1.0 + 0.4 * (2 * abs(normalized_timestep - 0.5))
            threshold /= time_weight
        
        return threshold

class SimpleSmartBlockCache:
    """简化的智能块缓存"""
    
    def __init__(self, block_index):
        self.block_index = block_index
        
        # 核心组件
        self.predictor = SimpleInputOutputPredictor()
        self.threshold_manager = SimpleAdaptiveThresholdManager()
        
        # 缓存状态
        self.previous_modulated_input = None
        self.previous_output = None
        self.previous_encoder_output = None
        self.cached_output = None
        self.cached_encoder_output = None
        
        # 累积距离（TeaCache原理）
        self.accumulated_distance = 0
        
        # 性能统计
        self.compute_count = 0
        self.cache_count = 0
        self.total_prediction_error = 0
    
    def should_compute(self, modulated_input, force_compute=False, timestep=None):
        """智能决策是否需要重新计算"""
        if force_compute:
            self.compute_count += 1
            return True, "forced"
        
        if self.previous_modulated_input is None:
            self.previous_modulated_input = modulated_input.clone()
            self.compute_count += 1
            return True, "first_time"
        
        # 计算输入变化
        input_change = self._calculate_input_change(modulated_input)
        
        # 预测输出变化
        predicted_output_change = self.predictor.predict_output_change(input_change)
        
        # 累积预测的输出变化
        self.accumulated_distance += predicted_output_change
        
        # 获取动态阈值
        current_threshold = self.threshold_manager.get_threshold(timestep)
        
        # 决策
        should_compute = self.accumulated_distance >= current_threshold
        
        if should_compute:
            self.accumulated_distance = 0
            self.compute_count += 1
            reason = f"pred_change_{predicted_output_change:.4f}_>_thresh_{current_threshold:.4f}"
        else:
            self.cache_count += 1
            reason = f"pred_change_{predicted_output_change:.4f}_<_thresh_{current_threshold:.4f}"
        
        # 更新previous input
        self.previous_modulated_input = modulated_input.clone()
        
        return should_compute, reason
    
    def _calculate_input_change(self, current_input):
        """计算输入变化量（相对L1变化）"""
        if self.previous_modulated_input is None:
            return 0.0
        
        rel_change = (
            (current_input - self.previous_modulated_input).abs().mean() 
            / (self.previous_modulated_input.abs().mean() + 1e-8)
        ).cpu().item()
        
        return rel_change
    
    def update_with_actual_output(self, actual_output, actual_encoder_output=None):
        """用实际输出更新预测模型"""
        if self.previous_output is not None:
            # 计算实际输出变化
            actual_output_change = (
                (actual_output - self.previous_output).abs().mean()
                / (self.previous_output.abs().mean() + 1e-8)
            ).cpu().item()
            
            # 获取对应的输入变化
            if hasattr(self, '_last_input_change'):
                input_change = self._last_input_change
                
                # 添加训练样本
                self.predictor.add_sample(input_change, actual_output_change)
                
                # 计算预测误差
                predicted_change = self.predictor.predict_output_change(input_change)
                prediction_error = abs(predicted_change - actual_output_change)
                self.total_prediction_error += prediction_error
                
                # 更新阈值管理器
                self.threshold_manager.update(was_cached=False, prediction_error=prediction_error)
        
        # 更新previous outputs
        self.previous_output = actual_output.clone()
        if actual_encoder_output is not None:
            self.previous_encoder_output = actual_encoder_output.clone()
        
        # 缓存当前输出
        self.cached_output = actual_output.clone()
        if actual_encoder_output is not None:
            self.cached_encoder_output = actual_encoder_output.clone()
    
    def get_cache_statistics(self):
        """获取缓存统计信息"""
        total_ops = self.compute_count + self.cache_count
        avg_prediction_error = (
            self.total_prediction_error / max(1, self.compute_count)
        )
        
        return {
            'cache_hit_rate': self.cache_count / max(1, total_ops),
            'total_operations': total_ops,
            'average_prediction_error': avg_prediction_error,
            'current_threshold': self.threshold_manager.current_threshold,
            'predictor_info': self.predictor.get_model_info()
        }

def TeaCacheIntelligentSimpleForward(
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
    简化版本的智能TeaCache前向传播（不依赖sklearn）
    
    核心功能：
    1. 学习输入变化到输出变化的映射关系（使用简单多项式回归）
    2. 基于预测的输出变化决定是否缓存
    3. 在线更新预测模型和阈值
    """
    
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
        logger.warning("Passing `txt_ids` 3d torch.Tensor is deprecated.")
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning("Passing `img_ids` 3d torch.Tensor is deprecated.")
        img_ids = img_ids[0]

    ids = paddle.concat((txt_ids, img_ids), axis=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # 初始化简单智能缓存管理器
    if not hasattr(self, 'simple_intelligent_caches'):
        self.simple_intelligent_caches = {}
        self.simple_intelligent_single_caches = {}
    
    is_within_time_range = self.step_start <= timestep <= self.step_end
    
    if timestep == 1000 or self.cnt == 0:
        self.simple_intelligent_caches = {}
        self.simple_intelligent_single_caches = {}
        self.cnt = 0

    self.cnt += 1
    force_compute = (self.cnt == 1 or self.cnt == self.num_steps)

    # 处理transformer blocks
    for index_block, block in enumerate(self.transformer_blocks):
        if index_block not in self.simple_intelligent_caches:
            self.simple_intelligent_caches[index_block] = SimpleSmartBlockCache(index_block)
        
        cache_manager = self.simple_intelligent_caches[index_block]
        should_compute_block = force_compute
        
        if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
            # 计算modulated input
            inp = hidden_states.clone()
            temb_ = temb.clone()
            norm_result = block.norm1(inp, emb=temb_)
            
            if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
            elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                modulated_inp = norm_result[0]
            else:
                modulated_inp = norm_result
            
            # 智能决策
            should_compute_block, reason = cache_manager.should_compute(
                modulated_inp, force_compute, timestep.cpu().item()
            )
            
            # 保存输入变化用于后续学习
            if hasattr(cache_manager, '_calculate_input_change'):
                cache_manager._last_input_change = cache_manager._calculate_input_change(modulated_inp)
        else:
            if is_within_time_range and index_block >= self.block_cache_start:
                # 更新modulated input但不做决策
                inp = hidden_states.clone()
                temb_ = temb.clone()
                norm_result = block.norm1(inp, emb=temb_)
                
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result
                    
                cache_manager.previous_modulated_input = modulated_inp.clone()

        if should_compute_block:
            # 实际计算
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
            
            # 用实际输出更新预测模型
            if is_within_time_range and index_block >= self.block_cache_start:
                cache_manager.update_with_actual_output(hidden_states, encoder_hidden_states)
            
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
            # 使用缓存
            if (cache_manager.cached_output is not None and 
                cache_manager.cached_encoder_output is not None):
                hidden_states = cache_manager.cached_output
                encoder_hidden_states = cache_manager.cached_encoder_output
                
                # 更新阈值管理器（表示使用了缓存）
                cache_manager.threshold_manager.update(was_cached=True)

    # 拼接encoder和image hidden states
    hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

    # 处理single transformer blocks（简化版本，逻辑类似）
    for index_block, block in enumerate(self.single_transformer_blocks):
        if index_block not in self.simple_intelligent_single_caches:
            self.simple_intelligent_single_caches[index_block] = SimpleSmartBlockCache(index_block)
        
        cache_manager = self.simple_intelligent_single_caches[index_block]
        should_compute_block = force_compute
        
        if not force_compute and is_within_time_range and index_block >= self.single_block_cache_start:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            norm_result = block.norm(inp, emb=temb_)
            
            if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
            elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                modulated_inp = norm_result[0]
            else:
                modulated_inp = norm_result
            
            should_compute_block, _ = cache_manager.should_compute(
                modulated_inp, force_compute, timestep.cpu().item()
            )
            
            if hasattr(cache_manager, '_calculate_input_change'):
                cache_manager._last_input_change = cache_manager._calculate_input_change(modulated_inp)
        else:
            if is_within_time_range and index_block >= self.single_block_cache_start:
                inp = hidden_states.clone()
                temb_ = temb.clone()
                norm_result = block.norm(inp, emb=temb_)
                
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result
                    
                cache_manager.previous_modulated_input = modulated_inp.clone()

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
                cache_manager.update_with_actual_output(hidden_states)
            
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
        else:
            if cache_manager.cached_output is not None:
                hidden_states = cache_manager.cached_output
                cache_manager.threshold_manager.update(was_cached=True)

    # 提取image hidden states
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

def get_simple_intelligent_cache_summary(model):
    """获取简单智能缓存的性能摘要"""
    summary = {
        'transformer_blocks': {},
        'single_transformer_blocks': {},
        'overall_stats': {
            'total_cache_hit_rate': 0,
            'average_prediction_error': 0,
            'total_operations': 0,
            'best_models': {}
        }
    }
    
    total_hit_rate = 0
    total_prediction_error = 0
    total_ops = 0
    block_count = 0
    model_votes = {}
    
    # 分析transformer blocks
    if hasattr(model, 'simple_intelligent_caches'):
        for block_idx, cache_manager in model.simple_intelligent_caches.items():
            stats = cache_manager.get_cache_statistics()
            summary['transformer_blocks'][block_idx] = stats
            
            total_hit_rate += stats['cache_hit_rate']
            total_prediction_error += stats['average_prediction_error']
            total_ops += stats['total_operations']
            block_count += 1
            
            # 统计最佳模型
            best_model = stats['predictor_info']['current_best']
            model_votes[best_model] = model_votes.get(best_model, 0) + 1
    
    # 分析single transformer blocks
    if hasattr(model, 'simple_intelligent_single_caches'):
        for block_idx, cache_manager in model.simple_intelligent_single_caches.items():
            stats = cache_manager.get_cache_statistics()
            summary['single_transformer_blocks'][block_idx] = stats
            
            total_hit_rate += stats['cache_hit_rate']
            total_prediction_error += stats['average_prediction_error']
            total_ops += stats['total_operations']
            block_count += 1
            
            best_model = stats['predictor_info']['current_best']
            model_votes[best_model] = model_votes.get(best_model, 0) + 1
    
    if block_count > 0:
        summary['overall_stats'] = {
            'total_cache_hit_rate': total_hit_rate / block_count,
            'average_prediction_error': total_prediction_error / block_count,
            'total_operations': total_ops,
            'best_models': model_votes
        }
    
    return summary 