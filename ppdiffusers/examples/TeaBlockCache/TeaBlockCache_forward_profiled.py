from typing import Any, Dict, Optional, Tuple, Union
import time
import collections
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, logger, scale_lora_layers, unscale_lora_layers

class TeaBlockCacheProfiler:
    """
    性能分析器，用于统计 TeaBlockCache 各个组件的时间消耗
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有计时器"""
        self.timings = collections.defaultdict(list)
        self.current_timers = {}
        self.call_counts = collections.defaultdict(int)
    
    def start_timer(self, name: str):
        """开始计时"""
        self.current_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str):
        """结束计时并记录"""
        if name in self.current_timers:
            elapsed = time.perf_counter() - self.current_timers[name]
            self.timings[name].append(elapsed)
            self.call_counts[name] += 1
            del self.current_timers[name]
            return elapsed
        return 0
    
    def get_stats(self):
        """获取统计信息"""
        stats = {}
        total_time = 0
        
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'count': len(times)
                }
                total_time += stats[name]['total_time']
        
        # 计算百分比
        for name in stats:
            stats[name]['percentage'] = (stats[name]['total_time'] / total_time * 100) if total_time > 0 else 0
        
        return stats, total_time
    
    def print_stats(self):
        """打印统计信息"""
        stats, total_time = self.get_stats()
        
        print("\n" + "="*80)
        print("TeaBlockCache 性能分析报告")
        print("="*80)
        print(f"总计时时间: {total_time:.6f}s")
        print("-"*80)
        print(f"{'组件名称':<30} {'总时间(s)':<12} {'平均时间(s)':<12} {'占比(%)':<8} {'调用次数':<8}")
        print("-"*80)
        
        # 按照总时间排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for name, stat in sorted_stats:
            print(f"{name:<30} {stat['total_time']:<12.6f} {stat['avg_time']:<12.6f} {stat['percentage']:<8.2f} {stat['count']:<8}")
        
        print("-"*80)
        
        # 分析结果
        if sorted_stats:
            top_consumer = sorted_stats[0]
            print(f"\n最耗时的组件: {top_consumer[0]} ({top_consumer[1]['percentage']:.2f}%)")
            
            cache_related = [name for name, _ in sorted_stats if 'cache' in name.lower() or 'heuristic' in name.lower()]
            if cache_related:
                cache_time = sum(stats[name]['total_time'] for name in cache_related)
                cache_percentage = cache_time / total_time * 100
                print(f"缓存相关操作总时间: {cache_time:.6f}s ({cache_percentage:.2f}%)")

# 全局性能分析器实例
_profiler = TeaBlockCacheProfiler()

def TeaBlockCacheForwardProfiled(
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
    带性能分析的 TeaBlockCache forward 函数
    """
    
    _profiler.start_timer("total_forward")
    
    # 1. 初始化和预处理
    _profiler.start_timer("initialization")
    
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
    
    _profiler.end_timer("initialization")

    # 2. 时间和引导处理
    _profiler.start_timer("timestep_guidance_processing")
    
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
    
    _profiler.end_timer("timestep_guidance_processing")

    # 3. 位置编码处理
    _profiler.start_timer("positional_encoding")
    
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
    
    _profiler.end_timer("positional_encoding")

    # 4. IP Adapter 处理
    _profiler.start_timer("ip_adapter_processing")
    
    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
    
    _profiler.end_timer("ip_adapter_processing")

    # 5. 缓存状态初始化
    _profiler.start_timer("cache_state_initialization")
    
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
        self.cnt = 0

    # Global step counter
    self.cnt += 1
    
    # Force computation at first and last steps
    force_compute = (self.cnt == 1 or self.cnt == self.num_steps)
    
    _profiler.end_timer("cache_state_initialization")

    # 6. Transformer blocks 处理
    _profiler.start_timer("transformer_blocks_total")
    
    for index_block, block in enumerate(self.transformer_blocks):
        _profiler.start_timer("single_transformer_block")
        
        # 6.1 块状态初始化
        _profiler.start_timer("block_state_init")
        if index_block not in self.block_heuristic_states:
            self.block_heuristic_states[index_block] = {
                'accumulated_distance': 0,
                'previous_modulated_input': None,
                'cached_output': None,
                'cached_encoder_output': None,
                'should_compute': True
            }
        block_state = self.block_heuristic_states[index_block]
        _profiler.end_timer("block_state_init")
        
        # 6.2 启发式计算
        _profiler.start_timer("heuristic_computation")
        should_compute_block = force_compute
        
        if not force_compute and is_within_time_range and index_block >= self.block_cache_start:
            # 6.2.1 调制输入计算
            _profiler.start_timer("modulated_input_calculation")
            inp = hidden_states.clone()
            temb_ = temb.clone()
            norm_result = block.norm1(inp, emb=temb_)
            if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
            elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                modulated_inp = norm_result[0]
            else:
                modulated_inp = norm_result
            _profiler.end_timer("modulated_input_calculation")
            
            # 6.2.2 距离计算和阈值检查
            _profiler.start_timer("distance_calculation")
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
            _profiler.end_timer("distance_calculation")
            
            # 6.2.3 状态更新
            _profiler.start_timer("state_update")
            block_state['previous_modulated_input'] = modulated_inp.clone()
            _profiler.end_timer("state_update")
        else:
            should_compute_block = True
            if is_within_time_range and index_block >= self.block_cache_start:
                _profiler.start_timer("modulated_input_calculation")
                inp = hidden_states.clone()
                temb_ = temb.clone()
                norm_result = block.norm1(inp, emb=temb_)
                if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
                elif isinstance(norm_result, tuple) and len(norm_result) >= 1:
                    modulated_inp = norm_result[0]
                else:
                    modulated_inp = norm_result
                block_state['previous_modulated_input'] = modulated_inp.clone()
                _profiler.end_timer("modulated_input_calculation")
        
        _profiler.end_timer("heuristic_computation")

        # 6.3 块计算或缓存使用
        if should_compute_block:
            _profiler.start_timer("block_computation")
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
            _profiler.end_timer("block_computation")
            
            # 6.4 缓存操作
            _profiler.start_timer("cache_operations")
            if is_within_time_range and index_block >= self.block_cache_start:
                block_state['cached_output'] = hidden_states.clone()
                block_state['cached_encoder_output'] = encoder_hidden_states.clone()
            _profiler.end_timer("cache_operations")
            
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
            # 6.5 缓存使用
            _profiler.start_timer("cache_usage")
            if (block_state['cached_output'] is not None and 
                block_state['cached_encoder_output'] is not None):
                hidden_states = block_state['cached_output']
                encoder_hidden_states = block_state['cached_encoder_output']
            _profiler.end_timer("cache_usage")

        _profiler.end_timer("single_transformer_block")

    _profiler.end_timer("transformer_blocks_total")

    # 7. 状态连接
    _profiler.start_timer("state_concatenation")
    hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)
    _profiler.end_timer("state_concatenation")

    # 8. Single transformer blocks 处理 (类似的分析)
    _profiler.start_timer("single_transformer_blocks_total")
    
    for index_block, block in enumerate(self.single_transformer_blocks):
        _profiler.start_timer("single_block_processing")
        
        # 初始化和启发式计算 (类似前面的逻辑)
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
            _profiler.start_timer("single_block_heuristic")
            inp = hidden_states.clone()
            temb_ = temb.clone()
            norm_result = block.norm(inp, emb=temb_)
            if isinstance(norm_result, tuple) and len(norm_result) >= 5:
                modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_result
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
            _profiler.end_timer("single_block_heuristic")
        else:
            should_compute_block = True
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
                block_state['previous_modulated_input'] = modulated_inp.clone()

        if should_compute_block:
            _profiler.start_timer("single_block_computation")
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
            _profiler.end_timer("single_block_computation")
            
            if is_within_time_range and index_block >= self.single_block_cache_start:
                block_state['cached_output'] = hidden_states.clone()
            
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
        else:
            if block_state['cached_output'] is not None:
                hidden_states = block_state['cached_output']

        _profiler.end_timer("single_block_processing")

    _profiler.end_timer("single_transformer_blocks_total")

    # 9. 最终处理
    _profiler.start_timer("final_processing")
    
    # Extract only the image hidden states
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    # Reset counter if we've reached the end
    if self.cnt == self.num_steps:
        self.cnt = 0

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    
    _profiler.end_timer("final_processing")
    _profiler.end_timer("total_forward")

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

def get_profiling_results():
    """获取性能分析结果"""
    return _profiler.get_stats()

def print_profiling_results():
    """打印性能分析结果"""
    _profiler.print_stats()

def reset_profiler():
    """重置性能分析器"""
    _profiler.reset()

def save_profiling_results(filename="teablock_profiling_results.txt"):
    """保存性能分析结果到文件"""
    stats, total_time = _profiler.get_stats()
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("TeaBlockCache 性能分析报告\n")
        f.write("="*80 + "\n")
        f.write(f"总计时时间: {total_time:.6f}s\n")
        f.write("-"*80 + "\n")
        f.write(f"{'组件名称':<30} {'总时间(s)':<12} {'平均时间(s)':<12} {'占比(%)':<8} {'调用次数':<8}\n")
        f.write("-"*80 + "\n")
        
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for name, stat in sorted_stats:
            f.write(f"{name:<30} {stat['total_time']:<12.6f} {stat['avg_time']:<12.6f} {stat['percentage']:<8.2f} {stat['count']:<8}\n")
        
        f.write("-"*80 + "\n")
        
        if sorted_stats:
            top_consumer = sorted_stats[0]
            f.write(f"\n最耗时的组件: {top_consumer[0]} ({top_consumer[1]['percentage']:.2f}%)\n")
            
            cache_related = [name for name, _ in sorted_stats if 'cache' in name.lower() or 'heuristic' in name.lower()]
            if cache_related:
                cache_time = sum(stats[name]['total_time'] for name in cache_related)
                cache_percentage = cache_time / total_time * 100
                f.write(f"缓存相关操作总时间: {cache_time:.6f}s ({cache_percentage:.2f}%)\n")

# 使用示例和优化建议
def analyze_optimization_opportunities():
    """分析优化机会并提供建议"""
    stats, total_time = _profiler.get_stats()
    
    print("\n" + "="*80)
    print("优化建议")
    print("="*80)
    
    suggestions = []
    
    # 检查各个组件的时间占比
    for name, stat in stats.items():
        percentage = stat['percentage']
        
        if 'heuristic' in name.lower() and percentage > 10:
            suggestions.append(f"• 启发式计算 ({name}) 占用 {percentage:.2f}% 的时间，可以考虑：")
            suggestions.append("  - 减少启发式计算的频率")
            suggestions.append("  - 优化距离计算算法")
            suggestions.append("  - 使用更简单的相似性度量")
        
        if 'cache' in name.lower() and percentage > 5:
            suggestions.append(f"• 缓存操作 ({name}) 占用 {percentage:.2f}% 的时间，可以考虑：")
            suggestions.append("  - 减少不必要的 clone() 操作")
            suggestions.append("  - 使用更高效的缓存策略")
            suggestions.append("  - 批量更新缓存状态")
        
        if 'modulated_input' in name.lower() and percentage > 8:
            suggestions.append(f"• 调制输入计算 ({name}) 占用 {percentage:.2f}% 的时间，可以考虑：")
            suggestions.append("  - 复用已计算的 norm 结果")
            suggestions.append("  - 延迟计算直到确实需要")
        
        if 'state' in name.lower() and percentage > 3:
            suggestions.append(f"• 状态管理 ({name}) 占用 {percentage:.2f}% 的时间，可以考虑：")
            suggestions.append("  - 使用更高效的数据结构")
            suggestions.append("  - 减少状态更新频率")
    
    if not suggestions:
        suggestions.append("• 当前性能分布较为均衡，可以考虑整体算法优化")
    
    for suggestion in suggestions:
        print(suggestion)
    
    print("="*80) 