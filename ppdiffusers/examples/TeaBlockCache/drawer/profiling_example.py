"""
TeaBlockCache 性能分析示例

这个脚本展示了如何使用性能分析工具来识别 TeaBlockCache 中的性能瓶颈
"""

import paddle
import time
from TeaBlockCache_forward_profiled import (
    TeaBlockCacheForwardProfiled,
    print_profiling_results,
    reset_profiler,
    save_profiling_results,
    analyze_optimization_opportunities,
    get_profiling_results
)

def mock_transformer_model():
    """创建一个模拟的 transformer 模型用于测试"""
    class MockBlock:
        def norm1(self, x, emb=None):
            # 模拟 norm1 计算
            time.sleep(0.001)  # 模拟计算时间
            return x * 0.99  # 简单的归一化
        
        def __call__(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs=None):
            # 模拟 transformer block 计算
            time.sleep(0.005)  # 模拟实际计算时间
            return encoder_hidden_states * 1.01, hidden_states * 1.01
    
    class MockSingleBlock:
        def norm(self, x, emb=None):
            # 模拟 single block norm 计算
            time.sleep(0.0008)  # 模拟计算时间
            return x * 0.98
        
        def __call__(self, hidden_states, temb, image_rotary_emb, joint_attention_kwargs=None):
            # 模拟 single transformer block 计算
            time.sleep(0.003)  # 模拟实际计算时间
            return hidden_states * 1.02
    
    class MockModel:
        def __init__(self):
            # 模拟模型参数
            self.step_start = 0
            self.step_end = 1000
            self.block_cache_start = 0
            self.single_block_cache_start = 0
            self.block_rel_l1_thresh = 0
            self.single_block_rel_l1_thresh = 0
            self.num_steps = 50
            self.cnt = 0
            self.training = False
            self.gradient_checkpointing = False
            
            # 创建模拟的 transformer blocks
            self.transformer_blocks = [MockBlock() for _ in range(20)]
            self.single_transformer_blocks = [MockSingleBlock() for _ in range(38)]
            
            # 模拟的 embedder 和其他组件
            self.x_embedder = lambda x: x
            self.time_text_embed = lambda t, p, g=None: paddle.randn([1, 128])
            self.context_embedder = lambda x: x
            self.pos_embed = lambda x: paddle.randn([x.shape[0], 128])
            self.norm_out = lambda x, emb: x
            self.proj_out = lambda x: x
        
        def forward(self, *args, **kwargs):
            return TeaBlockCacheForwardProfiled(self, *args, **kwargs)
    
    return MockModel()

def run_profiling_test():
    """运行性能分析测试"""
    print("开始 TeaBlockCache 性能分析测试...")
    
    # 重置分析器
    reset_profiler()
    
    # 创建模拟模型
    model = mock_transformer_model()
    
    # 创建模拟输入数据
    batch_size = 1
    seq_length = 256
    hidden_dim = 1024
    
    hidden_states = paddle.randn([batch_size, seq_length, hidden_dim])
    encoder_hidden_states = paddle.randn([batch_size, 77, hidden_dim])
    pooled_projections = paddle.randn([batch_size, 128])
    timestep = paddle.to_tensor([500.0])
    img_ids = paddle.randn([seq_length - 77, 3])
    txt_ids = paddle.randn([77, 3])
    
    print(f"模拟数据形状:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  timestep: {timestep}")
    
    # 运行多次推理以收集统计数据
    num_runs = 5
    print(f"\n运行 {num_runs} 次推理以收集性能数据...")
    
    for i in range(num_runs):
        print(f"  运行 {i+1}/{num_runs}")
        with paddle.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=True
            )
    
    print("\n性能分析完成！")
    return get_profiling_results()

def analyze_results():
    """分析结果并提供优化建议"""
    print("\n" + "="*80)
    print("性能分析结果")
    print("="*80)
    
    # 打印详细统计信息
    print_profiling_results()
    
    # 提供优化建议
    analyze_optimization_opportunities()
    
    # 保存结果到文件
    save_profiling_results("teablock_performance_analysis.txt")
    print(f"\n详细分析结果已保存到: teablock_performance_analysis.txt")

def compare_cache_overhead():
    """比较缓存相关操作的开销"""
    stats, total_time = get_profiling_results()
    
    print("\n" + "="*60)
    print("缓存开销分析")
    print("="*60)
    
    # 分类统计
    categories = {
        'cache_operations': [],
        'heuristic_computation': [],
        'modulated_input': [],
        'state_management': [],
        'actual_computation': [],
        'others': []
    }
    
    for name, stat in stats.items():
        if 'cache' in name.lower():
            categories['cache_operations'].append((name, stat))
        elif 'heuristic' in name.lower():
            categories['heuristic_computation'].append((name, stat))
        elif 'modulated_input' in name.lower():
            categories['modulated_input'].append((name, stat))
        elif 'state' in name.lower() or 'init' in name.lower():
            categories['state_management'].append((name, stat))
        elif 'computation' in name.lower() or 'block' in name.lower():
            categories['actual_computation'].append((name, stat))
        else:
            categories['others'].append((name, stat))
    
    print(f"{'类别':<20} {'总时间(s)':<12} {'占比(%)':<8} {'组件数':<8}")
    print("-" * 60)
    
    for category, items in categories.items():
        if items:
            total_category_time = sum(stat['total_time'] for _, stat in items)
            percentage = (total_category_time / total_time * 100) if total_time > 0 else 0
            print(f"{category:<20} {total_category_time:<12.6f} {percentage:<8.2f} {len(items):<8}")
    
    # 计算缓存相关的总开销
    cache_related_categories = ['cache_operations', 'heuristic_computation', 'modulated_input', 'state_management']
    total_cache_overhead = sum(
        sum(stat['total_time'] for _, stat in categories[cat])
        for cat in cache_related_categories
        if categories[cat]
    )
    
    actual_computation_time = sum(
        stat['total_time'] for _, stat in categories['actual_computation']
    )
    
    print("-" * 60)
    print(f"缓存相关总开销: {total_cache_overhead:.6f}s ({total_cache_overhead/total_time*100:.2f}%)")
    print(f"实际计算时间: {actual_computation_time:.6f}s ({actual_computation_time/total_time*100:.2f}%)")
    
    if total_cache_overhead > 0 and actual_computation_time > 0:
        overhead_ratio = total_cache_overhead / actual_computation_time
        print(f"开销比率: {overhead_ratio:.2f}:1 (缓存开销:实际计算)")
        
        if overhead_ratio > 0.3:
            print("\n⚠️  警告: 缓存开销较高，可能需要优化!")
        elif overhead_ratio > 0.1:
            print("\n⚡ 提示: 缓存开销适中，有优化空间")
        else:
            print("\n✅ 缓存开销较低，性能良好")

def generate_optimization_report():
    """生成优化报告"""
    stats, total_time = get_profiling_results()
    
    report_filename = "optimization_report.md"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("# TeaBlockCache 性能优化报告\n\n")
        f.write("## 概述\n")
        f.write(f"- 总分析时间: {total_time:.6f}s\n")
        f.write(f"- 分析组件数: {len(stats)}\n\n")
        
        f.write("## 性能热点 (Top 5)\n")
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        for i, (name, stat) in enumerate(sorted_stats[:5]):
            f.write(f"{i+1}. **{name}**: {stat['percentage']:.2f}% ({stat['total_time']:.6f}s)\n")
        
        f.write("\n## 优化建议\n")
        
        # 根据性能数据生成具体建议
        for name, stat in sorted_stats:
            percentage = stat['percentage']
            if percentage > 15:
                f.write(f"\n### 🔴 高优先级: {name} ({percentage:.2f}%)\n")
                if 'heuristic' in name.lower():
                    f.write("- 考虑减少启发式计算频率\n")
                    f.write("- 使用更简单的相似性度量\n")
                elif 'cache' in name.lower():
                    f.write("- 优化缓存策略\n")
                    f.write("- 减少不必要的 clone 操作\n")
            elif percentage > 8:
                f.write(f"\n### 🟡 中优先级: {name} ({percentage:.2f}%)\n")
                f.write("- 考虑局部优化\n")
        
        f.write(f"\n## 详细数据\n")
        f.write("| 组件名称 | 总时间(s) | 平均时间(s) | 占比(%) | 调用次数 |\n")
        f.write("|----------|-----------|-------------|---------|----------|\n")
        for name, stat in sorted_stats:
            f.write(f"| {name} | {stat['total_time']:.6f} | {stat['avg_time']:.6f} | {stat['percentage']:.2f} | {stat['count']} |\n")
    
    print(f"\n📊 详细优化报告已生成: {report_filename}")

if __name__ == "__main__":
    try:
        # 运行性能分析
        run_profiling_test()
        
        # 分析结果
        analyze_results()
        
        # 比较缓存开销
        compare_cache_overhead()
        
        # 生成优化报告
        generate_optimization_report()
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 