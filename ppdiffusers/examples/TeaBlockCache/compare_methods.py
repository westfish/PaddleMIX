#!/usr/bin/env python3
"""
TeaBlockCache 方法对比测试脚本
比较原始 TeaBlockCache 和泰勒展开增强版本的性能
"""

import time
import paddle
from TeaBlockCache_forward import TeaBlockCacheForward
from TeaBlockCache_taylor_forward import TeaBlockCacheTaylorForward
from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

def test_method(method_name, forward_func, pipe, prompt, num_steps=20):
    """测试单个方法的性能"""
    print(f"\n=== 测试 {method_name} ===")
    
    # 替换 forward 方法
    FluxTransformer2DModel.forward = forward_func
    
    # 重置状态
    pipe.transformer.cnt = 0
    pipe.transformer.block_heuristic_states = {}
    pipe.transformer.single_block_heuristic_states = {}
    if hasattr(pipe.transformer, 'taylor_cache_system'):
        delattr(pipe.transformer, 'taylor_cache_system')
    
    # 第一次运行 (建立缓存)
    print("  第一次运行 (建立缓存)...")
    start_time = time.time()
    
    image1 = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=num_steps,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    
    first_run_time = time.time() - start_time
    print(f"  第一次运行时间: {first_run_time:.2f}s")
    
    # 第二次运行 (使用缓存)
    print("  第二次运行 (使用缓存)...")
    start_time = time.time()
    
    image2 = pipe(
        prompt,
        height=512,
        width=512,
        num_inference_steps=num_steps,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    
    second_run_time = time.time() - start_time
    print(f"  第二次运行时间: {second_run_time:.2f}s")
    
    # 计算加速比
    speedup = first_run_time / second_run_time if second_run_time > 0 else 1.0
    time_saved = first_run_time - second_run_time
    percentage_saved = (time_saved / first_run_time * 100) if first_run_time > 0 else 0
    
    print(f"  🚀 加速比: {speedup:.2f}x")
    print(f"  ⏱️  时间节省: {time_saved:.2f}s ({percentage_saved:.1f}%)")
    
    # 保存图像
    image1.save(f"{method_name.lower().replace(' ', '_')}_first_run.png")
    image2.save(f"{method_name.lower().replace(' ', '_')}_second_run.png")
    
    return {
        'method': method_name,
        'first_run_time': first_run_time,
        'second_run_time': second_run_time,
        'speedup': speedup,
        'time_saved': time_saved,
        'percentage_saved': percentage_saved
    }

def main():
    """主测试函数"""
    print("🔬 TeaBlockCache 方法对比测试")
    print("=" * 50)
    
    # 加载 FLUX 管道
    print("正在加载 FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
    
    # 配置基础参数
    pipe.transformer.cnt = 0
    pipe.transformer.num_steps = 20
    pipe.transformer.step_start = 100
    pipe.transformer.step_end = 800
    pipe.transformer.block_cache_start = 3
    pipe.transformer.single_block_cache_start = 5
    pipe.transformer.block_rel_l1_thresh = 0.3
    pipe.transformer.single_block_rel_l1_thresh = 0.4
    pipe.transformer.block_heuristic_states = {}
    pipe.transformer.single_block_heuristic_states = {}
    
    # 测试提示
    prompt = "A beautiful landscape with mountains and a lake"
    print(f"📝 测试提示: {prompt}")
    print(f"⚙️  推理步数: {pipe.transformer.num_steps}")
    print(f"🎯 图像尺寸: 512x512")
    
    # 测试结果存储
    results = []
    
    # 测试 1: 原始 TeaBlockCache
    result1 = test_method(
        "Original TeaBlockCache", 
        TeaBlockCacheForward, 
        pipe, 
        prompt, 
        pipe.transformer.num_steps
    )
    results.append(result1)
    
    # 测试 2: TeaBlockCache + Taylor 展开
    result2 = test_method(
        "TeaBlockCache + Taylor", 
        TeaBlockCacheTaylorForward, 
        pipe, 
        prompt, 
        pipe.transformer.num_steps
    )
    results.append(result2)
    
    # 输出对比结果
    print("\n" + "=" * 60)
    print("📊 对比结果总结")
    print("=" * 60)
    
    print(f"{'方法':<25} {'第一次(s)':<12} {'第二次(s)':<12} {'加速比':<10} {'时间节省':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['method']:<25} "
              f"{result['first_run_time']:<12.2f} "
              f"{result['second_run_time']:<12.2f} "
              f"{result['speedup']:<10.2f}x "
              f"{result['time_saved']:<12.2f}s")
    
    # 计算改进
    if len(results) >= 2:
        original = results[0]
        taylor = results[1]
        
        first_run_improvement = (original['first_run_time'] - taylor['first_run_time']) / original['first_run_time'] * 100
        second_run_improvement = (original['second_run_time'] - taylor['second_run_time']) / original['second_run_time'] * 100
        speedup_improvement = taylor['speedup'] - original['speedup']
        
        print("\n🎯 泰勒展开版本改进:")
        print(f"  第一次运行改进: {first_run_improvement:.1f}%")
        print(f"  第二次运行改进: {second_run_improvement:.1f}%")
        print(f"  加速比改进: +{speedup_improvement:.2f}x")
    
    print(f"\n✅ 测试完成! 生成的图像已保存:")
    for result in results:
        method_name = result['method'].lower().replace(' ', '_')
        print(f"  - {method_name}_first_run.png")
        print(f"  - {method_name}_second_run.png")

def quick_test():
    """快速测试函数 - 使用更少的步数"""
    print("\n🚀 快速测试模式 (10步推理)")
    print("=" * 40)
    
    # 加载管道
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
    
    # 配置参数 (更激进的设置用于快速测试)
    pipe.transformer.cnt = 0
    pipe.transformer.num_steps = 10
    pipe.transformer.step_start = 50
    pipe.transformer.step_end = 900
    pipe.transformer.block_cache_start = 2
    pipe.transformer.single_block_cache_start = 3
    pipe.transformer.block_rel_l1_thresh = 0.25
    pipe.transformer.single_block_rel_l1_thresh = 0.35
    
    prompt = "A simple cat sitting on a chair"
    
    # 只测试泰勒展开版本
    result = test_method(
        "TeaBlockCache + Taylor (Quick)", 
        TeaBlockCacheTaylorForward, 
        pipe, 
        prompt, 
        10
    )
    
    print(f"\n🎉 快速测试结果: {result['speedup']:.2f}x 加速")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()
        
    print("\n💡 提示: 使用 --quick 参数进行快速测试") 