#!/usr/bin/env python3
"""
简单测试脚本：验证 TeaBlockCache 混合缓存策略
"""

import time
import paddle
from TeaBlockCache_forward import TeaBlockCacheForward
from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

def test_teablock_cache():
    """测试 TeaBlockCache 混合策略"""
    print("=== TeaBlockCache 测试开始 ===")
    
    # 加载管道
    print("正在加载 FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
    
    # 替换 forward 方法
    FluxTransformer2DModel.forward = TeaBlockCacheForward
    print("✅ 已替换为 TeaBlockCache forward 方法")
    
    # 配置混合缓存参数
    pipe.transformer.cnt = 0
    pipe.transformer.num_steps = 28  # 使用较少步数进行快速测试
    
    # 时间维度参数
    pipe.transformer.step_start = 100
    pipe.transformer.step_end = 800
    
    # 块维度参数
    pipe.transformer.block_cache_start = 5
    pipe.transformer.single_block_cache_start = 10
    
    # 启发式阈值
    pipe.transformer.block_rel_l1_thresh = 0.3
    pipe.transformer.single_block_rel_l1_thresh = 0.4
    
    print("✅ 配置参数已设置")
    print(f"   时间范围: {pipe.transformer.step_start} - {pipe.transformer.step_end}")
    print(f"   Block cache start: {pipe.transformer.block_cache_start}")
    print(f"   Single block cache start: {pipe.transformer.single_block_cache_start}")
    print(f"   Block threshold: {pipe.transformer.block_rel_l1_thresh}")
    print(f"   Single block threshold: {pipe.transformer.single_block_rel_l1_thresh}")
    
    # 测试提示
    prompt = "A simple cat sitting on a chair"
    
    # 第一次运行（缓存构建）
    print("\n--- 第一次运行 (缓存构建) ---")
    start_time = time.time()
    
    image1 = pipe(
        prompt,
        height=512,  # 使用较小尺寸加快测试
        width=512,
        guidance_scale=3.5,
        num_inference_steps=28,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    
    first_run_time = time.time() - start_time
    print(f"✅ 第一次运行完成: {first_run_time:.2f} 秒")
    
    # 第二次运行（缓存利用）
    print("\n--- 第二次运行 (缓存利用) ---")
    start_time = time.time()
    
    image2 = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=28,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    
    second_run_time = time.time() - start_time
    print(f"✅ 第二次运行完成: {second_run_time:.2f} 秒")
    
    # 分析结果
    speedup = first_run_time / second_run_time if second_run_time > 0 else 1.0
    print(f"\n=== 性能分析 ===")
    print(f"第一次运行: {first_run_time:.2f} 秒")
    print(f"第二次运行: {second_run_time:.2f} 秒")
    print(f"加速比: {speedup:.2f}x")
    
    if speedup > 1.1:
        print("✅ 检测到性能提升！缓存策略工作正常")
    else:
        print("⚠️  未检测到明显性能提升，可能需要调整参数")
    
    # 保存结果
    image2.save("teablock_test_result.png")
    print("\n✅ 图像已保存为: teablock_test_result.png")
    
    # 检查缓存状态
    if hasattr(pipe.transformer, 'block_heuristic_states'):
        num_cached_blocks = len(pipe.transformer.block_heuristic_states)
        print(f"✅ Transformer blocks 缓存数量: {num_cached_blocks}")
    
    if hasattr(pipe.transformer, 'single_block_heuristic_states'):
        num_cached_single_blocks = len(pipe.transformer.single_block_heuristic_states)
        print(f"✅ Single blocks 缓存数量: {num_cached_single_blocks}")
    
    print("\n=== TeaBlockCache 测试完成 ===")
    return True

def test_different_thresholds():
    """测试不同阈值设置的影响"""
    print("\n=== 不同阈值测试 ===")
    
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
    FluxTransformer2DModel.forward = TeaBlockCacheForward
    
    # 基础配置
    pipe.transformer.cnt = 0
    pipe.transformer.num_steps = 20  # 更快的测试
    pipe.transformer.step_start = 100
    pipe.transformer.step_end = 800
    pipe.transformer.block_cache_start = 3
    pipe.transformer.single_block_cache_start = 5
    
    prompt = "A red apple on a table"
    
    # 测试不同的阈值组合
    test_configs = [
        (0.2, 0.25, "保守缓存"),
        (0.4, 0.5, "平衡缓存"), 
        (0.6, 0.7, "激进缓存")
    ]
    
    results = []
    
    for block_thresh, single_thresh, description in test_configs:
        pipe.transformer.block_rel_l1_thresh = block_thresh
        pipe.transformer.single_block_rel_l1_thresh = single_thresh
        
        print(f"\n--- {description} (Block: {block_thresh}, Single: {single_thresh}) ---")
        
        start_time = time.time()
        image = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            generator=paddle.Generator().manual_seed(123),
        ).images[0]
        
        elapsed = time.time() - start_time
        print(f"时间: {elapsed:.2f} 秒")
        results.append((description, elapsed))
        
        # 重置缓存状态
        pipe.transformer.block_heuristic_states = {}
        pipe.transformer.single_block_heuristic_states = {}
        pipe.transformer.cnt = 0
    
    print(f"\n=== 阈值测试结果对比 ===")
    for desc, time_taken in results:
        print(f"{desc}: {time_taken:.2f} 秒")
    
    return results

if __name__ == "__main__":
    try:
        # 基础功能测试
        test_teablock_cache()
        
        # 阈值对比测试
        test_different_thresholds()
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 