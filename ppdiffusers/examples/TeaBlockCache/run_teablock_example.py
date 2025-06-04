#!/usr/bin/env python3
"""
TeaBlockCache 快速示例脚本
演示如何使用 TeaBlockCache 进行 FLUX 模型推理加速
"""

import time
import paddle
from TeaBlockCache_forward import TeaBlockCacheForward
from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

def run_teablock_example():
    """运行 TeaBlockCache 示例"""
    print("=== TeaBlockCache FLUX 示例 ===")
    
    # 加载 FLUX 管道
    print("正在加载 FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
    
    # 配置 TeaBlockCache
    FluxTransformer2DModel.forward = TeaBlockCacheForward
    
    # 基础配置
    pipe.transformer.cnt = 0
    pipe.transformer.num_steps = 28
    pipe.transformer.step_start = 100
    pipe.transformer.step_end = 800
    pipe.transformer.block_cache_start = 5
    pipe.transformer.single_block_cache_start = 10
    pipe.transformer.block_rel_l1_thresh = 0.3
    pipe.transformer.single_block_rel_l1_thresh = 0.4
    pipe.transformer.block_heuristic_states = {}
    pipe.transformer.single_block_heuristic_states = {}
    
    print("✅ TeaBlockCache 配置完成")
    print(f"  时间范围: {pipe.transformer.step_start} - {pipe.transformer.step_end}")
    print(f"  Block cache start: {pipe.transformer.block_cache_start}")
    print(f"  Single block cache start: {pipe.transformer.single_block_cache_start}")
    print(f"  Block threshold: {pipe.transformer.block_rel_l1_thresh}")
    print(f"  Single block threshold: {pipe.transformer.single_block_rel_l1_thresh}")
    
    # 测试提示
    prompt = "An image of a squirrel in Picasso style"
    
    # 第一次运行
    print("\n--- 第一次运行 (建立缓存) ---")
    start_time = time.time()
    image1 = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=28,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    first_time = time.time() - start_time
    print(f"第一次运行时间: {first_time:.2f}s")
    
    # 第二次运行 (利用缓存)
    print("\n--- 第二次运行 (利用缓存) ---")
    start_time = time.time()
    image2 = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=28,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    second_time = time.time() - start_time
    print(f"第二次运行时间: {second_time:.2f}s")
    
    # 计算加速比
    speedup = first_time / second_time if second_time > 0 else 1.0
    print(f"\n=== 性能统计 ===")
    print(f"第一次运行: {first_time:.2f}s")
    print(f"第二次运行: {second_time:.2f}s")
    print(f"加速比: {speedup:.2f}x")
    
    # 缓存统计
    if hasattr(pipe.transformer, 'block_heuristic_states'):
        print(f"缓存的 transformer blocks: {len(pipe.transformer.block_heuristic_states)}")
    if hasattr(pipe.transformer, 'single_block_heuristic_states'):
        print(f"缓存的 single blocks: {len(pipe.transformer.single_block_heuristic_states)}")
    
    # 保存图片
    image2.save("teablock_example_result.png")
    print("\n✅ 图片已保存为: teablock_example_result.png")


def test_different_configurations():
    """测试不同的 TeaBlockCache 配置"""
    print("\n=== 测试不同配置 ===")
    
    configs = [
        {
            'name': '保守配置',
            'block_rel_l1_thresh': 0.2,
            'single_block_rel_l1_thresh': 0.25,
            'block_cache_start': 8,
            'single_block_cache_start': 15
        },
        {
            'name': '平衡配置',
            'block_rel_l1_thresh': 0.3,
            'single_block_rel_l1_thresh': 0.4,
            'block_cache_start': 5,
            'single_block_cache_start': 10
        },
        {
            'name': '激进配置',
            'block_rel_l1_thresh': 0.5,
            'single_block_rel_l1_thresh': 0.6,
            'block_cache_start': 3,
            'single_block_cache_start': 5
        }
    ]
    
    prompt = "A beautiful sunset over the ocean"
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        # 重新加载管道
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        FluxTransformer2DModel.forward = TeaBlockCacheForward
        
        # 应用配置
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 20  # 较少步数用于快速测试
        pipe.transformer.step_start = 100
        pipe.transformer.step_end = 800
        pipe.transformer.block_cache_start = config['block_cache_start']
        pipe.transformer.single_block_cache_start = config['single_block_cache_start']
        pipe.transformer.block_rel_l1_thresh = config['block_rel_l1_thresh']
        pipe.transformer.single_block_rel_l1_thresh = config['single_block_rel_l1_thresh']
        pipe.transformer.block_heuristic_states = {}
        pipe.transformer.single_block_heuristic_states = {}
        
        start_time = time.time()
        image = pipe(
            prompt,
            height=512,  # 较小尺寸用于快速测试
            width=512,
            num_inference_steps=20,
            generator=paddle.Generator().manual_seed(123),
        ).images[0]
        
        elapsed = time.time() - start_time
        print(f"{config['name']}: {elapsed:.2f}s")
        
        del pipe


if __name__ == "__main__":
    try:
        # 基础示例
        run_teablock_example()
        
        # 配置对比测试
        test_different_configurations()
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 