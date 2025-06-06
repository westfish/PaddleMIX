#!/usr/bin/env python3
"""
TeaBlockCache + Taylor 展开缓存复用示例脚本
演示如何使用集成了泰勒展开机制的 TeaBlockCache 进行 FLUX 模型推理加速
"""

import time
import paddle
from TeaBlockCache_taylor_forward import TeaBlockCacheTaylorForward
from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

def run_teablock_taylor_example():
    """运行 TeaBlockCache + Taylor 展开示例"""
    print("=== TeaBlockCache + Taylor 展开 FLUX 示例 ===")
    
    # 加载 FLUX 管道
    print("正在加载 FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
    
    # 配置 TeaBlockCache + Taylor 展开
    FluxTransformer2DModel.forward = TeaBlockCacheTaylorForward
    
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
    
    print("✅ TeaBlockCache + Taylor 展开配置完成")
    print(f"  时间范围: {pipe.transformer.step_start} - {pipe.transformer.step_end}")
    print(f"  Block cache start: {pipe.transformer.block_cache_start}")
    print(f"  Single block cache start: {pipe.transformer.single_block_cache_start}")
    print(f"  Block threshold: {pipe.transformer.block_rel_l1_thresh}")
    print(f"  Single block threshold: {pipe.transformer.single_block_rel_l1_thresh}")
    print(f"  泰勒展开阶数: 3")
    
    # 测试提示
    prompt = "An image of a squirrel in Picasso style"
    
    # 第一次运行 (建立缓存和泰勒展开基线)
    print("\n--- 第一次运行 (建立缓存和泰勒展开基线) ---")
    start_time = time.time()
    
    image1 = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    
    elapsed1 = time.time() - start_time
    print(f"第一次运行时间: {elapsed1:.2f}s")
    
    # 第二次运行 (使用缓存和泰勒展开预测)
    print("\n--- 第二次运行 (使用缓存和泰勒展开预测) ---")
    start_time = time.time()
    
    image2 = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    
    elapsed2 = time.time() - start_time
    print(f"第二次运行时间: {elapsed2:.2f}s")
    
    # 计算加速比
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 1.0
    print(f"\n🚀 加速比: {speedup:.2f}x")
    print(f"时间节省: {elapsed1 - elapsed2:.2f}s ({((elapsed1 - elapsed2) / elapsed1 * 100):.1f}%)")
    
    # 保存结果
    image1.save("teablock_taylor_first_run.png")
    image2.save("teablock_taylor_second_run.png")
    print("\n✅ 图像已保存:")
    print("  - teablock_taylor_first_run.png (第一次运行)")
    print("  - teablock_taylor_second_run.png (第二次运行)")
    
    print("\n=== 测试完成 ===")

def test_different_prompts():
    """测试不同提示词的性能"""
    print("\n=== 测试不同提示词的性能 ===")
    
    # 加载管道
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
    FluxTransformer2DModel.forward = TeaBlockCacheTaylorForward
    
    # 配置参数
    pipe.transformer.cnt = 0
    pipe.transformer.num_steps = 20  # 较少步数用于快速测试
    pipe.transformer.step_start = 100
    pipe.transformer.step_end = 800
    pipe.transformer.block_cache_start = 3
    pipe.transformer.single_block_cache_start = 5
    pipe.transformer.block_rel_l1_thresh = 0.35
    pipe.transformer.single_block_rel_l1_thresh = 0.45
    pipe.transformer.block_heuristic_states = {}
    pipe.transformer.single_block_heuristic_states = {}
    
    prompts = [
        "A beautiful sunset over mountains",
        "A cat sitting on a chair",
        "Abstract geometric patterns in bright colors",
        "A portrait of a woman in Renaissance style"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- 测试提示词 {i+1}: {prompt} ---")
        
        # 重置状态
        pipe.transformer.cnt = 0
        pipe.transformer.block_heuristic_states = {}
        pipe.transformer.single_block_heuristic_states = {}
        
        start_time = time.time()
        image = pipe(
            prompt,
            height=512,  # 较小尺寸用于快速测试
            width=512,
            num_inference_steps=20,
            generator=paddle.Generator().manual_seed(123 + i),
        ).images[0]
        
        elapsed = time.time() - start_time
        print(f"生成时间: {elapsed:.2f}s")
        
        image.save(f"teablock_taylor_prompt_{i+1}.png")

if __name__ == "__main__":
    # 运行主要示例
    run_teablock_taylor_example()
    
    # 测试不同提示词
    test_different_prompts() 