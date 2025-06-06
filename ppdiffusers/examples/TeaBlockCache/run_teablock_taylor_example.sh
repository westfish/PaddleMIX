#!/bin/bash

# TeaBlockCache + Taylor 展开示例运行脚本

echo "=== TeaBlockCache + Taylor 展开 FLUX 示例 ==="

# 基础配置
INFERENCE_STEPS=20
DATASET="coco10k"
SEED=42

# TeaBlockCache 配置
STEP_START=100
STEP_END=800
BLOCK_CACHE_START=5
SINGLE_BLOCK_CACHE_START=10
BLOCK_THRESH=0.3
SINGLE_BLOCK_THRESH=0.4

# Taylor 展开配置
TAYLOR_MAX_ORDER=3
TAYLOR_FIRST_ENHANCE=2

# 输出路径
OUTPUT_PATH="./output/teablock_taylor_demo"

# 测试提示词
TEST_PROMPT="A beautiful sunset over mountains with vibrant colors"

echo "运行配置："
echo "  推理步数: $INFERENCE_STEPS"
echo "  时间范围: $STEP_START - $STEP_END"
echo "  Block缓存开始: $BLOCK_CACHE_START"
echo "  Single Block缓存开始: $SINGLE_BLOCK_CACHE_START"
echo "  Block阈值: $BLOCK_THRESH"
echo "  Single Block阈值: $SINGLE_BLOCK_THRESH"
echo "  Taylor最大阶数: $TAYLOR_MAX_ORDER"
echo "  Taylor增强开始步数: $TAYLOR_FIRST_ENHANCE"
echo ""

# 运行 TeaBlockCache + Taylor
echo "🚀 开始运行 TeaBlockCache + Taylor..."
python teablock_generation.py \
    --teablock_taylor \
    --prompt "$TEST_PROMPT" \
    --saved_path "$OUTPUT_PATH" \
    --inference_step $INFERENCE_STEPS \
    --seed $SEED \
    --step_start $STEP_START \
    --step_end $STEP_END \
    --block_cache_start $BLOCK_CACHE_START \
    --single_block_cache_start $SINGLE_BLOCK_CACHE_START \
    --block_rel_l1_thresh $BLOCK_THRESH \
    --single_block_rel_l1_thresh $SINGLE_BLOCK_THRESH \
    --taylor_max_order $TAYLOR_MAX_ORDER \
    --taylor_first_enhance $TAYLOR_FIRST_ENHANCE \
    --dataset $DATASET

echo ""
echo "✅ TeaBlockCache + Taylor 运行完成！"
echo "图片保存在: $OUTPUT_PATH"

# 运行对比测试（如果需要）
read -p "是否运行原始FLUX对比测试？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 开始运行原始FLUX对比..."
    python teablock_generation.py \
        --origin \
        --prompt "$TEST_PROMPT" \
        --saved_path "$OUTPUT_PATH" \
        --inference_step $INFERENCE_STEPS \
        --seed $SEED \
        --dataset $DATASET
    
    echo "✅ 原始FLUX运行完成！"
fi

echo ""
echo "🎉 所有测试完成！请检查输出目录中的图片质量。" 