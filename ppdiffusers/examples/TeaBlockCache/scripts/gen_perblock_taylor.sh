#!/bin/bash

echo "🚀 Running PerBlock Taylor Prediction Method..."
echo "This method uses per-block Taylor caches for more efficient prediction"
echo ""

# Conservative configuration for better quality
echo "📊 Running Conservative PerBlock Taylor configuration..."
CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
--step_start 100 \
--step_end 800 \
--block_cache_start 5 \
--single_block_cache_start 8 \
--block_rel_l1_thresh 50 \
--single_block_rel_l1_thresh 50 \
--inference_step 50 \
--perblock_taylor \
--seed 42 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

echo ""
echo "⚡ Running Aggressive PerBlock Taylor configuration..."
CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
--step_start 50 \
--step_end 950 \
--block_cache_start 2 \
--single_block_cache_start 3 \
--block_rel_l1_thresh 100 \
--single_block_rel_l1_thresh 100 \
--inference_step 50 \
--perblock_taylor \
--seed 42 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

echo ""
echo "🔥 Running Ultra Aggressive PerBlock Taylor configuration..."
CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
--step_start 20 \
--step_end 980 \
--block_cache_start 1 \
--single_block_cache_start 1 \
--block_rel_l1_thresh 200 \
--single_block_rel_l1_thresh 200 \
--inference_step 50 \
--perblock_taylor \
--seed 42 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

echo ""
echo "✅ PerBlock Taylor experiments completed!"
echo "Check output directory for generated images and performance statistics." 