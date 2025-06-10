# echo ""
# echo "🚀 Running PerBlock Taylor configuration..."
# CUDA_VISIBLE_DEVICES=1  python teablock_generation.py \
# --step_start 100 \
# --step_end 800 \
# --block_cache_start 3 \
# --single_block_cache_start 5 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --perblock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

echo "🔥 Running ULTRA AGGRESSIVE configuration..."
CUDA_VISIBLE_DEVICES=1  python teablock_generation.py \
--step_start 50 \
--step_end 950 \
--block_cache_start 1 \
--single_block_cache_start 1 \
--block_rel_l1_thresh 200 \
--single_block_rel_l1_thresh 200 \
--rel_l1_thresh 2 \
--inference_step 50 \
--taylor_max_order 1 \
--taylor_first_enhance 1 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache


