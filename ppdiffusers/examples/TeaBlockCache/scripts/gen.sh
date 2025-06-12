CUDA_VISIBLE_DEVICES=1  python teablock_generation.py \
--step_start 50 \
--step_end 950 \
--block_cache_start 1 \
--single_block_cache_start 1 \
--block_rel_l1_thresh 0.5 \
--single_block_rel_l1_thresh 0.5  \
--inference_step 50 \
--perblock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache


# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 1.0 \
# --single_block_rel_l1_thresh 1.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=4  python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --perblock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --rel_l1_thresh 2.1 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache


# CUDA_VISIBLE_DEVICES=7 nohup python teablock_generation.py \
# --step_start 0 \
# --step_end 1000 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --rel_l1_thresh 0.5 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_0_1000_1_1_2.0_2.0_1_1_0.5 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup python teablock_generation.py \
# --step_start 0 \
# --step_end 1000 \
# --block_cache_start 0 \
# --single_block_cache_start 0 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --rel_l1_thresh 0.5 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_0_1000_0_0_2.0_2.0_1_1_0.5 2>&1 &


# CUDA_VISIBLE_DEVICES=1 nohup python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --taylor_max_order 2 \
# --taylor_first_enhance 1 \
# --rel_l1_thresh 0.5 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_2_1_0.5_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --rel_l1_thresh 2 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_1_1_2.0_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --rel_l1_thresh 1 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_1_1_1.0_coco1k.log 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 2 \
# --single_block_rel_l1_thresh 2 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --rel_l1_thresh 0.5 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_1_1_0.5_coco1k.log 2>&1 &



# echo "🔥 Running ULTRA AGGRESSIVE configuration..."
# CUDA_VISIBLE_DEVICES=1  python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 200 \
# --single_block_rel_l1_thresh 200 \
# --rel_l1_thresh 0 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --teablock_taylor \
# --seed 42 \
# --dataset irag \
# --anno_path /root/paddlejob/workspace/env_run/zx/PaddleMIX-westfish-teablockcache/ppdiffusers/examples/TeaBlockCache/temp.txt \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

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

# echo "🔥 Running ULTRA AGGRESSIVE configuration..."
# CUDA_VISIBLE_DEVICES=1  python teablock_generation.py \
# --step_start 50 \
# --step_end 950 \
# --block_cache_start 1 \
# --single_block_cache_start 1 \
# --block_rel_l1_thresh 200 \
# --single_block_rel_l1_thresh 200 \
# --rel_l1_thresh 1 \
# --inference_step 50 \
# --taylor_max_order 1 \
# --taylor_first_enhance 1 \
# --teablock_taylor \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache


