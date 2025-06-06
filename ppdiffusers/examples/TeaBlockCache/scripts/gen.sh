# export FLAGS_sdpa_select_math=yes 

CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
--step_start 100 \
--step_end 900 \
--block_cache_start 3 \
--single_block_cache_start 3 \
--block_rel_l1_thresh 0.9 \
--single_block_rel_l1_thresh 0.9 \
--inference_step 50 \
--taylor_max_order 3 \
--taylor_first_enhance 2 \
--teablock_taylor \
--seed 124 \
--dataset coco1k \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache


# CUDA_VISIBLE_DEVICES=4 nohup python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.0_0.0_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.3 \
# --single_block_rel_l1_thresh 0.3 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.3_0.3_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.9 \
# --single_block_rel_l1_thresh 0.9 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.9_0.9_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 nohup python teablock_generation.py \
# --inference_step 50 \
# --origin \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 python teablock_generation.py \
# --inference_step 50 \
# --origin \
# --seed 124 \
# --dataset irag \
# --anno_path prompt.txt \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=2 python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset irag \
# --anno_path prompt.txt \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=2 python teablock_generation.py \
# --step_start 000 \
# --step_end 1000 \
# --block_cache_start 0 \
# --single_block_cache_start 0 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 nohup python teablock_generation.py \
# --inference_step 50 \
# --origin \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache > /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 0 \
# --step_end 1000 \
# --block_cache_start 0 \
# --single_block_cache_start 0 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.0 \
# --single_block_rel_l1_thresh 0.0 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache

# CUDA_VISIBLE_DEVICES=1 python teablock_generation.py \
# --step_start 100 \
# --step_end 900 \
# --block_cache_start 3 \
# --single_block_cache_start 3 \
# --block_rel_l1_thresh 0.3 \
# --single_block_rel_l1_thresh 0.3 \
# --inference_step 50 \
# --teablock \
# --seed 124 \
# --dataset coco1k \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --saved_path /root/paddlejob/workspace/env_run/zx/output/computation_cache






# CUDA_VISIBLE_DEVICES=1 nohup python generation.py \Add commentMore actions
# --model 'flux' \
# --gate_step 25 \
# --sp_interval 5 \
# --fi_interval 1 \
# --warm_up 2 \
# --inference_step 50 \
# --seed 124 \
# --tgate > output_tgate_50steps.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python generation.py \
# --model 'flux' \
# --gate_step 25 \
# --sp_interval 5 \
# --fi_interval 1 \
# --warm_up 2 \
# --inference_step 50 \
# --seed 124 \
# --origin > output_50steps.log 2>&1 &

# ### coco1k

# CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
# --model 'flux' \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --teacache > output.log 2>&1 &



# CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
# --model 'flux' \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --origin > output_coco1k.log 2>&1 &