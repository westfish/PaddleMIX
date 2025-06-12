CUDA_VISIBLE_DEVICES=6  python evaluation_sd3.py \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.0_0.0_coco1k \
--resolution 1024 

CUDA_VISIBLE_DEVICES=6  python evaluation_sd3.py \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.3_0.3_coco1k \
--resolution 1024

CUDA_VISIBLE_DEVICES=6  python evaluation_sd3.py \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_50steps_100_900_3_3_0.9_0.9_coco1k \
--resolution 1024

# CUDA_VISIBLE_DEVICES=4  python evaluation_sd3.py \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_0_1000_0_0_2.0_2.0_1_1_0.5_coco1k \
# --resolution 1024 

# CUDA_VISIBLE_DEVICES=4  python evaluation_sd3.py \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_2_1_0.5_coco1k \
# --resolution 1024

# CUDA_VISIBLE_DEVICES=4  python evaluation_sd3.py \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_1_1_2.0_coco1k \
# --resolution 1024 

# CUDA_VISIBLE_DEVICES=4  python evaluation_sd3.py \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_1_1_1.0_coco1k \
# --resolution 1024 

# CUDA_VISIBLE_DEVICES=4  python evaluation_sd3.py \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_2.0_2.0_1_1_0.5_coco1k \
# --resolution 1024

# CUDA_VISIBLE_DEVICES=4  python evaluation_sd3.py \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_200.0_200.0_1_coco1k \
# --resolution 1024 

# CUDA_VISIBLE_DEVICES=4  python evaluation.py \
# --model 'flux' \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_200.0_200.0_1_coco1k \
# --resolution 1024 

# CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
# --model 'flux' \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco10k/subset \
# --generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/zx/output/computation_cache/teablock_taylor_50steps_50_950_1_1_200.0_200.0_1_coco1k \
# --resolution 1024 > output.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
# --model 'flux' \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco10k/subset \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/tgate_50steps \
# --resolution 1024 > output.log 2>&1 &