CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
--model 'flux' \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco10k/subset \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/tgate_50steps \
--resolution 1024 > output.log 2>&1 &