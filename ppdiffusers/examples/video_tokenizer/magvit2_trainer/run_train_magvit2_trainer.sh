# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

log_dir="/root/project/paddlemix/magvit2/zhangxu/drawer/log"
# python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir ${log_dir} train_magvit2_trainer.py \
python -u train_magvit2_trainer.py \
    --optim "adamw" \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --adam_epsilon 1e-08 \
    --per_device_train_batch_size 1 \
    --max_steps 50000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 0.0 \
    --output_dir "./checkpoints" \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --lr_scheduler_type "constant" \
    --do_eval False \
    --fp16_opt_level "O1" \
    --bf16 False \
    --fp16 True \
    --recompute True \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
    --logging_steps 50 \
    --logging_dir "./logging", \
    --seed 0 \
    --overwrite_output_dir True \
    --warmup_steps 0 \
    --disable_tqdm True \
    --dataset_folder "/root/project/paddlemix/magvit2/magvit2-pytorch/data/training_data/videos" \
    --dataset_type "videos" \
    --results_folder "./results" \
    --valid_frac 0.05 \
    --num_frames 17 \
    --flash_attn False \
    --dataloader_shuffle True \
    --resolution 256 \
    --benchmark False \
    --image_logging_steps 1000