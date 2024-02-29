# flake8: noqa: E116
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

# import types
from dataclasses import dataclass, field

import paddle

# from typing import Optional
from beartype.typing import Literal, Optional, Union

# from paddlenlp.utils.log import logger

VideosOrImagesLiteral = Union[Literal["videos"], Literal["images"]]


# fmt: off
@dataclass
class TrainerArguments:
    """
    Parameters for Trainer except paddlenlp.trainer.TrainingArguments
    """
    # batch_size: int,
        # per_device_train_batch_size
    # num_train_steps: int,
        # max_steps
    # learning_rate: float = 1e-05,
        # learning_rate
    # grad_accum_every: int = 1,
        # gradient_accumulation_steps
    # apply_gradient_penalty_every: int = 4,
        # None
    # max_grad_norm: Optional[float] = None,
        # max_grad_norm
    dataset: Optional[paddle.io.Dataset] = field(
        default=None,
        metadata={"help": "The dataset to use for training."},
    )
    dataset_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset folder."},
    )
    dataset_type: VideosOrImagesLiteral = field(
        default="videos",
        metadata={"help": "dataset_type, one of `videos` or `images`"},
    )
    # checkpoints_folder="./checkpoints",
        # output_dir
    results_folder: Optional[str] = field(
        default="./results",
        metadata={"help": "The folder to save results."},
    )
    # random_split_seed=42,
        # Not used
    valid_frac: float = field(
        default=0.05, 
        metadata={"help": "valid frac"}
    )

    # validate_every_step=100,
        # evaluation_strategy="steps"
        # eval_steps=100
    # checkpoint_every_step=100,
    num_frames: int = field(
        default=17,
        metadata={
            "help": "The number of frames in the input video."
        },
    )
    # use_wandb_tracking=False,
        # None
    # discr_start_after_step=0.0,
        # Do not support
    # warmup_steps=1000,
        # Do not supoort
    # scheduler: Optional[Type[paddle.optimizer.lr.LRScheduler]] = None,
        # scheduler_type
    # scheduler_kwargs: dict = dict(),
        # Do not support
    # ema_kwargs: dict = dict(),
        # Not used
    # optimizer_kwargs: dict = dict(),
        # Do not support, use TrainingArguments' relavant parameters
    dataset_kwargs: dict = field(
        default=dict(),
        metadata={"help": "dataset_kwargs."},
    )

    # new added parameters
    ## advanced control
    flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether or not to use flash attention."},
    )
    dataloader_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether or not to shuffle the dataset."},
    )
    
    ## for custom_visualdl
    resolution: int = field(
        default=256,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        },
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    image_logging_steps: Optional[int] = field(
        default=1000, 
        metadata={"help": "Log image every X steps."}
    )
# fmt: on


"""
TrainerArguments
TrainingArguments
    optimizer
        optim="adamw",
        weight_decay=0.01,
        adam_beta1=0.9, 
        adam_beta2=0.99,
        adam_epsilon=1e-08,
    original
        per_device_train_batch_size=1,
        max_steps=50000,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        max_grad_norm=0.0,
        output_dir="./checkpoints",
        evaluation_strategy="steps",
        eval_steps=100,
        scheduler_type="constant",
    others
        do_eval=False
"""
