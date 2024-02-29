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
# import itertools
import math
import os

# import numpy as np
import paddle
from magvit2 import (
    AdvancedVideoTokenizerTrainer,
    ImageDataset,
    TrainerArguments,
    VideoDataset,
    VideoTokenizer,
)
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.utils.log import logger


def exists(v):
    return v is not None


def main():
    parser = PdArgumentParser((TrainerArguments, TrainingArguments))
    trainer_args, training_args = parser.parse_args_into_dataclasses()
    # report to custom_visualdl
    training_args.results_folder = trainer_args.results_folder
    training_args.report_to = ["custom_visualdl"]
    training_args.resolution = trainer_args.resolution
    training_args.benchmark = trainer_args.benchmark
    training_args.profiler_options = trainer_args.profiler_options
    training_args.image_logging_steps = trainer_args.image_logging_steps = (
        (math.ceil(trainer_args.image_logging_steps / training_args.logging_steps) * training_args.logging_steps)
        if trainer_args.image_logging_steps > 0
        else -1
    )
    # advanced control
    training_args.dataloader_shuffle = trainer_args.dataloader_shuffle

    training_args.print_config(training_args, "Training")
    training_args.print_config(trainer_args, "Trainer")

    paddle.set_device(training_args.device)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    if training_args.seed is not None:
        set_seed(training_args.seed)

    model = VideoTokenizer(
        image_size=128,
        init_dim=64,
        max_dim=512,
        use_gan=False,
        use_fsq=False,
        codebook_size=2**18,
        perceptual_loss_weight=0,
        layers=(
            "residual",
            "compress_space",
            ("consecutive_residual", 2),
            "compress_space",
            ("consecutive_residual", 2),
            "linear_attend_space",
            "compress_space",
            ("consecutive_residual", 2),
            "attend_space",
            "compress_time",
            ("consecutive_residual", 2),
            "compress_time",
            ("consecutive_residual", 2),
            "attend_time",
        ),
        flash_attn=trainer_args.flash_attn,
    )

    # Setup data:
    trainer_args.dataset_kwargs.update(channels=model.channels)
    if not exists(trainer_args.dataset):
        if trainer_args.dataset_type == "videos":
            dataset_klass = VideoDataset
            trainer_args.dataset_kwargs = {**trainer_args.dataset_kwargs, "num_frames": trainer_args.num_frames}
        else:
            dataset_klass = ImageDataset
        assert exists(trainer_args.dataset_folder)
        dataset = dataset_klass(
            trainer_args.dataset_folder, image_size=model.image_size, **trainer_args.dataset_kwargs
        )

    assert 0 <= trainer_args.valid_frac < 1.0
    if trainer_args.valid_frac > 0:
        train_size = int((1 - trainer_args.valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        dataset, valid_dataset = paddle.io.random_split(dataset=dataset, lengths=[train_size, valid_size])
        print(
            f"training with dataset of {len(dataset)} samples and validating with randomly splitted {len(valid_dataset)} samples"
        )
    else:
        valid_dataset = dataset
        print(f"training with shared training and valid dataset of {len(dataset)} samples")

    trainer = AdvancedVideoTokenizerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=valid_dataset if training_args.do_eval else None,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
