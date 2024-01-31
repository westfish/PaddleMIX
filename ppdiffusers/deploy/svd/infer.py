# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import random
import time

import paddle

# isort: split
import numpy as np
import paddle.inference as paddle_infer
from paddlenlp.trainer.argparser import strtobool
from tqdm.auto import trange

from ppdiffusers import (  # noqa
    DiffusionPipeline,
    PaddleInferStableVideoDiffusionPipeline,
)
from ppdiffusers.utils import load_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="runwayml/stable-diffusion-v1-5@paddleinfer",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
        help="The number of unet inference steps.",
    )
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=1,
        help="The number of performance benchmark steps.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle_tensorrt",
        # Note(zhoushunjie): Will support 'tensorrt' soon.
        choices=["onnx_runtime", "paddle", "paddlelite", "paddle_tensorrt"],
        help="The inference runtime backend of unet model and text encoder model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        # Note(shentanyue): Will support more devices.
        choices=[
            "cpu",
            "gpu",
            "huawei_ascend_npu",
            "kunlunxin_xpu",
        ],
        help="The inference runtime device of models.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="text2img",
        choices=[
            "text2img",
            "img2video",
            "inpaint",
            "all",
        ],
        help="The task can be one of [text2img, img2video, inpaint, pix2pix, all]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--use_bf16", type=strtobool, default=False, help="Wheter to use BF16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="preconfig-euler-ancestral",
        choices=[
            "pndm",
            "lms",
            "euler",
            "euler-ancestral",
            "preconfig-euler-ancestral",
            "dpm-multi",
            "dpm-single",
            "unipc-multi",
            "ddim",
            "ddpm",
            "deis-multi",
            "heun",
            "kdpm2-ancestral",
            "kdpm2",
        ],
        help="The scheduler type of stable diffusion.",
    )
    parser.add_argument(
        "--infer_op",
        type=str,
        default="zero_copy_infer",
        choices=[
            "zero_copy_infer",
            "raw",
            "all",
        ],
        help="The type of infer op.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of input image")
    parser.add_argument("--width", type=int, default=576, help="Width of input image")

    return parser.parse_args()


def create_paddle_inference_runtime(
    model_dir="",
    model_name="",
    use_trt=False,
    precision_mode=paddle_infer.PrecisionType.Half,
    device_id=0,
    disable_paddle_trt_ops=[],
    disable_paddle_pass=[],
    workspace=24 * 1024 * 1024 * 1024,
    tune=False,
):
    config = paddle_infer.Config()
    config.enable_new_executor()
    config.enable_memory_optim()
    shape_file = f"{model_dir}/{model_name}/shape_range_info.pbtxt"
    if tune:
        config.collect_shape_range_info(shape_file)
    if device_id != -1:
        config.use_gpu()
        config.enable_use_gpu(memory_pool_init_size_mb=2000, device_id=device_id, precision_mode=precision_mode)
    for pass_name in disable_paddle_pass:
        config.delete_pass(pass_name)
    if use_trt:
        if not os.path.exists(shape_file):
            config.collect_shape_range_info(shape_file)
        config.enable_tensorrt_engine(
            workspace_size=workspace,
            precision_mode=precision_mode,
            max_batch_size=1,
            min_subgraph_size=3,
            use_static=True,
        )
        config.enable_tensorrt_memory_optim()
        config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)
        if precision_mode != paddle_infer.PrecisionType.Half:
            only_fp16_passes = [
                "trt_cross_multihead_matmul_fuse_pass",
                "trt_flash_multihead_matmul_fuse_pass",
                "preln_elementwise_groupnorm_act_pass",
                "elementwise_groupnorm_act_pass",
                "conv_elementwise_add_fuse_pass",
            ]
            for curr_pass in only_fp16_passes:
                config.delete_pass(curr_pass)
    return config


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")
    seed = 1024

    disable_paddle_pass = [
        "auto_mixed_precision_pass",
        "conv_elementwise_add_fuse_pass",
        "gpu_cpu_map_matmul_to_mul_pass",
    ]
    # only_fp16_passes = [
    #     "trt_cross_multihead_matmul_fuse_pass",
    #     "trt_flash_multihead_matmul_fuse_pass",
    #     "preln_elementwise_groupnorm_act_pass",
    #     "elementwise_groupnorm_act_pass",
    # ]
    no_need_passes = [
        "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",
        "add_support_int8_pass",
        "auto_mixed_precision_pass",
        # "conv_elementwise_add_fuse_pass",
    ]
    # conv_bias_mkldnn_fuse_pass,
    args.use_trt = args.backend == "paddle_tensorrt"
    infer_configs = dict(
        vae_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_encoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            tune=False,
        ),
        vae_decoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_decoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_pass=disable_paddle_pass,
            tune=False,
        ),
        unet=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="unet",
            use_trt=args.use_trt,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_pass=no_need_passes,
            tune=False,
        ),
        image_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="image_encoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            tune=False,
        ),
    )
    pipe = PaddleInferStableVideoDiffusionPipeline.from_pretrained(
        args.model_dir,
        infer_configs=infer_configs,
        use_optim_cache=False,
    )
    pipe.set_progress_bar_config(disable=False)
    # pipe.change_scheduler(args.scheduler)
    width = args.width
    height = args.height

    if args.infer_op == "all":
        infer_op_list = ["zero_copy_infer", "raw"]
    else:
        infer_op_list = [args.infer_op]
    if args.device == "kunlunxin_xpu" or args.backend == "paddle":
        print("When device is kunlunxin_xpu or backend is paddle, we will use `raw` infer op.")
        infer_op_list = ["raw"]

    for infer_op in infer_op_list:
        folder = f"infer_op_{infer_op}_fp16" if args.use_fp16 else f"infer_op_{infer_op}_fp32"
        os.makedirs(folder, exist_ok=True)

        if args.task_name in ["img2video", "all"]:
            # img2video
            img_url = (
                "https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/rocket.png"
            )
            init_image = load_image(img_url)
            time_costs = []
            # warmup
            print("==> Warmup.")
            pipe(
                image=init_image,
                num_inference_steps=3,
                height=height,
                width=width,
            )
            print("==> Test img2video performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                frames = pipe(
                    image=init_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                ).frames
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            frames[0][0].save("test_svd.gif", save_all=True, append_images=frames[0][1:], loop=0)


if __name__ == "__main__":
    seed = 2024
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = parse_arguments()
    main(args)
