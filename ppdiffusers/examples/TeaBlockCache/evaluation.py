import os
import argparse
import paddle

from common_metrics.fid_score import ImagePathDataset,calculate_fid_given_paths
from common_metrics.inception import InceptionV3
from common_metrics.calculate_ssim import calculate_ssim_function
from common_metrics.calculate_psnr import img_psnr
from ppdiffusers import StableDiffusionXLPipeline, PixArtAlphaPipeline, StableVideoDiffusionPipeline
from ppdiffusers import UNet2DConditionModel, LCMScheduler,FluxPipeline
from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.utils import load_image, export_to_video

import paddle.vision.transforms as TF
from tqdm import tqdm 
import pathlib
import re
import numpy as np


# 使用正则表达式提取文件名中的数字部分
def extract_number(filename):
    filename = os.path.basename(filename)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE V2.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the input prompts",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="the dir of input image to generate video",
    )
    # parser.add_argument(
    #     "--saved_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to save the generated results.",
    # )
    parser.add_argument(
        "--model",
        type=str,
        default='pixart',
        help="[pixart_alpha,sdxl,lcm_sdxl,lcm_pixart_alpha,svd]",
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=25,
        help="total inference steps",
    )
    parser.add_argument(
        '--deepcache', 
        action='store_true', 
        default=False, 
        help='do deep cache',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for generation. Set for reproducible results.',
    )
    parser.add_argument(
        '--tgate', 
        action='store_true', 
        default=False, 
        help='do add tgate',
    )
    parser.add_argument(
        "--training_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated results.",
    )
    parser.add_argument(
        "--generation_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated results.",
    )
    parser.add_argument(
        "--speed_generation_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the speed up generated results.",
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=None, 
        help="The resolution to resize."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--fid_batch_size", type=int, default=128, help="Batch size to use")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for data loading")
    parser.add_argument(
    "--dims",
    type=int,
    default=2048,
    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    help=("Dimensionality of Inception features to use. " "By default, uses pool3 features"),
)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 获取原始方法生成的图片
    gen_path = pathlib.Path(args.generation_path)
    # 进行排序 因为psnr 与 ssim需要同prompt 图片进行对比
    gen_files = sorted([file for ext in IMAGE_EXTENSIONS for file in gen_path.glob("*.{}".format(ext))],key=extract_number)
    dataset_gen = ImagePathDataset(gen_files, transforms=TF.ToTensor(), resolution=args.resolution)
    dataloader_gen = paddle.io.DataLoader(
        dataset_gen,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    #获取使用推理加速的方法
    speedgen_path = pathlib.Path(args.speed_generation_path)
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in speedgen_path.glob("*.{}".format(ext))],key=extract_number)
    dataset_speedgen = ImagePathDataset(files, transforms=TF.ToTensor(), resolution=args.resolution)
    dataloader_speedgen = paddle.io.DataLoader(
        dataset_speedgen,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    print(len(dataloader_gen))
    print(len(dataloader_speedgen))
    ssim_value_list=[]
    psnr_value_list=[]
    # 计算ssim与psnr
    for batch_gen, batch_speedgen in tqdm(zip(dataloader_gen, dataloader_speedgen),
                                       total=len(dataloader_gen),
                                       desc="Calculating SSIM and PSNR"):
        batch_speedgen = batch_speedgen["img"]
        batch_gen = batch_gen["img"]
        batch_speedgen = batch_speedgen.squeeze().numpy()  # 将Tensor转换为numpy数组，并调整通道顺序
        batch_gen = batch_gen.squeeze().numpy()
        ssim_value = calculate_ssim_function(batch_gen,batch_speedgen)
        psnr_value = img_psnr(batch_gen,batch_speedgen)
        ssim_value_list.append(ssim_value)
        psnr_value_list.append(psnr_value)
    # 计算fid
    fid_value_origin = calculate_fid_given_paths(
        [args.training_path,args.generation_path], args.fid_batch_size,args.dims, args.num_workers, resolution=args.resolution
    )
    fid_value_speed = calculate_fid_given_paths(
        [args.training_path,args.speed_generation_path], args.fid_batch_size,args.dims, args.num_workers, resolution=args.resolution
    )
    mean_ssim = np.mean(ssim_value_list)
    mean_psnr = np.mean(psnr_value_list)
    from pathlib import Path

    path = Path(args.generation_path)
    parent_path = path.parent
    # os.makedirs(save_dir, exist_ok=True)
    # 将对应的指标保存起来
    res_txt = os.path.basename(args.speed_generation_path)
    with open(os.path.join(parent_path, f"{res_txt}.txt"), "w") as f:  # ← 注意这里用 "a"
        f.write(f"mean_ssim: {mean_ssim}\n")
        f.write(f"mean_psnr: {mean_psnr}\n")
        f.write(f"fid_score_origin: {fid_value_origin}\n")
        f.write(f"fid_score_speed: {fid_value_speed}\n")
        #f.write("-" * 40 + "\n")  # 分隔线，方便查看日志
    print('mean_ssim: ',mean_ssim,'mean_psnr: ',mean_psnr,'fid_score_origin: ',fid_value_origin,'fid_score_speed:',fid_value_speed)