import os
import argparse
import pickle
import time
from typing_extensions import Self
from numpy import imag
import paddle
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from tqdm import tqdm

from TeaBlockCache_forward import TeaBlockCacheForward
from ppdiffusers import FluxPipeline
from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.utils import load_image, export_to_video

# Try to import other methods for comparison
try:
    from tgate import TgateFLUXLoader
    TGATE_AVAILABLE = True
except ImportError:
    TGATE_AVAILABLE = False
    print("Warning: TGate not available for comparison")

try:
    from teacache_flux import TeaCacheForward
    TEACACHE_AVAILABLE = True
except ImportError:
    TEACACHE_AVAILABLE = False
    print("Warning: TeaCache not available for comparison")

import sys
sys.stdout.isatty = lambda: False

def parse_args():
    parser = argparse.ArgumentParser(description="TeaBlockCache Generation Script for FLUX")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for testing (if not using dataset)",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default='./output/teablock_generation',
        help="The path to save generated images",
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=50,
        help="Total inference steps",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for generation. Set for reproducible results.',
    )
    
    # TeaBlockCache specific parameters
    parser.add_argument(
        "--step_start",
        type=int,
        default=100,
        help="Start timestep for TeaBlockCache caching",
    )
    parser.add_argument(
        "--step_end",
        type=int,
        default=800,
        help="End timestep for TeaBlockCache caching",
    )
    parser.add_argument(
        "--block_cache_start",
        type=int,
        default=5,
        help="Start block index for transformer blocks caching",
    )
    parser.add_argument(
        "--single_block_cache_start",
        type=int,
        default=10,
        help="Start block index for single transformer blocks caching",
    )
    parser.add_argument(
        "--block_rel_l1_thresh",
        type=float,
        default=0.3,
        help="Relative L1 threshold for transformer blocks",
    )
    parser.add_argument(
        "--single_block_rel_l1_thresh",
        type=float,
        default=0.4,
        help="Relative L1 threshold for single transformer blocks",
    )
    
    # Method selection
    parser.add_argument(
        '--origin', 
        action='store_true', 
        default=False, 
        help='Run original FLUX without acceleration',
    )
    parser.add_argument(
        '--teablock', 
        action='store_true', 
        default=False, 
        help='Run TeaBlockCache method',
    )
    parser.add_argument(
        '--teacache', 
        action='store_true', 
        default=False, 
        help='Run TeaCache method for comparison',
    )
    parser.add_argument(
        '--tgate', 
        action='store_true', 
        default=False, 
        help='Run TGate method for comparison',
    )
    
    # TGate parameters (if used)
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="Gate step for TGate method",
    )
    parser.add_argument(
        '--sp_interval',
        type=int,
        default=5,
        help="SP interval for TGate method",
    )
    parser.add_argument(
        '--fi_interval',
        type=int,
        default=1,
        help="FI interval for TGate method",
    )
    parser.add_argument(
        '--warm_up',
        type=int,
        default=2,
        help="Warm up steps for TGate method",
    )
    
    # Dataset parameters
    parser.add_argument(
        "--anno_path",
        type=str,
        default='/root/paddlejob/workspace/env_run/test_data/coco10k/all_prompts.pkl',
        help="Path to evaluation annotations (pkl or tsv file)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='coco10k',
        help="Dataset type: coco10k, coco1k, irag",
    )
    
    args = parser.parse_args()
    return args


def load_prompts(args):
    """Load prompts based on dataset type"""
    if args.dataset == "coco10k":
        all_prompts = pickle.load(open(args.anno_path, "rb"))
        return all_prompts
    
    elif args.dataset == "coco1k":
        import pandas as pd
        df = pd.read_csv(os.path.join(args.anno_path, "coco1k.tsv"), sep="\t")
        assert args.anno_path == "/root/paddlejob/workspace/env_run/test_data/coco1k"
        all_prompts = df['caption_en'].tolist()
        return all_prompts
    
    elif args.dataset == "irag":
        # Read the irag_prompt.txt file and parse prompts
        with open(args.anno_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Split by double newlines to get paragraphs
        all_prompts = [p.strip() for p in content.split("\n\n") if p.strip()]
        return all_prompts
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Use 'coco10k', 'coco1k', or 'irag'.")


def main():
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    
    # Get prompts
    all_prompts = load_prompts(args)
    
    # Create generator if seed is provided
    generator = None
    if args.seed is not None:
        generator = paddle.Generator().manual_seed(args.seed)
    
    # Original FLUX generation
    if args.origin == True:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16
        )
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path, f"origin_{args.inference_step}steps")
        else:
            saved_path = os.path.join(args.saved_path, f"origin_{args.inference_step}steps_{args.dataset}")
        os.makedirs(saved_path, exist_ok=True)
        
        print(f"=== Generating with Original FLUX ({len(all_prompts)} images) ===")
        start_time = time.time()
        
        for i, prompt in enumerate(tqdm(all_prompts, desc="Original FLUX")):
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=args.inference_step,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(all_prompts)
        print(f"Original FLUX: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")
        del pipe

    # TGate method
    if args.tgate == True and TGATE_AVAILABLE:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        pipe = TgateFLUXLoader(pipe)
        saved_path = os.path.join(args.saved_path, f"tgate_{args.inference_step}steps")
        os.makedirs(saved_path, exist_ok=True)
        
        print(f"=== Generating with TGate ({len(all_prompts)} images) ===")
        start_time = time.time()
        
        for i, prompt in enumerate(tqdm(all_prompts, desc="TGate")):
            image = pipe.tgate(
                prompt=prompt,
                height=1024,
                width=1024,
                gate_step=args.gate_step,
                sp_interval=args.sp_interval,
                fi_interval=args.fi_interval,
                warm_up=args.warm_up,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(all_prompts)
        print(f"TGate: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")
        del pipe

    # TeaCache method
    if args.teacache == True and TEACACHE_AVAILABLE:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        
        FluxTransformer2DModel.forward = TeaCacheForward
        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = args.inference_step
        pipe.transformer.rel_l1_thresh = 0.25  # Standard TeaCache threshold
        pipe.transformer.accumulated_rel_l1_distance = 0
        pipe.transformer.previous_modulated_input = None
        pipe.transformer.previous_residual = None
        
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path, f"teacache_{args.inference_step}steps")
        else:
            saved_path = os.path.join(args.saved_path, f"teacache_{args.inference_step}steps_{args.dataset}")
        os.makedirs(saved_path, exist_ok=True)
        
        print(f"=== Generating with TeaCache ({len(all_prompts)} images) ===")
        start_time = time.time()
        
        for i, prompt in enumerate(tqdm(all_prompts, desc="TeaCache")):
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(all_prompts)
        print(f"TeaCache: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")
        del pipe

    # TeaBlockCache method  
    if args.teablock == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        
        # Replace forward method with TeaBlockCache
        FluxTransformer2DModel.forward = TeaBlockCacheForward
        
        # Configure TeaBlockCache parameters
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = args.inference_step
        pipe.transformer.step_start = args.step_start
        pipe.transformer.step_end = args.step_end
        pipe.transformer.block_cache_start = args.block_cache_start
        pipe.transformer.single_block_cache_start = args.single_block_cache_start
        pipe.transformer.block_rel_l1_thresh = args.block_rel_l1_thresh
        pipe.transformer.single_block_rel_l1_thresh = args.single_block_rel_l1_thresh
        
        # Initialize state dictionaries
        pipe.transformer.block_heuristic_states = {}
        pipe.transformer.single_block_heuristic_states = {}
        
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path, f"teablock_{args.inference_step}steps")
        else:
            saved_path = os.path.join(args.saved_path, f"teablock_{args.inference_step}steps_{args.dataset}")
        os.makedirs(saved_path, exist_ok=True)
        
        print(f"=== Generating with TeaBlockCache ({len(all_prompts)} images) ===")
        print(f"TeaBlockCache Configuration:")
        print(f"  Time range: {args.step_start} - {args.step_end}")
        print(f"  Block cache start: {args.block_cache_start}")
        print(f"  Single block cache start: {args.single_block_cache_start}")
        print(f"  Block threshold: {args.block_rel_l1_thresh}")
        print(f"  Single block threshold: {args.single_block_rel_l1_thresh}")
        
        start_time = time.time()
        
        for i, prompt in enumerate(tqdm(all_prompts, desc="TeaBlockCache")):
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(all_prompts)
        print(f"TeaBlockCache: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")
        
        # Report cache statistics
        if hasattr(pipe.transformer, 'block_heuristic_states'):
            num_cached_blocks = len(pipe.transformer.block_heuristic_states)
            print(f"Transformer blocks cached: {num_cached_blocks}")
        
        if hasattr(pipe.transformer, 'single_block_heuristic_states'):
            num_cached_single_blocks = len(pipe.transformer.single_block_heuristic_states)
            print(f"Single blocks cached: {num_cached_single_blocks}")
        
        del pipe

    print("\n🎉 Generation completed!")


if __name__ == '__main__':
    main() 