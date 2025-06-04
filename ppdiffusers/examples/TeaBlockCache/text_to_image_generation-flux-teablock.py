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

import time

import paddle
from TeaBlockCache_forward import TeaBlockCacheForward 
from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

# Load the pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)

# Replace the forward method with our hybrid caching strategy
FluxTransformer2DModel.forward = TeaBlockCacheForward

# Configure the hybrid caching parameters
# Time dimension parameters (similar to TeaCache)
pipe.transformer.cnt = 0
pipe.transformer.num_steps = 50
pipe.transformer.step_start = 100  # Start caching from this timestep
pipe.transformer.step_end = 800    # End caching at this timestep

# Block dimension parameters for transformer_blocks
pipe.transformer.block_cache_start = 5  # Start caching from this block index
pipe.transformer.block_rel_l1_thresh = 0.3  # Heuristic threshold for transformer blocks

# Block dimension parameters for single_transformer_blocks  
pipe.transformer.single_block_cache_start = 10  # Start caching from this block index
pipe.transformer.single_block_rel_l1_thresh = 0.4  # Heuristic threshold for single blocks

# Initialize state dictionaries (will be auto-initialized but good to be explicit)
pipe.transformer.block_heuristic_states = {}
pipe.transformer.single_block_heuristic_states = {}

def run_inference(prompt, description=""):
    """Run inference and measure time."""
    start_time = time.time()
    
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{description} - Elapsed time: {elapsed_time:.2f} seconds")
    
    return image

# Test the hybrid caching strategy
prompt = "An image of a squirrel in Picasso style"

print("=== TeaBlockCache Hybrid Strategy Demo ===")
print("This strategy combines:")
print("- Time dimension partitioning (when to cache)")
print("- Block dimension partitioning (which blocks to cache)")  
print("- Heuristic decision making (adaptive caching)")
print()

# First run (will build up the cache states)
image1 = run_inference(prompt, "First run (cache building)")

# Second run (should benefit from caching)
image2 = run_inference(prompt, "Second run (cache utilization)")

# Save the result
image2.save("teablock_cache_flux_result.png")

print()
print("=== Configuration Summary ===")
print(f"Time range for caching: {pipe.transformer.step_start} - {pipe.transformer.step_end}")
print(f"Transformer blocks cache start: {pipe.transformer.block_cache_start}")
print(f"Transformer blocks threshold: {pipe.transformer.block_rel_l1_thresh}")
print(f"Single blocks cache start: {pipe.transformer.single_block_cache_start}")  
print(f"Single blocks threshold: {pipe.transformer.single_block_rel_l1_thresh}")
print()
print("=== Strategy Explanation ===")
print("TeaBlockCache works by:")
print("1. Monitoring input changes for each block individually")
print("2. Using heuristics to decide whether to recompute or use cached results")
print("3. Applying different thresholds for different types of blocks")
print("4. Operating within specified time step ranges for efficiency")
print("5. Always computing first and last steps for quality")

# Test with different thresholds for comparison
print()
print("=== Testing Different Threshold Settings ===")

# More aggressive caching (higher thresholds = more caching)
pipe.transformer.block_rel_l1_thresh = 0.5
pipe.transformer.single_block_rel_l1_thresh = 0.6
image3 = run_inference(prompt, "Aggressive caching (higher thresholds)")

# More conservative caching (lower thresholds = less caching, higher quality)
pipe.transformer.block_rel_l1_thresh = 0.2
pipe.transformer.single_block_rel_l1_thresh = 0.25
image4 = run_inference(prompt, "Conservative caching (lower thresholds)")

print()
print("Image saved as: teablock_cache_flux_result.png")
print("Experiment complete!") 