# SortTaylor Optimization Integration

This document explains how to use the newly integrated SortTaylor optimization framework for FLUX models.

## Overview

SortTaylor is an optimization technique that uses Taylor series approximation to skip certain transformer block computations during inference, reducing computational cost while maintaining output quality. It has been integrated into the ppdiffusers framework using a similar approach to Pyramid Attention Broadcast (PAB).

## Usage

### Method 1: Using apply_sort_taylor function

```python
import paddle
from ppdiffusers import FluxPipeline, SortTaylorConfig, apply_sort_taylor

# Load the pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)

# Configure SortTaylor optimization
config = SortTaylorConfig(
    num_inference_steps=50,
    timestep_start=900,
    timestep_end=100,
    percentage=1.0,
    step_num=1,
    step_num2=5,
    beta=0.3,
    current_timestep_callback=lambda: pipe._current_timestep,
)

# Apply SortTaylor optimization
apply_sort_taylor(pipe.transformer, config)

# Generate images with optimization
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(42)
).images[0]
```

### Method 2: Using enable_cache method

```python
import paddle
from ppdiffusers import FluxPipeline, SortTaylorConfig

# Load the pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)

# Configure and enable SortTaylor optimization
config = SortTaylorConfig(
    timestep_start=900,
    timestep_end=100,
    beta=0.3,
    current_timestep_callback=lambda: pipe._current_timestep,
)

pipe.transformer.enable_cache(config)

# Generate images with optimization
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(42)
).images[0]

# Disable optimization when done
pipe.transformer.disable_cache()
```

## Configuration Parameters

### SortTaylorConfig

- `num_inference_steps` (int, default=50): The number of denoising steps
- `timestep_start` (int, default=900): The timestep to start applying optimization
- `timestep_end` (int, default=100): The timestep to end applying optimization
- `percentage` (float, default=1.0): The percentage of blocks to compute in each layer
- `step_num` (int, default=1): The step number for normal operation
- `step_num2` (int, default=5): The step number when within the timestep range
- `beta` (float, default=0.3): The beta parameter for rescaling
- `current_timestep_callback` (Callable, optional): A callback function that returns the current inference timestep

## Key Features

1. **Framework Integration**: SortTaylor is now fully integrated into the ppdiffusers hook system
2. **Unified API**: Uses the same pattern as other optimizations like Pyramid Attention Broadcast
3. **Easy Configuration**: Simple configuration class with sensible defaults
4. **Cache Management**: Supports enable/disable cache functionality
5. **State Management**: Automatic state reset and management for multiple inferences

## Examples

- Basic usage: `text_to_image_generation-flux-sort_taylor.py`
- Compare with original approach: `diffusers_sorttaylor_flux.py` (legacy)

## Migration from Legacy Code

If you were previously using the manual approach from `diffusers_sorttaylor_flux.py`, you can now replace the manual setup:

```python
# Legacy approach (manual setup)
pipeline.transformer.__class__.forward = SortTaylor_forward
pipeline.transformer.current_block_residual = [None] * len(pipeline.transformer.transformer_blocks)
# ... many more manual attribute assignments

# New integrated approach
config = SortTaylorConfig(beta=0.3, timestep_start=900, timestep_end=100)
apply_sort_taylor(pipeline.transformer, config)
```

The new approach handles all the setup automatically and provides better state management. 