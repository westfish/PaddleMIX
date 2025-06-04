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
from BlockCache_forward import BlockCacheForward 
from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)

FluxTransformer2DModel.forward = BlockCacheForward
num_dual_blocks=19
num_single_blocks=38
pipe.transformer.previous_block = None
pipe.transformer.previous_block_encoder = None
pipe.transformer.previous_single_block = None
pipe.transformer.step_start = 100
pipe.transformer.step_end = 800
pipe.transformer.block_step_single = 28
pipe.transformer.block_step = 13
pipe.transformer.block_step_N = 4
pipe.transformer.count = 0


start_time = time.time()
# prompt = "A cat holding a sign that says hello world"
prompt = "An image of a squirrel in Picasso style"
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
print(f"Elapsed time: {elapsed_time:.2f} seconds")
####
start_time = time.time()
# prompt = "A cat holding a sign that says hello world"
prompt = "An image of a squirrel in Picasso style"
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
print(f"Elapsed time: {elapsed_time:.2f} seconds")

image.save("text_to_image_generation-flux-dev-result.png")
