# TeaBlockCache: 混合时间-块维度启发式缓存策略

## 概述

TeaBlockCache 是一种创新的缓存策略，结合了 TeaCache 和 BlockCache 的优点：

- **时间维度划分**：像 TeaCache 一样，在特定时间步范围内启用缓存
- **块维度划分**：像 BlockCache 一样，对不同的 transformer 块进行独立控制
- **启发式决策**：像 TeaCache 一样，使用自适应启发式方法决定是否缓存

## 核心创新

### 1. 双维度分区策略
```
时间维度: [step_start, step_end] 范围内启用缓存
块维度:   对每个 transformer block 独立应用启发式
```

### 2. 按块启发式缓存
与 TeaCache 的全局启发式不同，TeaBlockCache 为每个块维护独立的状态：
- `accumulated_distance`: 每个块的累积变化距离
- `previous_modulated_input`: 每个块的前一次调制输入（经过 norm 层处理）
- `cached_output`: 每个块的缓存输出

**重要**: TeaBlockCache 使用 **modulated input**（调制输入）而不是原始输入来计算变化，这与 TeaCache 保持一致，能够更准确地检测语义变化。

### 3. 灵活的阈值配置
可以为不同类型的块设置不同的阈值：
- `block_rel_l1_thresh`: transformer_blocks 的阈值
- `single_block_rel_l1_thresh`: single_transformer_blocks 的阈值

## 算法工作流程

```python
for each timestep:
    if timestep in [step_start, step_end]:
        for each block >= cache_start_index:
            # 计算调制输入（经过 norm 层处理的输入）
            modulated_input = block.norm(input, emb=time_emb)
            
            # 计算调制输入的变化
            rel_change = compute_relative_change(modulated_input, previous_modulated_input)
            
            # 应用启发式函数
            accumulated_distance += rescale_function(rel_change)
            
            # 启发式决策
            if accumulated_distance < threshold:
                use_cached_output()
            else:
                recompute_block()
                reset_accumulated_distance()
                cache_new_output()
```

## 配置参数

### 时间维度参数
- `step_start`: 开始缓存的时间步 (例如: 100)
- `step_end`: 结束缓存的时间步 (例如: 800)
- `num_steps`: 总推理步数 (例如: 50)

### 块维度参数
- `block_cache_start`: 开始缓存的 transformer block 索引 (例如: 5)
- `single_block_cache_start`: 开始缓存的 single block 索引 (例如: 10)

### 启发式参数
- `block_rel_l1_thresh`: transformer blocks 的相对L1阈值 (例如: 0.3)
- `single_block_rel_l1_thresh`: single blocks 的相对L1阈值 (例如: 0.4)

## 使用示例

```python
from TeaBlockCache_forward import TeaBlockCacheForward
from ppdiffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel

# 加载管道
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)

# 替换forward方法
FluxTransformer2DModel.forward = TeaBlockCacheForward

# 配置缓存参数
pipe.transformer.cnt = 0
pipe.transformer.num_steps = 50
pipe.transformer.step_start = 100
pipe.transformer.step_end = 800
pipe.transformer.block_cache_start = 5
pipe.transformer.block_rel_l1_thresh = 0.3
pipe.transformer.single_block_cache_start = 10
pipe.transformer.single_block_rel_l1_thresh = 0.4

# 运行推理
image = pipe("A cat in Picasso style", num_inference_steps=50).images[0]
```

## 性能调优

### 加速优化策略
1. **增加阈值**: 更高的阈值 → 更多缓存 → 更快速度
   ```python
   pipe.transformer.block_rel_l1_thresh = 0.5  # 更激进的缓存
   pipe.transformer.single_block_rel_l1_thresh = 0.6
   ```

2. **提前开始缓存**: 更小的 cache_start → 更多块被缓存
   ```python
   pipe.transformer.block_cache_start = 3  # 从第3个块开始缓存
   pipe.transformer.single_block_cache_start = 5
   ```

### 质量优化策略
1. **降低阈值**: 更低的阈值 → 更少缓存 → 更高质量
   ```python
   pipe.transformer.block_rel_l1_thresh = 0.2  # 更保守的缓存
   pipe.transformer.single_block_rel_l1_thresh = 0.25
   ```

2. **延迟开始缓存**: 更大的 cache_start → 更少块被缓存
   ```python
   pipe.transformer.block_cache_start = 10  # 从第10个块开始缓存
   pipe.transformer.single_block_cache_start = 20
   ```

## 与现有方法对比

| 特性 | TeaCache | BlockCache | TeaBlockCache |
|------|----------|------------|---------------|
| 时间维度分区 | ✅ | ✅ | ✅ |
| 块维度分区 | ❌ | ✅ | ✅ |
| 启发式决策 | ✅ | ❌ | ✅ |
| 自适应性 | 高 | 低 | 很高 |
| 配置复杂度 | 低 | 中 | 中 |
| 性能潜力 | 中 | 中 | 高 |

## 启发式函数

TeaBlockCache 使用与 TeaCache 相同的启发式重缩放函数：

```python
coefficients = [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01]
rescale_func = np.poly1d(coefficients)
scaled_change = rescale_func(relative_l1_change)
```

这个函数将原始的相对L1变化映射到一个更合适的缓存决策空间。

## 最佳实践

1. **首次运行**: 第一次运行会建立缓存状态，可能不会显示加速效果
2. **连续运行**: 后续运行将充分利用缓存，显示真实的性能提升
3. **参数调优**: 根据具体用例调整阈值以平衡速度和质量
4. **监控**: 可以添加日志来监控每个块的缓存命中率

## 技术细节

### Modulated Input 的重要性
TeaBlockCache 使用 **modulated input** 进行变化检测，这是经过每个块的 norm 层处理后的输入：
- 对于 transformer_blocks：`modulated_inp = block.norm1(input, emb=time_emb)`
- 对于 single_transformer_blocks：`modulated_inp = block.norm(input, emb=time_emb)`

这样做的优势：
1. **包含时间信息**：modulated input 融合了时间嵌入，反映当前时间步的语义
2. **更准确的变化检测**：norm 层处理后的数据更直接反映实际的语义变化
3. **与 TeaCache 一致**：保持相同的启发式质量和行为

### 内存管理
- 每个块维护独立的缓存状态
- 在生成开始时自动重置状态
- 缓存的张量会克隆以避免意外修改
- modulated input 的计算会增加少量额外开销，但提高了缓存决策的准确性

### 兼容性
- 完全兼容现有的 FluxTransformer2DModel
- 支持 gradient checkpointing
- 支持 ControlNet 等扩展功能

## 故障排除

### 常见问题
1. **内存使用过高**: 减少 cache_start 参数
2. **质量下降**: 降低阈值参数
3. **加速不明显**: 增加阈值或减少 cache_start

### 调试建议
- 添加打印语句监控缓存决策
- 可视化不同块的累积距离变化
- 比较不同配置下的生成质量 