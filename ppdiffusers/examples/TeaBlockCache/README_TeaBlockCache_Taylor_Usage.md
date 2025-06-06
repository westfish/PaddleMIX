# TeaBlockCache + Taylor 展开使用指南

## 🎯 概述

现在 `teablock_generation.py` 已经支持 TeaBlockCache + Taylor 展开缓存复用机制！这是一个结合了 per-block 启发式缓存和泰勒级数预测的高级加速方法。

## 🚀 快速开始

### 基础用法

```bash
python teablock_generation.py --teablock_taylor --prompt "A beautiful landscape" --inference_step 20
```

### 完整配置示例

```bash
python teablock_generation.py \
    --teablock_taylor \
    --prompt "A majestic mountain sunset with vibrant colors" \
    --saved_path "./output/taylor_test" \
    --inference_step 20 \
    --seed 42 \
    --step_start 100 \
    --step_end 800 \
    --block_cache_start 5 \
    --single_block_cache_start 10 \
    --block_rel_l1_thresh 0.3 \
    --single_block_rel_l1_thresh 0.4 \
    --taylor_max_order 3 \
    --taylor_first_enhance 2 \
    --dataset coco10k
```

### 使用示例脚本

```bash
chmod +x run_teablock_taylor_example.sh
./run_teablock_taylor_example.sh
```

## ⚙️ 参数配置

### TeaBlockCache 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--step_start` | 100 | 开始缓存的时间步 |
| `--step_end` | 800 | 结束缓存的时间步 |
| `--block_cache_start` | 5 | Transformer blocks 开始缓存的索引 |
| `--single_block_cache_start` | 10 | Single blocks 开始缓存的索引 |
| `--block_rel_l1_thresh` | 0.3 | Transformer blocks 的相对L1阈值 |
| `--single_block_rel_l1_thresh` | 0.4 | Single blocks 的相对L1阈值 |

### Taylor 展开参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--taylor_max_order` | 3 | 泰勒展开的最大阶数 |
| `--taylor_first_enhance` | 2 | 开始使用导数增强的步数 |

## 📊 方法对比

### 支持的方法

1. **`--origin`** - 原始 FLUX（无加速）
2. **`--teablock`** - TeaBlockCache（per-block 启发式缓存）
3. **`--teablock_taylor`** - TeaBlockCache + Taylor（✨ 新增）
4. **`--teacache`** - TeaCache（全局启发式缓存）
5. **`--tgate`** - TGate（结构化剪枝）

### 性能对比示例

```bash
# 运行所有方法对比
python teablock_generation.py --origin --prompt "test prompt" --inference_step 20
python teablock_generation.py --teablock --prompt "test prompt" --inference_step 20  
python teablock_generation.py --teablock_taylor --prompt "test prompt" --inference_step 20
```

## 🔬 技术特性

### TeaBlockCache + Taylor 的优势

1. **数学严谨性**：使用泰勒级数而非简单缓存复用
2. **Per-block 精度**：在每个 transformer block 层面进行预测
3. **自适应性**：根据变化趋势动态调整预测
4. **数值稳定性**：内置异常处理和回退机制

### 工作原理

```python
# 启发式判断
if block_change < threshold:
    # 使用泰勒级数预测
    predicted_output = taylor_predict(cache_coefficients, time_delta)
else:
    # 重新计算并更新泰勒系数
    actual_output = compute_block(...)
    update_taylor_cache(actual_output)
```

## 📈 输出和统计

运行完成后会显示详细统计信息：

```
=== Generating with TeaBlockCache + Taylor (100 images) ===
TeaBlockCache + Taylor Configuration:
  Time range: 100 - 800
  Block cache start: 5
  Single block cache start: 10
  Block threshold: 0.3
  Single block threshold: 0.4
  Taylor max order: 3
  Taylor first enhance: 2

TeaBlockCache + Taylor: Total 45.67s, Avg 0.46s/image
Transformer blocks cached: 19
Single blocks cached: 38
Taylor cache activated steps: 15
Taylor cache coefficients stored: 4
```

## 🛠️ 调优建议

### 保守设置（高质量）

```bash
--block_rel_l1_thresh 0.2 \
--single_block_rel_l1_thresh 0.3 \
--taylor_max_order 2
```

### 激进设置（高速度）

```bash
--block_rel_l1_thresh 0.4 \
--single_block_rel_l1_thresh 0.5 \
--taylor_max_order 3 \
--block_cache_start 3
```

### 平衡设置（推荐）

```bash
--block_rel_l1_thresh 0.3 \
--single_block_rel_l1_thresh 0.4 \
--taylor_max_order 3 \
--taylor_first_enhance 2
```

## ⚠️ 注意事项

1. **内存使用**：Taylor 缓存会消耗额外内存存储系数
2. **数值稳定性**：高阶Taylor展开可能在某些情况下不稳定
3. **计算开销**：导数计算会带来少量额外开销
4. **兼容性**：需要正确的 `TeaBlockCache_taylor_forward.py` 文件

## 🔧 故障排除

### 常见问题

1. **导入错误**
   ```
   Warning: TeaBlockCache Taylor not available for comparison
   ```
   解决：确保 `TeaBlockCache_taylor_forward.py` 文件存在且语法正确

2. **数值不稳定**
   ```
   ⚠️ Block X: 泰勒预测失败，强制计算
   ```
   解决：降低 `taylor_max_order` 或调整阈值

3. **内存不足**
   解决：减少 `taylor_max_order` 或增加 `taylor_first_enhance`

## 📝 示例输出

成功运行时会看到类似输出：

```
✅ Block 0: 使用泰勒级数预测
✅ Block 1: 使用泰勒级数预测
⚠️ Block 2: 泰勒预测失败，强制计算
✅ Single Block 0: 使用泰勒级数预测
```

这表明系统正在智能地选择使用泰勒预测还是重新计算。

## 🎉 结语

TeaBlockCache + Taylor 展开为 FLUX 模型推理提供了数学上严谨且实用的加速方案。通过合理的参数调优，可以在保持图像质量的同时显著提升生成速度！ 