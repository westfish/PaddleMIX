# TeaBlockCache 性能优化报告

## 概述
- 总分析时间: 5.122188s
- 分析组件数: 17

## 性能热点 (Top 5)
1. **total_forward**: 28.23% (1.446207s)
2. **single_transformer_blocks_total**: 11.66% (0.597317s)
3. **single_block_processing**: 11.66% (0.597140s)
4. **single_block_computation**: 11.41% (0.584357s)
5. **transformer_blocks_total**: 10.24% (0.524474s)

## 优化建议

### 🔴 高优先级: total_forward (28.23%)

### 🟡 中优先级: single_transformer_blocks_total (11.66%)
- 考虑局部优化

### 🟡 中优先级: single_block_processing (11.66%)
- 考虑局部优化

### 🟡 中优先级: single_block_computation (11.41%)
- 考虑局部优化

### 🟡 中优先级: transformer_blocks_total (10.24%)
- 考虑局部优化

### 🟡 中优先级: single_transformer_block (10.24%)
- 考虑局部优化

### 🟡 中优先级: block_computation (10.09%)
- 考虑局部优化

## 详细数据
| 组件名称 | 总时间(s) | 平均时间(s) | 占比(%) | 调用次数 |
|----------|-----------|-------------|---------|----------|
| total_forward | 1.446207 | 0.289241 | 28.23 | 5 |
| single_transformer_blocks_total | 0.597317 | 0.119463 | 11.66 | 5 |
| single_block_processing | 0.597140 | 0.003143 | 11.66 | 190 |
| single_block_computation | 0.584357 | 0.003076 | 11.41 | 190 |
| transformer_blocks_total | 0.524474 | 0.104895 | 10.24 | 5 |
| single_transformer_block | 0.524398 | 0.005244 | 10.24 | 100 |
| block_computation | 0.516881 | 0.005169 | 10.09 | 100 |
| state_concatenation | 0.289514 | 0.057903 | 5.65 | 5 |
| cache_state_initialization | 0.032205 | 0.006441 | 0.63 | 5 |
| heuristic_computation | 0.003582 | 0.000036 | 0.07 | 100 |
| cache_operations | 0.003415 | 0.000034 | 0.07 | 100 |
| timestep_guidance_processing | 0.001870 | 0.000374 | 0.04 | 5 |
| positional_encoding | 0.000482 | 0.000096 | 0.01 | 5 |
| final_processing | 0.000269 | 0.000054 | 0.01 | 5 |
| block_state_init | 0.000066 | 0.000001 | 0.00 | 100 |
| initialization | 0.000008 | 0.000002 | 0.00 | 5 |
| ip_adapter_processing | 0.000003 | 0.000001 | 0.00 | 5 |
