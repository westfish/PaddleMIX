# TeaBlockCache + Taylor 优化方法对比

本目录包含了两种使用TeaBlockCache + Taylor优化的方法：

## 1. 原始方法 (text_to_image_generation_teablockcache_taylor_flux.py)

这是原始的实现方式，直接替换FluxTransformer2DModel的forward方法：

```python
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from forwards.teablockcache_taylor_flux_forward import TeaBlockCacheTaylorForward

# 直接替换forward方法
FluxTransformer2DModel.forward = TeaBlockCacheTaylorForward

# 手动配置所有参数
pipeline.transformer.cnt = 0
pipeline.transformer.num_steps = num_inference_steps
pipeline.transformer.step_start = 50
pipeline.transformer.step_end = 950
# ... 更多手动配置
```

### 特点：
- ✅ 直接有效
- ❌ 需要手动配置大量参数
- ❌ 难以管理和维护
- ❌ 无法轻易撤销优化
- ❌ 不符合现代软件设计模式

## 2. Hook方法 (text_to_image_generation_teablockcache_taylor_flux_hook.py)

这是新的hook-based实现方式，参考PyramidAttentionBroadcast的设计模式：

```python
from ppdiffusers import TeaBlockCacheTaylorConfig, apply_teablockcache_taylor

# 使用配置类定义所有参数
config = TeaBlockCacheTaylorConfig(
    step_start=50,
    step_end=950,
    block_cache_start=1,
    single_block_cache_start=1,
    block_rel_l1_thresh=2.0,
    single_block_rel_l1_thresh=2.0,
    taylor_max_order=1,
    taylor_first_enhance=1,
    rel_l1_thresh=2.0,
    num_inference_steps=num_inference_steps,
    current_timestep_callback=lambda: pipeline._current_timestep,
)

# 一行代码应用优化
apply_teablockcache_taylor(pipeline.transformer, config)
```

### 特点：
- ✅ 清晰的API设计，参考PyramidAttentionBroadcast
- ✅ 所有参数集中在配置类中，易于管理
- ✅ 支持通过enable_cache/disable_cache方法管理
- ✅ 可以轻易移除优化
- ✅ 符合ppdiffusers的设计模式
- ✅ 更好的错误处理和日志记录

## 3. 统一的Cache管理接口

Hook方法还支持通过模型的cache接口进行管理：

```python
# 方法1：直接使用apply函数
apply_teablockcache_taylor(pipeline.transformer, config)

# 方法2：通过模型的cache接口（推荐）
pipeline.transformer.enable_cache(config)

# 禁用缓存
pipeline.transformer.disable_cache()
```

## 4. 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `step_start` | 开始应用缓存的时间步 | 50 |
| `step_end` | 结束应用缓存的时间步 | 950 |
| `block_cache_start` | 开始缓存的transformer block索引 | 1 |
| `single_block_cache_start` | 开始缓存的single transformer block索引 | 1 |
| `block_rel_l1_thresh` | Transformer blocks的相对L1阈值 | 2.0 |
| `single_block_rel_l1_thresh` | Single transformer blocks的相对L1阈值 | 2.0 |
| `taylor_max_order` | Taylor展开的最大阶数 | 1 |
| `taylor_first_enhance` | Taylor缓存的first enhance参数 | 1 |
| `rel_l1_thresh` | Taylor缓存系统的相对L1阈值 | 2.0 |
| `num_inference_steps` | 推理步数 | 50 |
| `current_timestep_callback` | 获取当前时间步的回调函数 | 必需 |

## 5. 使用建议

建议使用**Hook方法**，因为：
1. 更容易集成到现有代码中
2. 参数管理更清晰
3. 支持动态启用/禁用
4. 遵循ppdiffusers的设计规范
5. 更好的可维护性

原始方法仍然保留用于参考和向后兼容。 