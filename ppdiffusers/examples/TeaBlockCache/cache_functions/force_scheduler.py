import paddle

def force_scheduler(cache_dic, current):
    # 1. fresh_ratio 为 0 时不强制刷新
    if cache_dic['fresh_ratio'] == 0:
        linear_step_weight = paddle.to_tensor(0.0)
    else:
        # 若后续有多种策略，可在此处区分
        linear_step_weight = paddle.to_tensor(0.0)

    # 2. 计算 step_factor（注意保持张量类型）
    step_factor = 1 - linear_step_weight + 2 * linear_step_weight * current['step'] / current['num_steps']

    # 3. 根据 step_factor 动态得到 threshold
    threshold = paddle.round(cache_dic['fresh_threshold'] / step_factor)

    # 4. 将结果写回字典
    cache_dic['cal_threshold'] = threshold
    # return threshold  # 如需返回，可解开注释