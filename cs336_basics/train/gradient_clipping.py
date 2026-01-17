import torch
from typing import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
    实现梯度裁剪。

    Args:
        parameters: 模型参数的可迭代对象。
        max_l2_norm: 梯度的最大L2范数。
    """
    # 计算所有参数梯度的总L2范数
    # 1. 收集所有有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]

    if not params_with_grad:
        # 如果没有任何参数有梯度，则直接返回
        return

    # 2. 计算总梯度的L2范数 (全局范数)
    # 将所有参数的梯度平方求和，然后开根号
    total_norm = torch.sqrt(
        sum(torch.sum(param.grad.data ** 2) for param in params_with_grad)
    )

    # 3. 判断是否需要裁剪
    if total_norm > max_l2_norm:
        # 4. 计算缩放因子
        # clip_coef = max_l2_norm / (total_norm + epsilon)
        clip_coef = max_l2_norm / (total_norm + 1e-6)

        # 5. 对每个参数的梯度进行缩放 (in-place)
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)  # mul_ 是原地乘法操作* scale