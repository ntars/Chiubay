"""常用损失：只用自研 Tensor 算子组合，不调用外部实现。"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def cross_entropy(logits: Tensor, labels) -> Tensor:
    """多分类交叉熵（含 softmax）。

    logits: [B, C]；labels: [B] 整数索引。
    数值稳定：先减每行 max（常数偏移，不参与梯度）。
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    b, c = logits.shape
    if labels.size != b:
        raise ValueError(f"labels 长度 {labels.size} 与 batch {b} 不符")
    if labels.min() < 0 or labels.max() >= c:
        raise ValueError("labels 越界")

    shift = -np.max(logits.data, axis=1, keepdims=True)
    z = logits + Tensor(shift, dtype=logits.data.dtype)
    log_z = z.exp().sum(axis=1).log()

    onehot = np.zeros((b, c), dtype=logits.data.dtype)
    onehot[np.arange(b), labels] = 1.0
    picked = (z * Tensor(onehot, dtype=logits.data.dtype)).sum(axis=1)
    return (log_z - picked).mean()


def mse(pred: Tensor, target) -> Tensor:
    """均方误差，target 可以是 ndarray / Tensor。"""
    diff = pred - target
    return (diff * diff).mean()


def log_softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """手写 log-softmax：先减每行 max 做常数平移，再减 logsumexp。

    返回 Tensor，梯度由自研反向传播计算；策略梯度需要它作为入口。
    """
    shift = -np.max(logits.data, axis=axis, keepdims=True)
    z = logits + Tensor(shift, dtype=logits.data.dtype)
    return z - z.exp().sum(axis=axis, keepdims=True).log()