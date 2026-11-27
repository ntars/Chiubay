"""发育与分化指标：全部用纯 NumPy 计算，不依赖任何统计库。

指标口径说明（M1 修订）：
Top-K 路由下 `argmax` 不稳定——当 k≥2 时两个专家的概率可能接近，
同一类样本的 top1 会在两者间抖动，于是“众数占比”会把健康的分化误判为未分化。
因此 G0 三指标改用**软路由分布**定义，对 k=1 与 k≥2 都成立：

- 归一化熵：H(w)/log(k)，0 = 只激活 1 个，1 = k 个均等。
- 软互信息：I(专家; 状态)，用类条件平均路由分布计算，单位 bit。
- 路由一致性：类内每个样本的路由分布与类均值分布的余弦相似度均值。
  k=1 时它仍以“类内平均路由方向”为基准，与“众数占比”同向但更平滑
  （one-hot 样本的得分是 p̄_i/‖p̄‖，而非 0/1）。
"""
from __future__ import annotations

import numpy as np


def normalized_entropy(probs, axis: int = -1, eps: float = 1e-12,
                       top_k: int | None = None) -> np.ndarray:
    """归一化激活熵。

    默认除以 log(N)；给定 top_k 则除以 log(k)，此时 0 = 单专家，
    1 = 被激活的 k 个专家权重均等。G0 口径用后者，因为它不随 N 漂移。
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    ent = -(p * np.log(p)).sum(axis=axis)
    n = p.shape[axis]
    if top_k is not None:
        n = min(int(top_k), n)
    if n <= 1:
        return np.zeros_like(ent)
    return ent / np.log(n)


def _normalize_rows(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    s = p.sum(axis=1, keepdims=True)
    return p / np.clip(s, 1e-12, None)


def mutual_information(assign, labels) -> float:
    """离散互信息（bit）：MI = Σ p(i,j) log2(p(i,j)/(p(i)p(j)))。

    用于对已硬分配的专家索引计算，单元测试的基准实现。
    真实分化评估请用 soft_mutual_information。
    """
    a = np.asarray(assign, dtype=np.int64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if a.size == 0 or a.size != y.size:
        raise ValueError("assign 与 labels 长度必须一致且非空")

    _, ai = np.unique(a, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((ai.max() + 1, yi.max() + 1), dtype=np.float64)
    np.add.at(joint, (ai, yi), 1.0)
    joint /= a.size

    p_a = joint.sum(axis=1, keepdims=True)
    p_y = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    ratio = joint[nz] / (p_a @ p_y)[nz]
    return float(np.sum(joint[nz] * np.log2(ratio)))


def soft_mutual_information(probs, labels) -> float:
    """软互信息（bit）：I(专家; 状态)，用类条件平均路由分布计算。

    I = Σ_y p(y) Σ_i p(i|y) log2( p(i|y) / p(i) )
    """
    p = _normalize_rows(probs)
    y = np.asarray(labels, dtype=np.int64).ravel()
    if p.shape[0] != y.size:
        raise ValueError("probs 与 labels 长度必须一致")

    p_i = p.mean(axis=0)
    mi = 0.0
    for label in np.unique(y):
        mask = y == label
        p_y = mask.mean()
        p_cond = p[mask].mean(axis=0)
        nz = p_cond > 0
        ratio = p_cond[nz] / np.clip(p_i[nz], 1e-12, None)
        mi += p_y * float(np.sum(p_cond[nz] * np.log2(ratio)))
    return float(mi)


def routing_consistency(probs, labels) -> float:
    """类内路由一致性：0-1。

    对每个状态簇，取该类平均路由分布 p̄，计算每个样本分布与 p̄ 的余弦相似度，
    再按样本数加权平均。k=1 时等价于“类内众数占比”。
    """
    p = _normalize_rows(probs)
    y = np.asarray(labels, dtype=np.int64).ravel()
    if p.shape[0] != y.size:
        raise ValueError("probs 与 labels 长度必须一致")

    total, count = 0.0, 0
    for label in np.unique(y):
        q = p[y == label]
        p_bar = q.mean(axis=0)
        nb = np.linalg.norm(p_bar)
        if nb < 1e-12:
            continue
        denom = np.clip(np.linalg.norm(q, axis=1) * nb, 1e-12, None)
        total += float(((q @ p_bar) / denom).sum())
        count += q.shape[0]
    return float(total / max(count, 1))


def routing_gain(probs, labels) -> float:
    """类内一致性相对全局基线的增益。

    routing_consistency 在“路由完全均匀”时会虚高到 1（所有样本都与类均值同向），
    这并不代表分化。这里减去每个样本与全局平均路由的余弦相似度作为基线：
    均匀路由 → 0；真正按状态分化 → 正值。
    """
    p = _normalize_rows(probs)
    g = p.mean(axis=0)
    ng = float(np.linalg.norm(g))
    if ng < 1e-12:
        return 0.0
    denom = np.clip(np.linalg.norm(p, axis=1) * ng, 1e-12, None)
    baseline = float(((p @ g) / denom).mean())
    return float(routing_consistency(p, labels) - baseline)


def summarize_route(probs, labels, top_k: int | None = None) -> dict:
    """一次算出 G0 三指标。probs: [B, N] 门控概率，labels: [B] 状态簇。"""
    p = _normalize_rows(probs)
    ent_k = normalized_entropy(p, top_k=top_k)
    return {
        "entropy_mean": float(ent_k.mean()),
        "entropy_std": float(ent_k.std()),
        "entropy_over_n": float(normalized_entropy(p).mean()),
        "mutual_info_bits": soft_mutual_information(p, labels),
        "routing_consistency": routing_consistency(p, labels),
        "routing_gain": routing_gain(p, labels),
        "expert_usage": p.mean(axis=0),
        "dead_experts": int((p.mean(axis=0) < 1e-3).sum()),
    }