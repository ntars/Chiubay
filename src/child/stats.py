"""统计口径：纯 NumPy，无 scipy。

M2.1 修正：原 `experiments/m2_s0.py` 里的 `iqm()` 取的是**最低 25% 的均值**，
这是"最差四分位"，不是 IQM。IQM（interquartile mean）的定义是**去掉最高与
最低各 25% 后的均值**。两者符号可能完全相反，导致上轮记录的 -0.021 被误读。

同时提供 bootstrap 置信区间，避免用 ±std 冒充区间。
"""
from __future__ import annotations

import numpy as np


def iqm(values) -> float:
    """四分位均值：去掉最低 25% 与最高 25% 后求均值。

    样本数 n 较小时按 n//4 截断，两侧各去 k 个；若去掉后为空则退回均值。
    """
    v = np.sort(np.asarray(values, dtype=float).ravel())
    n = v.size
    if n == 0:
        raise ValueError("iqm 需要非空样本")
    k = n // 4
    core = v[k:n - k] if n - 2 * k >= 1 else v
    return float(core.mean())


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int | None = 0) -> tuple[float, float]:
    """IQM 的 bootstrap 置信区间（百分位法）。"""
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 2:
        return float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    stats = np.array([iqm(v[i]) for i in idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def describe(values) -> dict:
    """均值 / 标准差 / IQM / IQM 置信区间，一次给全。"""
    v = np.asarray(values, dtype=float).ravel()
    lo, hi = bootstrap_ci(v)
    return {
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std(ddof=0)), 4),
        "median": round(float(np.median(v)), 4),
        "min": round(float(v.min()), 4),
        "max": round(float(v.max()), 4),
        "iqm": round(iqm(v), 4),
        "iqm_ci95": [round(lo, 4), round(hi, 4)],
        "n": int(v.size),
    }
