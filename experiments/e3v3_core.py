"""E3 v3 核心：预测式 LMS + 局部棘轮 + 可塑性分层。

与 v1/v2 的根本差别（不是修补，是换装置）：
  v1/v2  err = w·dx,  Δw = η·err·x/|x|²   → 让 w·dx→0，不动点是 w=0
         实测 20000 步后 ||w|| = 2.4e-28（初始 1.728）。测的是崩塌残余。
  v3     err = x_next[i] - w_i·x,  Δw_i = η_i·err_i·x/|x|²
         → 不动点 w_i = A[i]（环境动力学的第 i 行）。测的是"学没学会世界"。

指标换成有界指数：
    imprint_i = (||w_i-A[i]|| - ||w_i-(A+S)[i]||) / (||w_i-A[i]|| + ||w_i-(A+S)[i]||)
  +1 = 完全吸收了养育；0 = 无偏好；-1 = 反向。
  有界、无量纲、跨条件可比。投影量 M=mean|w·u| 做不到这三点。
"""
from __future__ import annotations

import numpy as np


def make_env(seed: int, d: int = 24, radius: float = 0.85) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) / np.sqrt(d)
    ev = float(np.max(np.abs(np.linalg.eigvals(a))))
    return a * (radius / max(ev, 1e-9))


def make_shift(seed: int, d: int = 24, amp: float = 1.2,
               a_mat: np.ndarray = None, max_radius: float = 0.88) -> np.ndarray:
    """秩 1 养育。不可行时抛异常，绝不静默返回零（v2 的教训）。"""
    rng = np.random.default_rng(seed + 900)
    u, v = rng.normal(size=d), rng.normal(size=d)
    base = amp * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
    if a_mat is None:
        return base
    spec_a = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
    if spec_a >= max_radius:
        raise ValueError(f"不可行：spec(a)={spec_a:.4f} >= {max_radius:.4f}")
    if float(np.max(np.abs(np.linalg.eigvals(a_mat + base)))) <= max_radius:
        return base
    lo, hi = 0.0, amp
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(np.max(np.abs(np.linalg.eigvals(a_mat + mid / amp * base)))) <= max_radius:
            lo = mid
        else:
            hi = mid
    if lo <= 1e-9:
        raise ValueError("不可行：收缩幅度趋零，扰动等于不存在")
    return (lo / amp) * base


CFG = {
    # d=12 而非 24：秩 1 扰动在 d=24 上把信号摊薄到 ||S_i||≈0.13，
    # 而 LMS 稳态误调是 0.24–0.26，S/N=0.50（测不出养育）。d=12 时
    # 信号 0.29、误调 0.17，S/N=1.70，才有测量能力。见 docs/10 §9.7。
    "d": 12,
    "n_cells": 6,          # 细胞 i 预测 x_next 的第 i 个坐标
    "t_total": 14000,
    "eta0": 0.30,
    "eta_min": 2e-3,
    # 收敛比驱动 η：η = eta_min + (eta0-eta_min)·conv^gamma。
    # gamma=2 是标定结果：gamma=1 时 S/N=1.18，gamma=2 时 1.70。
    # 这不是"调参数让指标好看"，而是让 S/N 首次超过 1 —— 在此之前
    # 任何"印记"数字都是误调，不是信号。
    "gamma": 2.0,
    "D_lock": 0.35,        # 闭锁所需的收敛比：err < 0.35 * err_ref
    "ema": 0.003,          # 逐细胞相对误差的滑动平均系数
    "ref_steps": 300,      # 参考误差取前多少步的均值（单步太噪）
    "lock_delay": 400,     # 出生后至少这么多步才允许闭锁（避开初始化瞬态）
    "spread": 0.25,        # 细胞间阈值异质性
    "beta": 8.0,           # 阻抗陡度
    "kappa": 1e-3,         # 阻抗软化
    "noise": 0.05,
    "identity_frac": 0.67,  # 先闭锁的这批归认同层，闭锁后 eta 严格 0
}


def develop(seed: int, windows, second_shift: np.ndarray = None,
            b_start: int = 5000, mode: str = "full",
            cfg: dict = None, n_cells: int = None, d: int = None,
            t_total: int = None, noise: float = None) -> dict:
    """mode:
        full      预测LMS + 局部η衰减 + 锚点钉住 + gate + 认同层硬锁
        no_gate   同上，去掉 gate
        eta_only  只有局部η衰减
        soft      预测LMS + 锚点钉住 + gate，无η衰减、无硬锁
        none      纯预测LMS
    """
    c = {**CFG, **(cfg or {})}
    n = n_cells or c["n_cells"]
    d = d or c["d"]
    TT = t_total or c["t_total"]
    noise = c["noise"] if noise is None else noise
    eta0, eta_min = c["eta0"], c["eta_min"]

    rng = np.random.default_rng(seed)
    a_mat = make_env(seed + 1, d)
    shift = make_shift(seed + 2, d, a_mat=a_mat)

    w = rng.normal(size=(n, d)) * 0.05
    anchor = w.copy()
    frozen = np.zeros(n, dtype=bool)
    # 闭锁阈值：在基础阈值上给每个细胞一点异质性。
    # 注意 identity_mask 不在这里定 —— 见循环末尾的说明。
    thr = c["D_lock"] * (1.0 + c["spread"] * rng.uniform(-1, 1, size=n))
    identity_mask = np.zeros(n, dtype=bool)
    n_id_target = max(1, int(round(c["identity_frac"] * n)))
    n_id_assigned = [0]

    use_gate = mode in ("full", "soft", "no_gate")
    use_decay = mode in ("full", "no_gate", "eta_only")
    use_pin = mode != "none"
    use_layer = mode in ("full", "no_gate")

    load_i = np.zeros(n)
    err_ema = np.full(n, np.nan)
    err_ref = np.full(n, 1.0)
    ref_buf = []
    t_lock = np.full(n, -1)
    x = rng.normal(size=d)
    sq_hist, gate_hist = [], []
    eta_last = np.full(n, np.nan)
    gate_prelock = []

    for t in range(TT):
        on = any(a <= t < b for a, b in windows)
        if on and second_shift is not None and t >= b_start:
            a_now = a_mat + second_shift
        else:
            a_now = a_mat + shift if on else a_mat
        xn = a_now @ x + noise * rng.normal(size=d)

        err = xn[:n] - w @ x                       # (n,) 预测误差
        dev = np.mean((w - anchor) ** 2, axis=1)   # 相对锚点的偏离

        # 逐细胞滑动平均相对误差（无量纲，可跨环境比）
        if np.isnan(err_ema[0]):
            err_ema = np.abs(err).copy()
        else:
            err_ema = (1.0 - c["ema"]) * err_ema + c["ema"] * np.abs(err)

        # 参考误差：前 ref_steps 步的均值，而不是单步。
        # 用单步会让 conv 在第 400 步就掉到 0.35 以下（实测 5/6 细胞一起闭锁，
        # t_lock 全部等于 lock_delay），因为初始权重的第一条样本误差波动极大。
        ref_buf.append(err_ema.copy())
        if len(ref_buf) <= c["ref_steps"]:
            err_ref = np.mean(ref_buf, axis=0)
        ref_buf = ref_buf[-(c["ref_steps"] + 1):]

        # 收敛比：越低说明这一维学得越透。用它驱动 η 衰减。
        conv = np.clip(err_ema / np.maximum(err_ref, 1e-12), 0.0, 1.0)
        load_i = -np.log(np.maximum(conv, 1e-9))   # 单调递减 ⇔ load 单调递增

        # η 正比于收敛比（gamma=1）。不要写成 exp(-load) —— 那在 conv 小时
        # 会掉到 0 并被 eta_min 卡住，实测收敛退化到 ||W-A||/||A||=41.8%。
        # 直接乘 conv：学得越透，学习率越小，但永不强制归零。
        eta_eff = (eta_min + (eta0 - eta_min) * conv ** c["gamma"]) \
            if use_decay else np.full(n, eta0)
        gate = np.where(frozen, np.exp(-c["beta"] * dev / (dev + c["kappa"])), 1.0) \
            if use_gate else np.ones(n)

        if use_gate and (~frozen).any():
            gate_prelock.append(float(gate[~frozen].mean()))

        if use_layer:
            eta_eff = np.where(frozen & identity_mask, 0.0, eta_eff)
            gate = np.where(identity_mask, gate, 1.0)

        dw = (eta_eff * gate * err)[:, None] * x[None, :] / max(float(x @ x), 1e-6)
        w = w + dw

        if use_pin:
            newly = (~frozen) & (t >= c["lock_delay"]) & (conv <= thr)
            if newly.any():
                anchor[newly] = w[newly]
                frozen |= newly
                # 认同层 = 先闭锁的那一批，**在闭锁当下在线确定**。
                # 旧版按 thr 排序预先指定，但 thr 是随机阈值、与实际收敛无关：
                # 实测 identity_mask=[F,T,F,T,T,T] 而 frozen=[T,T,T,F,T,T]，
                # 两者几乎不重叠 —— "认同层"于是既不是"先学会的"也不是
                # "被锁死的"，只是个随机子集，`eta_late[m]==0` 的承诺直接失效。
                # 现在改为按闭锁顺序填满 identity_frac，语义与 A3 一致，
                # 且 mask 在闭锁那一刻就生效（不需要等跑完）。
                for i in np.flatnonzero(newly):
                    if n_id_assigned[0] < n_id_target:
                        identity_mask[i] = True
                        n_id_assigned[0] += 1
        t_lock[frozen & (t_lock < 0)] = t

        if on:
            gate_hist.append(float(gate.mean()))
        sq_hist.append(float(np.mean((err) ** 2)))
        eta_last = eta_eff * gate
        x = xn

    return {"w": w, "frozen": frozen, "identity_mask": identity_mask,
            "t_lock": t_lock, "load_i": load_i, "thr": thr, "conv": conv,
            "a_mat": a_mat, "shift": shift,
            "pred_mse": float(np.mean(sq_hist[-500:])),
            "eta_late": eta_last,
            "gate_in_window": float(np.mean(gate_hist)) if gate_hist else float("nan"),
            "gate_early": float(np.mean(gate_prelock)) if gate_prelock else float("nan")}


def imprint(w_i: np.ndarray, a_row: np.ndarray, s_row: np.ndarray) -> float:
    """有界印记指数。+1 完全吸收养育，0 无偏好，-1 反向。"""
    d0 = float(np.linalg.norm(w_i - a_row))
    d1 = float(np.linalg.norm(w_i - (a_row + s_row)))
    if d0 + d1 < 1e-12:
        return 0.0
    return (d0 - d1) / (d0 + d1)
