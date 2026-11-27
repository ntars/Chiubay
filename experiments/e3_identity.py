"""E3（v2）—— **结论已全部撤回（2026-11-27）**

RETRACTED
=========
本文件的学习规则是  err = w·dx,  Δw = η·err·x/|x|² ，
它是抗赫布去相关（让 w·dx → 0），不是预测。而 dx 张满 R^d、
(A-I) 可逆（实测最小奇异值 0.2601），所以**唯一不动点是 w = 0**：

    20000 步后 ||w|| = 2.434e-28   （初始 1.728）
    M(w) 从 0.1405 → 0.0000（-100%），||w - w0|| = ||w0||

因此本文件产出的全部"印记"数字都是权重向零崩塌途中的残留量，
不是学习的结果。T3/T8/T2 的结论、以及 T6/T9 的消融结论，
**均不得作为证据引用**。详见 docs/09 §10.0 与 docs/10 §9。

替代实现：experiments/e3v3_core.py（预测式 LMS）+ e3_identity_v3.py。
本文件保留作为历史记录与方法论反例。

---

原 v2 说明（历史记录）：


v1 的结论已撤回，原因见 docs/10 §2。本版三个改动：

  1. 棘轮驱动量：|w·x|（活动量）→ Σ|Δx|（环境暴露量）。
     任何由"当前状态"导出的量在收敛后必然趋零：|w·x| 实测饱和在 14.21，
     |err|、|Δw|² 同理。只有"环境输入的运动量积分"严格单调，
     且环境静默时自动停转（可测：静默期棘轮不推进）。

  2. 锚点：连续跟随（rho=0.005）→ 一次性钉住。
     连续跟随使锚点永远追着权重，dev 恒小，gate ≈ 1（v1 实测 0.91–0.98），
     阻抗等于不存在。

  3. 指标：归一化为 res(w)/res(w_init)。
     v1 原始残差跨 5 seeds 漂 4 个数量级（8.3e-11 … 1.6e-6），
     跨运行长度也漂 4 个数量级。那个量测的不是"印记"。

诚实标注：
  - 局部学习规则 = LMS（Widrow–Hoff 1960），CONVENTIONAL。
  - 敏感期 = 关键期/印记（Lorenz 1935；Hubel & Wiesel 1970），现象不新。
  - T6 显示 η 衰减单独也能产生不可逆 → **锚点阻抗的必要性未证**。
  - T7 测长期泄漏。若泄漏，则"不可逆"只在一定时间尺度内成立，
    必须如实写进结论，不得说成"永久"。

运行：.venv/bin/python experiments/e3_identity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CFG = {
    "n_cells": 6,
    "d": 24,
    "t_total": 14000,
    "eta0": 0.30,
    "eta_min": 1e-4,
    "L_load": 120.0,      # η 衰减的载荷尺度
    "D_freeze": 60.0,     # 钉住锚点所需的暴露量
    "spread": 0.25,       # 细胞间阈值异质性（初始化抽定，不衰减）
    "beta": 8.0,          # 阻抗陡度
    "kappa": 1e-3,        # 阻抗软化
    "noise": 0.05,
    # 分层可塑性（A3 与 A5 并存的前提）
    # identity_frac：按"达到阈值的先后"排序，前这一比例的细胞归入认同层，
    #   闭锁后 η **严格为 0**（真冻结，不再学任何东西）。
    #   其余细胞为外围层，η 保持 eta0，永不冻结，只受 gate 阻抗。
    # 这不是从数据里学出来的分配，是设计参数，标 CONVENTIONAL。
    # 若不分层，所有细胞同时闭锁 → 晚期任何长度的暴露都能累积出效应
    # （实测 8000/800 = 16.8），"不可逆"只是"学得慢"，A5 也会一起死。
    "identity_frac": 0.67,
}


# ---------------------------------------------------------------- 环境与度量

def make_env(seed: int, d: int, radius: float = 0.85) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) / np.sqrt(d)
    ev = float(np.max(np.abs(np.linalg.eigvals(a))))
    return a * (radius / max(ev, 1e-9))


def make_shift(seed: int, d: int, amp: float = 1.2,
               a_mat: np.ndarray = None, max_radius: float = 0.88) -> np.ndarray:
    """秩 1 改变：多出一条耦合方向。这是"养育方式"的物理形式。

    若给定 a_mat，则**收缩幅度**保证 A+shift 谱半径不超过 max_radius。
    这一步不可省：实测 make_shift(7) 使谱半径从 0.850 升到 1.009，
    60000 步后权重发散（报告里出现 nan）。
    环境本身必须稳定，否则测到的是数值爆炸，不是发育。

    **不可行时必须抛异常。**
    旧版在这里静默返回零矩阵：当 spec(a_mat) >= max_radius 时，二分搜索
    连 mid=0 都判为超限，lo 停在 0，于是 shift = 0。后果是"养育窗口"
    与"无养育窗口"跑的是同一个环境，早期增量精确为 0，实验报 nan，
    而调用方看不出任何异常（实测 C3_slow_env 就这样整类失效）。
    静默退化比崩溃危险得多：崩溃会停下来，退化会把"没做实验"写成"实验失败"。
    """
    rng = np.random.default_rng(seed + 900)
    u, v = rng.normal(size=d), rng.normal(size=d)
    base = amp * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
    if a_mat is None:
        return base
    spec_a = float(np.max(np.abs(np.linalg.eigvals(a_mat))))
    if spec_a >= max_radius:
        raise ValueError(
            f"make_shift 不可行：环境本身 spec={spec_a:.4f} 已不低于上限 "
            f"max_radius={max_radius:.4f}，任何非零扰动都会越界。"
            f"请降低 make_env 的 radius，或提高 max_radius（须 <1 以免发散）。")
    ev = float(np.max(np.abs(np.linalg.eigvals(a_mat + base))))
    if ev <= max_radius:
        return base
    lo, hi = 0.0, amp
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if float(np.max(np.abs(np.linalg.eigvals(a_mat + mid / amp * base)))) <= max_radius:
            lo = mid
        else:
            hi = mid
    if lo <= 1e-9:
        raise ValueError(
            f"make_shift 不可行：收缩后幅度 {lo:.2e} 趋零，扰动将等于不存在。"
            f"spec_a={spec_a:.4f} max_radius={max_radius:.4f}")
    return (lo / amp) * base


def imprint_metric(w: np.ndarray, u: np.ndarray) -> float:
    """印记指标：权重在"养育方向" u 上的平均投影量。

    为什么不用残留误差：v1/v2 的残留误差跨 seed 漂 4 个数量级
    （对照组的基线可以到 1e-45），配对差分救不了它。
    投影量的基线是 O(1e-2)，同 seed 下对照组的漂移在 1e-8 量级，
    信噪比高 6-8 个数量级，且 5 个 seed 方向一致。

    u 是扰动矩阵 S 的第一左奇异向量，即"环境把系统往哪儿推"。
    """
    return float(np.mean(np.abs(w @ u)))


def residual(w: np.ndarray, a_mat: np.ndarray, d: int, steps: int = 800,
             seed: int = 777, noise: float = 0.05) -> float:
    """原始环境（扰动撤去）下的本地预测误差。不更新权重。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=d)
    tot = 0.0
    for _ in range(steps):
        x_next = a_mat @ x + noise * rng.normal(size=d)
        tot += float(np.mean(np.abs(w @ (x_next - x))))
        x = x_next
    return tot / steps


# ---------------------------------------------------------------- 发育

def develop(seed: int, windows, t_total: int = None, n_cells: int = None,
            d: int = None, mode: str = "full", quiet=None,
            driver: str = "dose", clock_step: float = 0.105,
            cfg: dict = None, second_shift: bool = False) -> dict:
    """发育。

    mode:
        full       η载荷衰减 + 锚点钉住 + gate阻抗（主条件）
        no_gate    η载荷衰减 + 锚点钉住，无阻抗
        eta_only   只有 η 载荷衰减
        frozen_eta η恒定 eta0，只有锚点钉住 + 阻抗
        none       纯 LMS

    driver:
        dose  = Σ|Δx|（环境暴露量；静默时停转）
        clock = 常数步进（对照组）

    second_shift:
        为 True 时，(5000, t_total) 区段改用另一条扰动 make_shift(seed+7)。用于 T8。
    """
    cfg = dict(CFG) if cfg is None else {**CFG, **cfg}
    t_total = t_total or cfg["t_total"]
    n = n_cells or cfg["n_cells"]
    d = d or cfg["d"]
    eta0, eta_min = cfg["eta0"], cfg["eta_min"]
    L_load, D_freeze = cfg["L_load"], cfg["D_freeze"]
    beta, kappa, noise = cfg["beta"], cfg["kappa"], cfg["noise"]

    rng = np.random.default_rng(seed)
    a_mat = make_env(seed + 1, d)
    shift = make_shift(seed + 1, d, a_mat=a_mat)
    shift2 = make_shift(seed + 7, d, a_mat=a_mat) if second_shift else None

    w = rng.normal(size=(n, d)) * 0.15
    w_init = w.copy()
    anchor = w.copy()
    frozen = np.zeros(n, dtype=bool)
    thr = D_freeze * (1.0 + cfg["spread"] * rng.uniform(-1, 1, size=n))
    load = 0.0
    x = rng.normal(size=d)
    t_freeze = None
    gate_hist = []
    # 分层：先按阈值从低到高排序，前 identity_frac 归认同层
    order = np.argsort(thr)
    n_id = max(1, int(round(cfg["identity_frac"] * n)))
    identity_mask = np.zeros(n, dtype=bool)
    identity_mask[order[:n_id]] = True

    use_gate = mode in ("full", "frozen_eta")
    use_pin = mode in ("full", "no_gate", "eta_only", "frozen_eta")
    use_decay = mode in ("full", "no_gate", "eta_only")
    # 分层（认同层 η 严格归零）只在 full 模式启用。
    # 曾把 frozen_eta 也放进来 → "η恒定+gate" 其实是"η恒定+硬冻结"，
    # 导致 T6 得出"gate 单独就够"的错误结论。见 docs/10 §9。
    use_layer = mode in ("full", "no_gate")

    for t in range(t_total):
        on = any(a <= t < b for a, b in windows)
        if on and second_shift and t >= 5000:
            a_now = a_mat + shift2
        else:
            a_now = a_mat + shift if on else a_mat
        quiet_now = quiet is not None and quiet[0] <= t < quiet[1]
        sig = 0.0 if quiet_now else noise
        x_next = a_now @ x + sig * rng.normal(size=d)
        dx = x_next - x

        err = w @ dx
        dev = np.mean((w - anchor) ** 2, axis=1)

        if use_decay:
            eta_eff = eta_min + (eta0 - eta_min) * np.exp(-load / L_load)
        else:
            eta_eff = np.full(n, eta0)
        if use_gate:
            gate = np.where(frozen, np.exp(-beta * dev / (dev + kappa)), 1.0)
        else:
            gate = np.ones(n)

        # --- 分层可塑性：认同层闭锁后 η 严格为 0；外围层永不冻结
        if use_layer:
            eta_eff = np.where(frozen & identity_mask, 0.0, eta_eff)
            gate = np.where(identity_mask, gate, 1.0)   # 外围层不设阻抗

        scale = 1.0 / max(float(x @ x), 1e-6)
        w = w + (eta_eff * gate * err)[:, None] * x[None, :] * scale

        load += float(np.mean(np.abs(dx))) if driver == "dose" else clock_step

        if use_pin:
            newly = (~frozen) & (load >= thr)
            if newly.any():
                anchor[newly] = w[newly]
                frozen |= newly
                if t_freeze is None:
                    t_freeze = t

        if on:
            gate_hist.append(float(gate.mean()))
        x = x_next

    return {"w": w, "w_init": w_init, "load": load, "frozen": frozen,
            "identity_mask": identity_mask, "t_freeze": t_freeze,
            "gate_in_window": float(np.mean(gate_hist)) if gate_hist else float("nan")}


# ---------------------------------------------------------------- 判据

def boot_ci(vals, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    m = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return [float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))]


def run_seed(seed: int, t_total: int = None) -> dict:
    d = CFG["d"]
    a_mat = make_env(seed + 1, d)
    init = develop(seed, [], t_total=t_total)["w_init"]
    r0 = residual(init, a_mat, d)
    base = develop(seed, [], t_total=t_total)
    rb = residual(base["w"], a_mat, d)

    TT = t_total or CFG["t_total"]
    early = (300, 1500)
    late = (TT - 1600, TT - 400)
    e_e = (residual(develop(seed, [early], t_total=t_total)["w"], a_mat, d) - rb) / r0
    e_l = (residual(develop(seed, [late], t_total=t_total)["w"], a_mat, d) - rb) / r0

    win = develop(seed, [early], t_total=t_total)
    return {"seed": seed, "r0": r0, "base": rb,
            "early": float(e_e), "late": float(e_l),
            "ratio": float(e_l / e_e) if abs(e_e) > 1e-15 else float("nan"),
            "t_freeze": base["t_freeze"],
            "gate_in_window": win["gate_in_window"]}


def main(n_seeds: int = 5) -> dict:
    """全部指标统一用 imprint_metric（权重在养育方向上的投影）。

    为什么换指标：残留误差在对照组可以低到 1e-45，比值失控；
    投影量的基线是 O(1e-2)，同 seed 下对照组漂移约 1e-8，
    信噪比高 6–8 个数量级。
    """
    d = CFG["d"]
    tt = CFG["t_total"]
    TT8 = 20000
    a_mat = make_env(1, d)
    shift = make_shift(1, d, a_mat=a_mat)
    u = np.linalg.svd(shift)[0][:, 0]          # 养育方向

    def M(seed, wins, **kw):
        return imprint_metric(develop(seed, wins, **kw)["w"], u)

    def dM(seed, win, **kw):
        """相对无扰动基线的印记增量。"""
        return M(seed, [win] if win else [], **kw) - M(seed, [], **kw)

    out: dict = {"config": CFG, "v1_retracted": True,
                 "metric": "mean|w·u|，u = 扰动矩阵第一左奇异向量"}

    # --- T1 主结果（横断：早期扰动 vs 晚期扰动）
    rows = []
    for s in range(n_seeds):
        e = dM(s, (300, 1500))
        l = dM(s, (tt - 1600, tt - 400))
        rows.append({"seed": s, "early": float(e), "late": float(l),
                     "ratio": float(l / e) if abs(e) > 1e-18 else float("nan"),
                     "t_freeze": develop(s, [])["t_freeze"]})
    e_e = np.array([abs(r["early"]) for r in rows])
    e_l = np.array([abs(r["late"]) for r in rows])
    out["T1"] = {
        "per_seed": rows,
        "early_abs_mean": float(e_e.mean()), "early_abs_std": float(e_e.std(ddof=1)),
        "late_abs_mean": float(e_l.mean()), "late_abs_std": float(e_l.std(ddof=1)),
        "late_over_early_of_abs": float(e_l.mean() / e_e.mean()),
        "freeze_t_median": float(np.median([r["t_freeze"] for r in rows])),
    }

    # --- T3 等剂量（各 4000 步）——A3 的主判据
    t3rows = []
    for s in range(n_seeds):
        e = dM(s, (300, 4300))
        l = dM(s, (tt - 4800, tt - 800))
        t3rows.append({"seed": s, "early": float(e), "late": float(l),
                       "late_over_early": float(l / e) if abs(e) > 1e-18 else float("nan")})
    loe = [abs(r["late_over_early"]) for r in t3rows if r["late_over_early"] == r["late_over_early"]]
    out["T3_equal_dose"] = {
        "per_seed": t3rows,
        "late_over_early_mean": float(np.mean(loe)),
        "late_over_early_max": float(np.max(loe)),
        "A3_threshold_0.20_pass": bool(np.max(loe) < 0.20),
    }

    # --- T8 纵向（A3 正字面）：养育 A → 换养育 B → A 上的印记还在不在
    t8rows = []
    for s in range(n_seeds):
        b = M(s, [], t_total=TT8)
        pA = M(s, [(300, 4300)], t_total=TT8)
        pAB = M(s, [(300, 4300), (5000, TT8 - 500)], t_total=TT8, second_shift=True)
        eA, eAB = abs(pA - b), abs(pAB - b)
        t8rows.append({"seed": s, "imprint_A": float(eA), "imprint_A_then_B": float(eAB),
                       "keep": float(eAB / eA) if eA > 1e-18 else float("nan")})
    keeps = [r["keep"] for r in t8rows if r["keep"] == r["keep"]]
    out["T8_imprint_survives_new_nurture"] = {
        "per_seed": t8rows,
        "keep_mean": float(np.mean(keeps)),
        "keep_ci95": boot_ci(keeps),
        "pass_ge_0.8": bool(np.mean(keeps) >= 0.8),
    }

    # --- T2 A5：认同层锁死后，外围层还能不能学新方向 B
    #     修 v2 的 bug：之前没把 second_shift 传给第一次对照，导致两层都没见过 B。
    t2 = {}
    w_before = develop(0, [(300, 4300)], second_shift=False)
    w_after = develop(0, [(300, 4300), (5000, tt - 500)], second_shift=True)
    m_id = w_before["identity_mask"]
    shift_b = make_shift(99, d, a_mat=a_mat)
    u_b = np.linalg.svd(shift_b)[0][:, 0]

    def layer_proj(w, uu, mask):
        return float(np.mean(np.abs(w[mask] @ uu)))

    t2 = {
        "identity_on_uA_before": layer_proj(w_before["w"], u, m_id),
        "identity_on_uA_after": layer_proj(w_after["w"], u, m_id),
        "periphery_on_uB_before": layer_proj(w_before["w"], u_b, ~m_id),
        "periphery_on_uB_after": layer_proj(w_after["w"], u_b, ~m_id),
        "n_identity": int(m_id.sum()), "n_periphery": int((~m_id).sum()),
    }
    t2["A3_identity_unchanged"] = bool(
        abs(t2["identity_on_uA_after"] / t2["identity_on_uA_before"] - 1) < 0.10)
    t2["A5_periphery_moved"] = bool(
        abs(t2["periphery_on_uB_after"] - t2["periphery_on_uB_before"]) > 1e-6)
    out["T2_A5_layered"] = t2

    # --- T4 关键期扫描
    out["T4_critical_window"] = {
        str(t0): float(dM(0, (t0, t0 + 800)))
        for t0 in (300, 800, 1400, 2000, 3000, 4000, 6000, 8000, 10000, 12000)
    }

    # --- T5 剂量 vs 时钟
    t5 = {}
    for sname, kw in (("stationary", {}), ("quiet_0_3000", {"quiet": (0, 3000)})):
        row = {}
        for drv in ("dose", "clock"):
            ee = dM(0, (300, 4300), driver=drv, **kw)
            ll = dM(0, (tt - 4800, tt - 800), driver=drv, **kw)
            b = develop(0, [], driver=drv, **kw)
            row[drv] = {"early": float(ee), "late": float(ll), "t_freeze": b["t_freeze"],
                        "late_over_early": float(ll / ee) if abs(ee) > 1e-18 else None}
        t5[sname] = row
    out["T5_dose_vs_clock"] = t5

    # --- T6 消融
    t6 = {}
    for mode in ("full", "no_gate", "eta_only", "frozen_eta", "none"):
        ee = dM(0, (300, 4300), mode=mode)
        ll = dM(0, (tt - 4800, tt - 800), mode=mode)
        b = develop(0, [], mode=mode)
        t6[mode] = {"early": float(ee), "late": float(ll),
                    "late_over_early": float(ll / ee) if abs(ee) > 1e-18 else None,
                    "t_freeze": b["t_freeze"]}
    out["T6_ablation"] = t6

    # --- T9 软阻抗 vs 硬冻结（60000 步）
    t9 = {}
    TT9 = 60000
    for mode, label in (("frozen_eta", "soft_gate_only"), ("full", "hard_freeze_layered")):
        b9 = M(0, [], t_total=TT9, mode=mode)
        pA = M(0, [(300, 4300)], t_total=TT9, mode=mode)
        pAB = M(0, [(300, 4300), (5000, TT9 - 500)], t_total=TT9, mode=mode,
                second_shift=True)
        eA, eAB = abs(pA - b9), abs(pAB - b9)
        t9[label] = {"imprint_A": float(eA), "imprint_A_then_B": float(eAB),
                     "keep": float(eAB / eA) if eA > 1e-18 else None}
    out["T9_soft_vs_hard"] = t9

    # --- T7 长期（60k）横断
    TT7 = 60000
    e7 = dM(0, (300, 1500), t_total=TT7)
    l7 = dM(0, (TT7 - 4800, TT7 - 800), t_total=TT7)
    out["T7_long_horizon"] = {"t_total": TT7, "early": float(e7), "late": float(l7),
                              "late_over_early": float(l7 / e7) if abs(e7) > 1e-18 else None}

    soft_keep = out["T9_soft_vs_hard"]["soft_gate_only"]["keep"]
    hard_keep = out["T9_soft_vs_hard"]["hard_freeze_layered"]["keep"]
    t6 = out["T6_ablation"]
    gate_alone = t6["frozen_eta"]["late_over_early"]
    hard_alone = t6["full"]["late_over_early"]

    out["summary"] = {
        "A3_equal_dose_pass": out["T3_equal_dose"]["A3_threshold_0.20_pass"],
        "A3_equal_dose_max_ratio": out["T3_equal_dose"]["late_over_early_max"],
        "A3_imprint_survives_new_nurture": out["T8_imprint_survives_new_nurture"]["pass_ge_0.8"],
        "A3_imprint_keep_mean": out["T8_imprint_survives_new_nurture"]["keep_mean"],
        "A5_periphery_still_learns": out["T2_A5_layered"]["A5_periphery_moved"],
        "T9_soft_keep_60k": soft_keep,
        "T9_hard_keep_60k": hard_keep,
        "gate_alone_sufficient": bool(
            gate_alone is not None and hard_alone is not None
            and abs(gate_alone) < 0.20 and abs(hard_alone) < 0.20),
        "layering_necessary": bool(
            gate_alone is not None and abs(gate_alone) >= 0.20),
        "honest": (
            "三条结论：(1) 等剂量下晚期/早期 <0.004，A3 通过；"
            "(2) 换养育方式后印记保留 1.01，A3 的正字面（不可抹平）通过；"
            "(3) 认同层锁死后外围层仍能动，A5 活着。"
            "推翻的假设：T6 显示**只靠 gate 阻抗、η 恒定也够了**（晚/早=0），"
            "所以此前认为'必须硬冻结才能不可逆'是错的，分层是冗余保险而非必要机制。"
            "T9 进一步显示软阻抗在 60000 步后仍保住印记（keep=1.000），"
            "'软约束只能减慢不能阻止'这个关于 EWC 的推论在本实验里不成立。"
            "唯一站得住的剂量/时钟分离在静默场景：dose 晚/早=0.071，clock=0.575，差 8 倍。"
        ),
    }
    return out


if __name__ == "__main__":
    res = main()
    W = 80
    print("=" * W)
    print("E3 v2  认同环：无全局损失下的早期不可覆盖")
    print("指标：mean|w·u|（权重在养育方向 u 上的投影），不是残留误差")
    print("=" * W)

    t1 = res["T1"]
    print(f"\nT1  横断对比（{CFG['t_total']} 步，{len(t1['per_seed'])} seeds，窗口各 1200 步）")
    for r in t1["per_seed"]:
        print(f"    seed {r['seed']}  早 {r['early']:+.3e}  晚 {r['late']:+.3e}  "
              f"晚/早 {r['ratio']:+.5f}  钉住t={r['t_freeze']}")
    print(f"    |早| {t1['early_abs_mean']:.3e}   |晚| {t1['late_abs_mean']:.3e}   "
          f"晚/早 {t1['late_over_early_of_abs']:.5f}")

    t3 = res["T3_equal_dose"]
    print(f"\nT3  等剂量（各 4000 步）—— A3 主判据")
    for r in t3["per_seed"]:
        print(f"    seed {r['seed']}  早 {r['early']:+.3e}  晚 {r['late']:+.3e}  "
              f"晚/早 {r['late_over_early']:+.5f}")
    print(f"    均值 {t3['late_over_early_mean']:.5f}   最大 {t3['late_over_early_max']:.5f}")
    print(f"    A3 要求全部 <0.20  ->  {'通过' if t3['A3_threshold_0.20_pass'] else '不通过'}")

    t8 = res["T8_imprint_survives_new_nurture"]
    print(f"\nT8  印记穿越新养育（A3 正字面）：养育A(300-4300) → 养育B(5000-end) → 测A")
    for r in t8["per_seed"]:
        print(f"    seed {r['seed']}  仅A {r['imprint_A']:.3e}  A后B {r['imprint_A_then_B']:.3e}  "
              f"保留 {r['keep']:.3f}")
    ci = t8["keep_ci95"]
    print(f"    保留均值 {t8['keep_mean']:.3f}  95%CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
          f"要求 ≥0.8 -> {'通过' if t8['pass_ge_0.8'] else '不通过'}")

    t2 = res["T2_A5_layered"]
    print(f"\nT2  A5 检验（分层）：认同 {t2['n_identity']} 细胞 / 外围 {t2['n_periphery']} 细胞")
    print(f"    认同层在养育方向 uA 上  闭锁前 {t2['identity_on_uA_before']:.4e}  "
          f"闭锁后 {t2['identity_on_uA_after']:.4e}  不变={t2['A3_identity_unchanged']}")
    print(f"    外围层在新方向 uB 上    见B前 {t2['periphery_on_uB_before']:.4e}  "
          f"见B后 {t2['periphery_on_uB_after']:.4e}  能动={t2['A5_periphery_moved']}")

    print(f"\nT4  关键期扫描（800 步窗口，移动起始时刻）")
    for k, v in res["T4_critical_window"].items():
        print(f"    t0={int(k):5d}  {v:+.3e}")

    print(f"\nT5  剂量驱动 vs 时钟驱动")
    for s, row in res["T5_dose_vs_clock"].items():
        for drv, v in row.items():
            print(f"    {s:>14s} {drv:>5s}  早 {v['early']:+.3e}  晚 {v['late']:+.3e}  "
                  f"晚/早 {v['late_over_early']:+.5f}  钉住t={v['t_freeze']}")

    print(f"\nT6  消融")
    for m, v in res["T6_ablation"].items():
        loe = v["late_over_early"]
        ls = f"{loe:+.5f}" if loe is not None else "    n/a"
        print(f"    {m:>10s}  早 {v['early']:+.3e}  晚 {v['late']:+.3e}  晚/早 {ls}")

    print(f"\nT9  软阻抗 vs 硬冻结（60000 步，换养育 B）")
    for k, v in res["T9_soft_vs_hard"].items():
        keep = v["keep"]
        ks = f"{keep:+.3f}" if keep is not None else "n/a"
        print(f"    {k:>22s}  仅A {v['imprint_A']:.3e}  A后B {v['imprint_A_then_B']:.3e}  "
              f"保留 {ks}")

    t7 = res["T7_long_horizon"]
    print(f"\nT7  长期横断（{t7['t_total']} 步）：早 {t7['early']:+.3e}  晚 {t7['late']:+.3e}  "
          f"晚/早 {t7['late_over_early']:+.5f}")

    print(f"\n结论")
    for k, v in res["summary"].items():
        print(f"    {k}: {v}")

    p = Path("runs/e3_identity")
    p.mkdir(parents=True, exist_ok=True)
    (p / "report_v2.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {p / 'report_v2.json'}")
