"""E3e —— ③ 的第三次重设计：先把 A3 的测量方式修正，再谈 A5。

为什么必须重做（本轮最重要的发现）
================================

第四轮（`docs/10` §9.8）我写下过一个结论：

    "闭锁发生在 529–1943 步，在线 LMS 收敛需要 30k–100k 步，差 20–100 倍。"

**这个结论是错的。** 本轮实测把它拆成两条：

  1. 恒定 η=0.30 的 LMS **永远不会收敛到 A**，它停在稳态误调地板
     `‖W−A‖/‖A‖ ≈ 0.39–0.42`，**加多少步都不下降**（实测 500 → 30000 步
     始终在 0.386–0.453 之间摆动）。这是 LMS 的理论性质：
     稳态误调 ≈ η·tr(R)/2，与步数无关。
  2. 有 η 衰减时，**2000 步就到 0.111，30000 步到 0.077**。
     不存在"需要 30k–100k 步"这回事。

我把"恒定 η 的误调地板"误读成了"学习很慢"，于是编出一个
20–100 倍的时间尺度冲突。**那个冲突不存在。**

顺带作废的还有 `docs/10` §9.8 里"去掉闭锁 30k 步可达 0.056"——
本轮复现得到 0.2224，不可重现，来源不明。

真正的错误在 A3 的测量设计
==========================

旧设计（`e3d_emergence.py` 与 `docs/10` §9.8）用：

    早期窗口 (300, 4300)   → 之后还有 25,700 步继续学习
    晚期窗口 (TT-4800, TT-800) → 之后只剩  800 步

**窗口长度相同，但窗口关闭后到测量的"巩固时间"不同**，
前者多 24,900 步。所以旧比值测的是"恢复时间"，不是"不可逆性"。
早期条件有 25,700 步可以恢复，晚期只有 800 步，**早期必然看起来更可逆**。
那个"晚/早 = 0.0057"因此毫无意义。

修正：等剂量 + 等巩固时间，**只改窗口的年龄**。

    L     = 4000   剂量长度（两条件相同）
    DELTA = 6000   窗口关闭后到测量的巩固步数（两条件相同）
    早期窗口起点 T1 = 300
    晚期窗口起点 T2 = 20000
    效应 = ‖w(有养育) − w(无养育对照)‖

A3 = 效应_晚 / 效应_早 < 0.20

指标也用另一种：不计 `imprint`，直接比"同等剂量的养育留下多大痕迹"。
`imprint` 的零点问题另记（见下）。

imprint 的零点在哪（纠正一个误解）
================================

`imprint = (d0−d1)/(d0+d1)`，`d0=‖w−A_i‖`，`d1=‖w−(A+S)_i‖`。实测三点：

    w = 0      → mean −0.0245   （值域 −0.17 ~ +0.07，≈ 无偏好）
    w = A      → mean −1.0000   （完全记住旧环境）
    w = A+S    → mean +1.0000   （完全吸收养育）

**零点在"无偏好"处，不在 `w=A` 处。** 上一轮我把"闭锁细胞印记为负"
读成"A5 失败"，其实是 **A3 成立**——负印记的意思是"没被养育改动"。
A3 与 A5 本来就该在不同细胞上测，不该在同一个平均值里互相抵消。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3v3_core as V

D = V.CFG["d"]
N = V.CFG["n_cells"]

# ---- A3 的等剂量等巩固设计 ----
DOSE = 4000          # 剂量长度
DELTA = 6000         # 窗口关闭后的巩固步数
T_EARLY = 300        # 早期窗口起点
T_LATE = 20000       # 晚期窗口起点
A3_THR = 0.20        # 判据：晚期效应 < 早期的 20%

# ---- A5 的两阶段设计 ----
T_TOTAL_A5 = 30000
B_START = 6000
A5_THR = 0.15


def shifts_of(seed: int):
    """早期养育与晚期养育必须是**两个不同**的扰动。

    旧脚本用 `make_shift(seed+2)` 同时当内建 shift 和 second_shift，
    两者恰好是同一个矩阵，于是"换了养育"实际没换。
    """
    a = V.make_env(seed + 1, D)
    s_early = V.make_shift(seed + 2, D, amp=2.5, a_mat=a)
    s_late = V.make_shift(seed + 13, D, amp=2.5, a_mat=a)
    return a, s_early, s_late


def effect(seed: int, t0: int, mode: str, cfg: dict, dose: int = DOSE,
           delta: int = DELTA) -> float:
    """等剂量、等巩固：窗口 (t0, t0+dose)，测量时刻 t0+dose+delta。

    对照是同一时刻、同一 mode、**完全无养育窗口**的那一次。
    两个条件只有 t0 不同。
    """
    a, s, _ = shifts_of(seed)
    t_end = t0 + dose + delta
    ctrl = V.develop(seed, [], n_cells=N, d=D, t_total=t_end, mode=mode, cfg=cfg)
    cond = V.develop(seed, [(t0, t0 + dose)], second_shift=s, b_start=0,
                     n_cells=N, d=D, t_total=t_end, mode=mode, cfg=cfg)
    return float(np.linalg.norm(cond["w"] - ctrl["w"]))


def imprint_vec(w: np.ndarray, a: np.ndarray, s: np.ndarray, n: int) -> np.ndarray:
    d0 = np.linalg.norm(w - a[:n], axis=1)
    d1 = np.linalg.norm(w - (a + s)[:n], axis=1)
    return (d0 - d1) / np.maximum(d0 + d1, 1e-12)


def run_a3(seeds=4) -> list[dict]:
    rows = []
    configs = [
        ("full", 0.25),   # η衰减 + 闭锁 + gate（完整 ③）
        ("full", 0.40),   # 更早闭锁
        ("eta_only", 0.25),  # 只 η 衰减，不闭锁 ← 关键消融
        ("soft", 0.25),   # η衰减 + gate，不硬锁
    ]
    for mode, dl in configs:
        cfg = {"D_lock": dl}
        ratios, e_es, e_ls = [], [], []
        for sd in range(seeds):
            eE = effect(sd, T_EARLY, mode, cfg)
            eL = effect(sd, T_LATE, mode, cfg)
            r = eL / eE if eE > 1e-9 else float("nan")
            ratios.append(r)
            e_es.append(eE)
            e_ls.append(eL)
        rows.append({
            "mode": mode, "D_lock": dl,
            "eff_early": [round(x, 4) for x in e_es],
            "eff_late": [round(x, 4) for x in e_ls],
            "ratio": [None if np.isnan(x) else round(x, 4) for x in ratios],
            "ratio_median": float(np.nanmedian(ratios)),
            "A3": bool(np.nanmedian(ratios) < A3_THR),
        })
    return rows


def run_a5(seeds=4, n_cells_list=(6, 12)) -> list[dict]:
    """认同层锁死后，未闭锁的外围细胞能否学会**新的**养育。

    两阶段：0 → B_START 用 A（细胞逐个闭锁），B_START 之后换成 A+s_late。
    只在**未闭锁**的细胞上测对 s_late 的印记 —— 闭锁细胞按定义不该有印记。
    """
    rows = []
    for n in n_cells_list:
        for dl in (0.10, 0.15, 0.25):
            cfg = {"D_lock": dl}
            per_seed = []
            for sd in range(seeds):
                a, s_early, s_late = shifts_of(sd)
                r = V.develop(sd, [(300, T_TOTAL_A5 - 300)], second_shift=s_late,
                              b_start=B_START, n_cells=n, d=D,
                              t_total=T_TOTAL_A5, cfg=cfg)
                fz = r["frozen"]
                v_late = imprint_vec(r["w"], a, s_late, n)
                per_seed.append({
                    "seed": sd,
                    "n_locked": int(fz.sum()),
                    "n_free": int((~fz).sum()),
                    "imp_locked": (float(v_late[fz].mean()) if fz.any() else None),
                    "imp_free": (float(v_late[~fz].mean()) if (~fz).any() else None),
                    "t_lock": [int(x) for x in r["t_lock"]],
                })
            free_vals = [x["imp_free"] for x in per_seed if x["imp_free"] is not None]
            med = float(np.median(free_vals)) if free_vals else float("nan")
            rows.append({
                "n_cells": n, "D_lock": dl,
                "median_imprint_free": None if not free_vals else round(med, 4),
                "median_n_free": float(np.median([x["n_free"] for x in per_seed])),
                "A5": bool(free_vals and med > A5_THR),
                "per_seed": [{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in x.items()} for x in per_seed],
            })
    return rows


def convergence_curve() -> dict:
    """把"LMS 收敛慢"这个错误结论钉死：恒定 η 永不收敛，η 衰减 2000 步就够。"""
    a = V.make_env(1, D)
    out = {}
    for mode in ("none", "eta_only", "full"):
        series = []
        for T in (500, 1000, 2000, 4000, 8000, 14000, 30000):
            r = V.develop(0, [], n_cells=N, d=D, t_total=T,
                          mode=mode, cfg={"D_lock": 0.25})
            rel = float(np.linalg.norm(r["w"] - a[:N]) / np.linalg.norm(a[:N]))
            series.append({"steps": T, "rel_err": round(rel, 4)})
        out[mode] = series
    return out


def main(seeds: int = 4) -> dict:
    print("=" * 74)
    print("E3e  ③ 重设计：等剂量等巩固的 A3 + 分离的 A5")
    print("=" * 74)

    print("\n[0] 收敛曲线：推翻「LMS 需要 30k–100k 步」这个说法")
    cc = convergence_curve()
    print(f"    {'steps':>7} {'none(η定)':>11} {'eta_only':>10} {'full':>8}")
    for i, row in enumerate(cc["none"]):
        print(f"    {row['steps']:>7} {row['rel_err']:>11.4f} "
              f"{cc['eta_only'][i]['rel_err']:>10.4f} {cc['full'][i]['rel_err']:>8.4f}")
    print("    → 恒定 η 停在 0.39–0.45，加步数不下降（误调地板，非收敛慢）")
    print("    → η 衰减 2000 步已到 0.111")

    print("\n[1] A3（等剂量 4000 + 等巩固 6000，只改年龄）")
    a3 = run_a3(seeds)
    print(f"    {'mode':>9} {'D_lock':>7} {'效应早':>8} {'效应晚':>8} {'晚/早':>8} {'A3':>4}")
    for r in a3:
        print(f"    {r['mode']:>9} {r['D_lock']:>7.2f} "
              f"{np.mean(r['eff_early']):>8.4f} {np.mean(r['eff_late']):>8.4f} "
              f"{r['ratio_median']:>+8.4f} {'是' if r['A3'] else '否':>4}")

    print("\n[2] A5（A → 新养育，只在未闭锁细胞上测）")
    a5 = run_a5(seeds)
    print(f"    {'n':>3} {'D_lock':>7} {'闭锁数':>7} {'自由数':>7} {'自由印记':>9} {'A5':>4}")
    for r in a5:
        nf = r["median_n_free"]
        print(f"    {r['n_cells']:>3} {r['D_lock']:>7.2f} "
              f"{r['n_cells'] - nf:>7.1f} {nf:>7.1f} "
              f"{(r['median_imprint_free'] if r['median_imprint_free'] is not None else float('nan')):>+9.4f} "
              f"{'是' if r['A5'] else '否':>4}")

    both = []
    for r3 in a3:
        for r5 in a5:
            if r3["mode"] == "full" and r3["A3"] and r5["A5"]:
                both.append({"D_lock_A3": r3["D_lock"], "n_cells_A5": r5["n_cells"],
                             "D_lock_A5": r5["D_lock"]})
    print(f"\n[3] A3 与 A5 同时成立的配置组合：{len(both)} 个")
    for b in both:
        print(f"    A3@D_lock={b['D_lock_A3']}  A5@n={b['n_cells_A5']},D_lock={b['D_lock_A5']}")

    return {"design": "equal-dose equal-consolidation A3 + separated A5",
            "params": {"d": D, "dose": DOSE, "delta": DELTA,
                       "t_early": T_EARLY, "t_late": T_LATE,
                       "a3_thr": A3_THR, "a5_thr": A5_THR, "b_start": B_START},
            "convergence": cc, "a3": a3, "a5": a5,
            "simultaneous": both}


if __name__ == "__main__":
    res = main()
    p = Path("runs/e3e")
    p.mkdir(parents=True, exist_ok=True)
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {p / 'report.json'}")
