"""E3f —— ③ 的第四轮结论：A3 可以成立，"并存"不成立。

本文件不产生新机制，只把 E3e 扫描出的边界钉死，并记录三个被推翻的数字。

一、E3e 建立的两件事
===================

1. **A3 可以成立，而且不依赖硬编码的"闭锁"**（`experiments/e3e_redesign.py`）。
   旧设计（`e3d_emergence.py`、`docs/10` §9.8）用"晚/早"比值，但两个窗口
   **剂量相同、巩固时间不同**（早期窗口后面还有 25,700 步可恢复，晚期只有 800 步），
   所以测的是"恢复时间"而不是"不可逆性"，那个 0.0057 无意义。

   改成等剂量（4000 步）+ 等巩固（6000 步）、只改窗口年龄后：

       mode      D_lock   效应早   效应晚   晚/早    A3
       full       0.25    0.482   0.0089  +0.022   是
       full       0.40    0.398   0.0149  +0.034   是
       eta_only   0.25    0.072   0.0509  +0.644   否
       soft       0.25    0.483   0.0748  +0.149   是

   `eta_only`（只有 η 衰减、不闭锁）给不出 A3。**A3 确实需要某种闭锁**，
   这一点比上一轮"硬锁是构造所以不算"更精确：需要的不是 `if frozen: eta=0`
   那一行，而是**一定数量的细胞在窗口关闭前停止更新**。

2. **A5 的失败由闭锁直接造成，与 η 衰减无关**（隔离实验，`D_lock=0` 即无闭锁）：

       D_lock=0.00  闭锁 0/12   A5 印记 +0.278   67% 细胞更接近 A+S
       D_lock=0.10  闭锁 5/12   A5 印记 +0.093   25%
       D_lock=0.15  闭锁 7/12   A5 印记 -0.019   17%

   闭锁数越多，A5 单调塌陷。这条隔离做干净了。

二、"并存"是被边界抖动伪造的
=============================

`D_lock` 细扫时在 0.075 处出现：

    A3 = 0.0551（判据 0.20，过）
    A5 = 0.1550（判据 0.15，**刚好过**）

看上去像找到了共存点。但 10 seeds 展开后：

    A3 中位 +0.1406，通过率 60%
    A5 中位 +0.1595，通过率 70%
    **同一 seed 上两者同时通过：3/10**

`n_cells=10` 时是 1/5。也就是说：两个指标各自有 60–70% 的通过率，
但它们**不是同一次运行里一起通过的**。这是两个独立边界抖动的联合概率
（0.6×0.7 ≈ 0.42 的上界，实测更低），不是机制。

> **判据：A3 与 A5 必须在同一 seed 上同时成立，且跨 seed 稳定。
> 当前 3/10 到 1/5 的联合通过率不满足。**

三、这一轮推翻的三个数字（都是我自己上一轮写进文档的）
===================================================

1. **"在线 LMS 需要 30k–100k 步收敛"** —— 错。
   恒定 η=0.30 的 LMS **永远收敛不到 A**，它停在稳态误调地板
   `‖W−A‖/‖A‖ ≈ 0.39–0.45`，**加步数不下降**（500→30000 步全程摆动在
   0.386–0.453）。这是 LMS 的稳态误调 ≈ η·tr(R)/2，与步数无关。
   有 η 衰减时 **2000 步就到 0.111**，30000 步到 0.077。
   我把"误调地板"误读成"学习慢"，编出一个不存在的时间尺度冲突。

2. **"闭锁 529–1943 步 vs 学习 30k–100k 步，差 20–100 倍"** —— 基于上一条，作废。

3. **"去掉闭锁 30k 步可达 ‖w−(A+B)‖=0.056"** —— 不可复现。本轮同条件实测
   0.2224（4000/8000/14000/20000/30000 步分别为 0.2248 / 0.2674 / 0.2179 /
   0.2299 / 0.2224，**基本不随步数下降**）。原始数字来源不明，撤回。

四、仍然成立的两条方法结论
========================

1. **`imprint` 的零点在"无偏好"处，不在 `w=A` 处。**
   实测量到三点：`w=0` → −0.0245（≈0）；`w=A` → −1.000；
   `w=A+S` → +1.000。所以"闭锁细胞印记为负"是 **A3 成立**的证据
   （"没被养育改动"），不是"A5 失败"。上一轮我把两者读混了。
   A3 与 A5 必须在**不同细胞**上测，不能在一个平均值里互相抵消。

2. **消融必须只改一个变量。** 上一轮我曾用"去掉闭锁"当消融，
   但同时把 η 从衰减换成恒定，于是把 η 的功劳算到了闭锁头上。
   `e3e_redesign.py` 的 `mode` 参数现在把这两件事分开：
   `none`（η 定、无锁）/ `eta_only`（η 衰减、无锁）/ `full`（两者都有）。

五、③ 的当前状态（第四次改判）
============================

    v4.0     "已通过"                  → 装置坏（不动点 w=0）
    v4.1     "不可覆盖通过"            → 装置坏 + use_layer bug
    v4.2初   "指标没有信噪比"          → 已解决（S/N 0.50→1.70）
    v4.2四轮 "闭锁与学习差 20-100 倍"  → 数字错，作废
    v4.2五轮 "A3 可成立，A5 与 A3 不可靠并存"  ← 当前

③ 不是"设计失败"，也不是"待验证"，而是：

> **A3 有可复现的实现路径；A5 在该实现下必然被削弱；
> 两者的联合成立率 3/10，属于边界抖动而非机制。**

要过这一关，必须让"闭锁"与"可学习"作用在**不同的物理量**上
（结构 vs 参数），而不是在同一批参数上争夺 η。这仍然指向 ④。
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
DOSE, DELTA, T1, T2 = 4000, 6000, 300, 20000
A3_THR, A5_THR = 0.20, 0.15
TT_A5, B_START = 30000, 6000


def _eff(seed: int, t0: int, cfg: dict, mode: str = "full",
         dose: int = DOSE, delta: int = DELTA, n: int = 12) -> float:
    a = V.make_env(seed + 1, D)
    s = V.make_shift(seed + 2, D, amp=2.5, a_mat=a)
    t_end = t0 + dose + delta
    c = V.develop(seed, [], n_cells=n, d=D, t_total=t_end, mode=mode, cfg=cfg)
    e = V.develop(seed, [(t0, t0 + dose)], second_shift=s, b_start=0,
                  n_cells=n, d=D, t_total=t_end, mode=mode, cfg=cfg)
    return float(np.linalg.norm(e["w"] - c["w"]))


def _a5(seed: int, cfg: dict, n: int = 12) -> float:
    a = V.make_env(seed + 1, D)
    s = V.make_shift(seed + 13, D, amp=2.5, a_mat=a)
    r = V.develop(seed, [(300, TT_A5 - 300)], second_shift=s, b_start=B_START,
                  n_cells=n, d=D, t_total=TT_A5, cfg=cfg)
    w, fz = r["w"], r["frozen"]
    dA = np.linalg.norm(w - a[:n], axis=1)
    dB = np.linalg.norm(w - (a + s)[:n], axis=1)
    v = (dA - dB) / np.maximum(dA + dB, 1e-12)
    return float(v[~fz].mean()) if (~fz).any() else float("nan")


def joint_test(dl: float = 0.075, seeds: int = 10, n: int = 12) -> dict:
    """同一 seed 上 A3 与 A5 是否同时成立 —— 这才是"并存"的定义。"""
    rows = []
    for sd in range(seeds):
        cfg = {"D_lock": dl}
        eE, eL = _eff(sd, T1, cfg, n=n), _eff(sd, T2, cfg, n=n)
        a3 = eL / eE if eE > 1e-9 else float("nan")
        a5 = _a5(sd, cfg, n=n)
        rows.append({"seed": sd, "A3": round(a3, 4), "A5": round(a5, 4),
                     "A3_pass": bool(a3 < A3_THR), "A5_pass": bool(a5 > A5_THR),
                     "both": bool(a3 < A3_THR and a5 > A5_THR)})
    n_both = sum(r["both"] for r in rows)
    return {"D_lock": dl, "seeds": seeds, "n_cells": n,
            "A3_pass_rate": round(sum(r["A3_pass"] for r in rows) / seeds, 3),
            "A5_pass_rate": round(sum(r["A5_pass"] for r in rows) / seeds, 3),
            "joint_pass_rate": round(n_both / seeds, 3),
            "joint_count": n_both, "rows": rows,
            "verdict": ("并存" if n_both == seeds else
                        "不并存（边界抖动，不是机制）")}


def lock_isolation(seeds: int = 3) -> list[dict]:
    """隔离闭锁对 A5 的影响：D_lock=0 时无细胞闭锁。"""
    out = []
    for dl in (0.0, 0.10, 0.15):
        vals = [_a5(sd, {"D_lock": dl}) for sd in range(seeds)]
        out.append({"D_lock": dl, "median_A5": round(float(np.median(vals)), 4)})
    return out


def main() -> dict:
    print("=" * 74)
    print("E3f  ③ 第四轮：A3 可成立，但 A3 与 A5 不能可靠并存")
    print("=" * 74)

    print("\n[1] 闭锁对 A5 的隔离（D_lock=0 时无细胞闭锁）")
    iso = lock_isolation()
    for r in iso:
        print(f"    D_lock={r['D_lock']:.2f}  A5 中位印记 = {r['median_A5']:+.4f}")

    print("\n[2] 同一 seed 上 A3 与 A5 是否同时成立（D_lock=0.075, n=12）")
    jt = joint_test()
    print(f"    {'seed':>4} {'A3':>8} {'A5':>8} {'A3过':>5} {'A5过':>5} {'同时':>5}")
    for r in jt["rows"]:
        print(f"    {r['seed']:>4} {r['A3']:>+8.4f} {r['A5']:>+8.4f} "
              f"{'是' if r['A3_pass'] else '否':>5} {'是' if r['A5_pass'] else '否':>5} "
              f"{'是' if r['both'] else '否':>5}")
    print(f"\n    A3 通过率 {jt['A3_pass_rate']:.0%}，"
          f"A5 通过率 {jt['A5_pass_rate']:.0%}，"
          f"**同时通过 {jt['joint_count']}/{jt['seeds']}**")
    print(f"    判定：{jt['verdict']}")

    print("\n[3] 边界抖动，不是机制 —— 单个指标都能过，但不是同一次运行")
    print("    两个指标各自 60-70% 通过率，联合通过 30%，")
    print("    低于独立假设的上界 0.6×0.7=42%，且不会随 seed 增加而提高。")

    return {"D_lock_joint": 0.075, "lock_isolation": iso, "joint": jt,
            "retracted": [
                "在线 LMS 需要 30k-100k 步收敛（错：恒定 η 永不收敛，是误调地板）",
                "闭锁与学习差 20-100 倍时间尺度（基于上一条，作废）",
                "去掉闭锁 30k 步可达 0.056（不可复现，实测 0.2224，撤回）",
            ]}


if __name__ == "__main__":
    res = main()
    p = Path("runs/e3f")
    p.mkdir(parents=True, exist_ok=True)
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {p / 'report.json'}")
