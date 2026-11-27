"""E4h —— 决定性检验：非保守性是否带来"晚期抹不平"的历史依赖。

## 为什么这是唯一算数的一步

E4g 证明 B-heur 的更新场非保守（归一化非对称 1.41，对照 2e-9）。
但非保守只是必要条件，**不是创新**。理由：`∂F_c/∂c = 0`（`dc` 里没有 `c`），
所以 `c` 是个纯积分器 —— 纯积分器**必然**非保守，这是平凡的，
等于说"一个只会累加的变量不守梯度"。而且实测 `c` 单调漂移撞到裁剪界
`0.05` 后饱和，结构定死。

所以要问的是：**非保守能不能做到梯度学习做不到的事。**

梯度学习（存在 L）：
    长期行为由 L 的极小点集合决定。晚期给同一个环境，
    不管早期是什么，都会收敛到同一个（或同分布的）极小点。
    **早期差异最终被抹平。**

非保守（不存在 L）：
    没有共同的"地形"。终态依赖路径。
    **早期差异可能被永久保留。**

这就是 A3（不可逆性）。判据不是"有差别"，而是"差别在晚期抹不平"。

## 对照设计（必须，否则结论无意义）

在线学习中，梯度下降**本来就有**路径依赖（多任务会互相覆盖残留）。
所以"早期不同 → 终态不同"**不能**证明非训练。

必须比较的是 **晚期抹平率**：

    retention = ‖θ_late1 − θ_late2‖ / ‖θ_early1 − θ_early2‖

    retention → 0   晚期环境抹平了早期差（梯度学习的典型行为）
    retention ≈ 1   早期差被完整保留（非保守的独有行为）

**两组必须在同一晚期环境、同一剂量、同一测量间隔下比。**

## 三种装置

    G     纯梯度（对照，必须 retention 小）
    I     积分器 c（e4g 的 B-heur，撞界，预期 retention 小 —— 因为撞界=抹平）
    R     旋转耦合（不撞界，有不动点，非保守）—— 本文件的主角

## 旋转耦合的构造

在 (w_i 的径向, c_i) 平面里把梯度转 90°：

    g_w = −∂L/∂w_i ∈ R^d        g_c = −∂L/∂c_i ∈ R
    û   = w_i / ‖w_i‖

    F_w = η·(g_w − κ·g_c·û)      ← 混合了 c 的梯度
    F_c = μ·(g_c + κ·g_w·û)      ← 混合了 w 的梯度

κ=0 时退化成纯梯度。κ>0 时雅可比含反对称项，非保守。
**但 L=0 处 F=0，所以它有不动点** —— 不是靠撞界停下来的，
而是真的会收敛。这才是"既收敛又不守势"。

上面这对式子正是把梯度向量 (g_w·û, g_c) 转 90°·κ，
即 [[1,−κ],[κ,1]]，其反对称部分是 κ·[[0,−1],[1,0]]。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3v3_core as V

D = 8
N = 4
NOISE = 0.05
ETA = 0.30
MU = 0.02
T_EARLY = 0          # 早期窗口起点
T_EARLY_END = 4000   # 早期窗口终点
T_LATE = 60000       # 总步数


def _env(seed: int, d: int = D) -> np.ndarray:
    return V.make_env(seed + 1, d)


def _shift(seed: int, a_mat: np.ndarray, amp: float = 2.0) -> np.ndarray:
    return V.make_shift(seed, D, amp=amp, a_mat=a_mat)


def run(seed: int, mode: str, early_shift=None, kappa: float = 0.0,
        t_total: int = T_LATE, t_early_end: int = T_EARLY_END,
        mu: float = MU, clip_c: float = 20.0) -> dict:
    """mode: grad | integ | rot"""
    rng = np.random.default_rng(seed)
    a = _env(seed)
    w = rng.normal(size=(N, D)) * 0.05
    c = np.ones(N)
    x = rng.normal(size=D)

    for t in range(t_total):
        in_early = t < t_early_end
        a_now = a + early_shift if (in_early and early_shift is not None) else a
        xn = a_now @ x + NOISE * rng.normal(size=D)

        u = w @ x
        p = np.tanh(c * u)
        err = xn[:N] - p

        # ---- 梯度 ----
        dLdp = -2.0 * err / N
        dpdu = c * (1.0 - p ** 2)
        g_w = (dLdp * dpdu)[:, None] * x[None, :]          # −∂L/∂w_i
        g_c = dLdp * (1.0 - p ** 2) * u                    # −∂L/∂c_i
        nrm = np.maximum(np.linalg.norm(w, axis=1), 1e-9)
        uhat = w / nrm[:, None]

        if mode == "grad":
            Fw, Fc = g_w, g_c
        elif mode == "integ":
            # 逐细胞误差平方对期望的偏离 × |u|（e4g 的 B-heur，原文复现）
            Fw = g_w
            Fc = (err ** 2 - 0.0075) * np.abs(u) / max(mu, 1e-9) * mu
        elif mode == "rot":
            Fw = g_w - kappa * g_c[:, None] * uhat
            Fc = g_c + kappa * np.sum(g_w * uhat, axis=1)
        else:
            raise ValueError(mode)

        w = w + ETA * Fw / max(float(x @ x), 1e-9) * D
        c = np.clip(c + mu * Fc, 0.05, clip_c)
        x = xn

    u = w @ x
    p = np.tanh(c * u)
    return {"w": w, "c": c, "pred_mse": float(np.mean((x[:N] - p) ** 2))}


def relate(s1: np.ndarray, s2: np.ndarray) -> float:
    """两个终态的差异，按"早期差"归一化之外的可比量。"""
    return float(np.linalg.norm(s1 - s2))


def main() -> dict:
    print("=" * 90)
    print("E4h  决定性检验：非保守性能否让早期差异在晚期抹不平")
    print("=" * 90)
    print(f"设计：早期 [0,{T_EARLY_END}) 给 s1 或 s2，后期 [{T_EARLY_END},{T_LATE}) "
          f"给**同一个**环境 a")
    print(f"判据：retention = ‖终态差‖ / ‖早期窗口末差‖，"
          f"越接近 0 = 抹平，越接近 1 = 保留\n")

    devices = [("G 纯梯度（对照）", "grad", 0.0),
               ("I 积分器 c（e4g原样）", "integ", 0.0),
               ("R 旋转 κ=1", "rot", 1.0),
               ("R 旋转 κ=3", "rot", 3.0)]

    print(f"{'装置':<22} {'seed':>5} {'早末 ‖c差‖':>11} {'末 ‖c差‖':>11} "
          f"{'retention_c':>12} {'末 ‖w差‖':>10} {'MSE':>9}")
    print("-" * 90)

    out = []
    for name, mode, kap in devices:
        ret_c, ret_w = [], []
        for sd in range(4):
            a = _env(sd)
            s1 = _shift(2, a)
            s2 = _shift(13, a)
            # 早期窗口末的快照
            r_e1 = run(sd, mode, s1, kap, t_total=T_EARLY_END)
            r_e2 = run(sd, mode, s2, kap, t_total=T_EARLY_END)
            r_l1 = run(sd, mode, s1, kap)
            r_l2 = run(sd, mode, s2, kap)
            de = relate(r_e1["c"], r_e2["c"])
            dl = relate(r_l1["c"], r_l2["c"])
            dw = relate(r_l1["w"], r_l2["w"])
            rc = dl / max(de, 1e-12)
            ret_c.append(rc)
            ret_w.append(dw)
            print(f"{name:<22} {sd:>5} {de:>11.4f} {dl:>11.4f} "
                  f"{rc:>12.4f} {dw:>10.4f} {r_l1['pred_mse']:>9.5f}")
        out.append({"device": name, "mode": mode, "kappa": kap,
                    "retention_c_median": float(np.median(ret_c)),
                    "retention_c_all": [float(v) for v in ret_c],
                    "w_gap_median": float(np.median(ret_w))})
        print(f"{'':<22} {'中位':>5} {'':>11} {'':>11} "
              f"{np.median(ret_c):>12.4f} {np.median(ret_w):>10.4f}")
        print("-" * 90)

    print("\n" + "=" * 90)
    print("结论")
    print("=" * 90)
    g = next(o for o in out if o["mode"] == "grad")
    i = next(o for o in out if o["mode"] == "integ")
    rot = [o for o in out if o["mode"] == "rot"]

    print(f"""
1. 对照 G（纯梯度）：retention_c 中位 = {g['retention_c_median']:.4f}
   这是"晚期抹平"的基准。任何装置的 retention 只有**显著高于它**
   才说明早期差被保留。

2. I（积分器，e4g 原样）：retention_c 中位 = {i['retention_c_median']:.4f}
   {'⟹ 撞界抹平了记忆。非保守性没有带来任何保留。' if i['retention_c_median'] < 0.5 else '⟹ 有保留'}

3. 旋转耦合：
""" + "\n".join(
        f"   κ={o['kappa']:.0f}: retention_c = {o['retention_c_median']:.4f}" for o in rot
    ) + f"""

## 怎么读这张表

retention 高**不一定是好事**。三种情况都会给高 retention：
  (a) 真·非保守（没有共同地形，路径不可逆）      ← 我们要的
  (b) c 撞界/饱和（停在不同的界上）               ← 假的，I 就是这种
  (c) c 根本不收敛（漂移）                        ← 假的，只是没跑到

**必须另做一个判别**：检查终态 c 是否在界上、是否还在动。
  - c 在裁剪界上 ⟹ (b)，作废
  - c 仍在动且非周期性抖动 ⟹ (c)，作废
  - c 稳定、不在界上、且早期差被保留 ⟹ (a) 成立

本文件尚未做这一步。**在补上之前，本文件任何高 retention 都不可信。**
""")
    return {"devices": out}


if __name__ == "__main__":
    p = Path("runs/e4h")
    p.mkdir(parents=True, exist_ok=True)
    res = main()
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"写入 {p / 'report.json'}")
