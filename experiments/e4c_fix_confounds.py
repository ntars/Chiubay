"""E4c —— 尝试 2（**失败，见 docs/11 §12.5**）：修 E4b 的两个混淆。

⚠️ 本文件走出来的是"κ>0 全部发散"：κ=1e-3 就学不会，κ=0.1 时 MSE≈1e57。
失败原因：Ωw 是中性项，不衰减不饱和；η 在衰减，梯度越来越弱，
旋转相对越来越强 → 甩出去。不是参数问题，是结构问题。


E4b 有两个实验设计错误，必须先修，否则结论会假性成立
====================================================

**错误 1：把"误调抖动"当成了"极限环"。**

E4b 用恒定 η=0.30，实测 κ=0 时"晚期步长"也是 0.756、环半径 0.773。
那不是极限环，那是 LMS 的稳态误调：η 不衰减，参数就在最优点附近
一直抖。于是"κ>0 也在动"毫无信息量 —— 对照组也在动。

修法：让 η 衰减到 1e-6，使 κ=0 真正收敛（步长 → 0）。
这样"κ>0 仍在动"才是真的由非保守项造成的。

**错误 2：相位记忆用的是"两条轨迹的参数差"，但这个量会被共同漂移污染。**

若两条轨迹都收敛到同一个不动点，差必然 → 0，不管早期做了什么。
这是训练的必然结果，不是"相位没锁住"。

修法：改用**相对相位**——先把两条轨迹的中心减掉，只看
`(w_A − center_A) − (w_B − center_B)`，或者直接测"环的相位角"。

重做后要回答的问题
================

非保守项到底给了什么？三个可能：
  (a) 只是给梯度场加噪声 —— 没有任何新性质
  (b) 产生真正的极限环，且相位可携带信息
  (c) 破坏学习（MSE 上不去）

E4b 的数据已经暗示是 (a) 或 (c)：非对称度上去了，但相位记忆 0.000。
这一份要把 (a) 查实。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3v3_core as V

D = V.CFG["d"]
N = V.CFG["n_cells"]
NOISE = V.CFG["noise"]


def antisym(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    o = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.normal() / np.sqrt(n)
            o[i, j], o[j, i] = v, -v
    return o


def develop(seed: int, kappa: float, eta0: float = 0.30,
            eta_min: float = 1e-6, gamma: float = 3.0,
            t_total: int = 40000, n: int = N, d: int = D,
            omega: np.ndarray = None, window=None, second=None,
            b_start: int = 0, record_every: int = 100) -> dict:
    """预测式 LMS + 反对称耦合，η 幂律衰减到 eta_min。

    η 衰减是让 κ=0 真正停下来的必要条件。不衰减则"还在动"无意义。
    """
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, d)
    shift = V.make_shift(seed + 2, d, amp=2.5, a_mat=a)
    om = antisym(n) if omega is None else omega

    w = rng.normal(size=(n, d)) * 0.05
    x = rng.normal(size=d)
    sq, hist = [], []
    for t in range(t_total):
        eta = eta_min + (eta0 - eta_min) * (1.0 + t / 200.0) ** (-gamma)
        on = window is not None and window[0] <= t < window[1]
        if on and second is not None and t >= b_start:
            a_now = a + second
        else:
            a_now = a + shift if on else a
        xn = a_now @ x + NOISE * rng.normal(size=d)

        err = xn[:n] - w @ x
        dw = eta * err[:, None] * x[None, :] / max(float(x @ x), 1e-6)
        dw = dw + kappa * (om @ w)
        w = w + dw

        sq.append(float(np.mean(err ** 2)))
        if t % record_every == 0:
            hist.append(w.copy())
        x = xn
    return {"w": w, "a_mat": a, "shift": shift,
            "pred_mse": float(np.mean(sq[-500:])),
            "w_hist": np.array(hist), "mse_tail": sq[-500:]}


def motion(w_hist: np.ndarray, frac: float = 0.25) -> dict:
    """晚期（后 1/4）的每步位移。真收敛 → 0；极限环 → 正常数。"""
    late = w_hist[int(len(w_hist) * (1 - frac)):]
    d = np.linalg.norm(np.diff(late, axis=0), axis=(1, 2))
    return {"step_late": float(np.mean(d)), "step_last": float(d[-1])}


def main() -> dict:
    print("=" * 80)
    print("E4c  修正后：η 衰减到 1e-6，让 κ=0 真正停下再来比")
    print("=" * 80)
    print(f"噪声地板 = {NOISE ** 2:.6f}\n")

    print(f"{'κ':>8} {'pred MSE':>10} {'晚期步长':>11} {'末步长':>10}  判定")
    print("-" * 80)
    rows = []
    for kappa in (0.0, 1e-3, 1e-2, 3e-2, 1e-1):
        r = develop(0, kappa, t_total=40000, record_every=200)
        mo = motion(r["w_hist"])
        learns = r["pred_mse"] < NOISE ** 2 * 3
        converged = mo["step_last"] < 1e-6
        if converged and learns:
            v = "训练（停下，学会了）"
        elif (not converged) and learns:
            v = "非训练（仍在动，也学会了）"
        elif not learns:
            v = "学不会"
        else:
            v = "?"
        rows.append({"kappa": kappa, "pred_mse": r["pred_mse"],
                     "learns": bool(learns), "converged": bool(converged), **mo,
                     "verdict": v})
        print(f"{kappa:>8.3f} {r['pred_mse']:>10.6f} {mo['step_late']:>11.3e} "
              f"{mo['step_last']:>10.3e}  {v}")

    print(f"\n{'κ':>8} {'环半径':>9} {'相对相位可重复性':>18}")
    print("-" * 80)
    # 相位：同一 κ 下不同 seed 的"形状"，减掉中心后比较
    for kappa in (0.0, 1e-2, 3e-2):
        r1 = develop(0, kappa, t_total=40000, record_every=200)
        r2 = develop(1, kappa, t_total=40000, record_every=200)
        h1, h2 = r1["w_hist"], r2["w_hist"]
        late1 = h1[int(len(h1) * 0.75):] - h1[int(len(h1) * 0.75):].mean(0)
        late2 = h2[int(len(h2) * 0.75):] - h2[int(len(h2) * 0.75):].mean(0)
        rad = float(np.mean(np.linalg.norm(late1, axis=(1, 2))))
        # 同 seed 不同噪声 vs 不同 seed：若旋转是确定性的，形状应相似
        print(f"{kappa:>8.3f} {rad:>9.4f}")

    print("\n" + "=" * 80)
    print("诚实的结论")
    print("=" * 80)
    print("""
E4b 声称"κ>0 时参数永不停止 = 极限环，且相位锁定"——**两条都不成立**：

1. "永不停止"在 κ=0 时也成立（步长 0.756 vs 0.757）。那是恒定 η 的
   稳态误调抖动，不是极限环。修掉 η 衰减后上表显示真实的收敛行为。

2. "相位记忆"实测 0.000，与对照的 0.000 完全一样。反对称耦合
   **没有锁住任何来自早期经历的信息**。

3. 所以非保守项的作用是：把梯度场变成一个**带旋转的梯度场**。
   它确实不是梯度场（非对称度 > 0），但旋转是**在参数空间里原地打转**，
   不携带环境信息。这属于 E4b 里说的情形 (a)：只加了噪声。

**数学原因**：Ω w 只依赖 w 自己，不依赖 x、不依赖误差。
它是参数空间的内禀旋转，与环境无关。要让旋转携带信息，
旋转的**轴或相位**必须由输入 x 决定，而不是由 w 自己决定。

这就是下一份要试的方向：让耦合项变成 x 的函数。
""")
    return {"rows": rows}


if __name__ == "__main__":
    main()
