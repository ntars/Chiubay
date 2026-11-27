"""E4d —— 尝试 3（**失败，见 docs/11 §12.5**）：有界旋转。

⚠️ 结论：发散解决了，但 6 档 κ 中只有 1 档（0.002）能同时"学会 + 仍在动"，
κ≥0.005 全部学不会（MSE 0.008–0.048 vs 地板 0.0025）。
相位差的 seed 3 出现 nan。可用窗口太窄、窗内信号太弱。
失败原因：旋转与梯度共享同一批参数，强则毁学习，弱则信号淹没。

以下为原设计说明（保留作为记录）。

E4c 的失败是数学上的必然，不是参数没调好
======================================

E4c 里 `Δw += κ·(Ωw)`，Ω 反对称 → 这个项**保范数**，但它同时是
**中性**的：不衰减、不饱和。而 η 在衰减（→1e-6），梯度项越来越弱，
旋转项却恒定 → 旋转项相对越来越强 → 参数被甩出可控范围。

实测：κ=1e-3 就学不会（MSE 0.0083），κ=0.1 时 MSE ≈ 1e57。

这不是"耦合太强"，而是**没有恢复力**。任何中性项在衰减梯度旁边都会
最终主导。要得到稳定极限环，必须有振幅限制。

正确的形式
==========

    Δw = η(t)·err·x/|x|²  +  κ·(Ω w)·(1 − |w|²/R²)

第二个因子让旋转在 |w| < R 时向外推、|w| > R 时向内拉 —— 这就是
Hopf 型极限环的最小实现。它保相位、限振幅。

这次的检验和 E4c 不同，是三个分开的量：
  1. **稳定**：MSE 不爆（< 3×噪声地板），|w| 收敛到 R 附近
  2. **还在动**：末期每步位移 → 正常数（不是 0，也不是 ∞）
  3. **相位有信息**：不同早期养育 → 不同相位，且后期环境相同时相位不合并

第 3 条是 A3 的候选机制：**锁相位而不是锁参数**。
参数冻结会同时杀死 A5；相位锁定不杀。

用 n=2 让相位可解释
==================

Ω = [[0,1],[−1,0]]，两行 w_1, w_2 互相旋转，复化成 z_k = w_1k + i·w_2k。
相位 φ = arg(mean_k z_k)。这样"相位"是一个能画出来的角度，
不是一个抽象的矩阵性质。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3v3_core as V

D = V.CFG["d"]
NOISE = V.CFG["noise"]
OM2 = np.array([[0.0, 1.0], [-1.0, 0.0]])   # 2×2 旋转生成元


def develop(seed: int, kappa: float, radius: float = 1.0,
            eta0: float = 0.30, eta_min: float = 1e-6, gamma: float = 3.0,
            t_total: int = 60000, d: int = D,
            window=None, second=None, b_start: int = 0,
            record_every: int = 100) -> dict:
    """n=2：两行互相旋转，�which 相位可读。"""
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, d)
    shift = V.make_shift(seed + 2, d, amp=2.5, a_mat=a)

    w = rng.normal(size=(2, d)) * 0.05
    x = rng.normal(size=d)
    sq, hist, mse_trace = [], [], []
    for t in range(t_total):
        eta = eta_min + (eta0 - eta_min) * (1.0 + t / 200.0) ** (-gamma)
        on = window is not None and window[0] <= t < window[1]
        if on and second is not None and t >= b_start:
            a_now = a + second
        else:
            a_now = a + shift if on else a
        xn = a_now @ x + NOISE * rng.normal(size=d)

        err = xn[:2] - w @ x
        dw = eta * err[:, None] * x[None, :] / max(float(x @ x), 1e-6)

        # 有界旋转：Ωw 保范数，再乘 (1 − |w|²/R²) 得到振幅限制
        norm2 = float(np.sum(w ** 2))
        limit = 1.0 - norm2 / (radius ** 2 * 2 * d)
        dw = dw + kappa * (OM2 @ w) * limit

        w = w + dw
        sq.append(float(np.mean(err ** 2)))
        if t % record_every == 0:
            hist.append(w.copy())
        x = xn
    return {"w": w, "hist": np.array(hist),
            "pred_mse": float(np.mean(sq[-500:])),
            "norm": float(np.linalg.norm(w))}


def phase_of(w: np.ndarray) -> float:
    """两行复化成 z_k = w_1k + i·w_2k，取主相位。"""
    z = w[0] + 1j * w[1]
    return float(np.angle(np.mean(z)))


def step_size(hist: np.ndarray, frac: float = 0.2) -> float:
    late = hist[int(len(hist) * (1 - frac)):]
    return float(np.mean(np.linalg.norm(np.diff(late, axis=0), axis=(1, 2))))


def main() -> dict:
    print("=" * 82)
    print("E4d  有界旋转：能不能同时做到『仍学会』『仍在动』『相位带信息』")
    print("=" * 82)
    print(f"噪声地板 = {NOISE ** 2:.6f}   |w| 目标 = {np.sqrt(2 * D):.3f}\n")

    print(f"{'κ':>7} {'pred MSE':>10} {'|w|':>8} {'晚期步长':>11}  判定")
    print("-" * 82)
    rows = []
    for kappa in (0.0, 0.002, 0.005, 0.01, 0.02, 0.04):
        r = develop(0, kappa, t_total=60000)
        st = step_size(r["hist"])
        learns = r["pred_mse"] < NOISE ** 2 * 3
        moving = st > 1e-5
        if kappa == 0:
            v = "训练（停下）" if not moving else "训练（仍在抖）"
        elif learns and moving:
            v = "✅ 非训练 + 学会 + 仍在动"
        elif learns:
            v = "非训练但停住了"
        else:
            v = "学不会"
        rows.append({"kappa": kappa, "pred_mse": r["pred_mse"],
                     "norm": r["norm"], "step": st,
                     "learns": bool(learns), "moving": bool(moving),
                     "verdict": v})
        print(f"{kappa:>7.3f} {r['pred_mse']:>10.6f} {r['norm']:>8.4f} "
              f"{st:>11.3e}  {v}")

    # ---- 相位是否携带早期养育信息 ----
    print("\n" + "=" * 82)
    print("相位记忆：早期给不同养育，后期给相同环境，看相位差是否保留")
    print("=" * 82)
    best = [r for r in rows if r["learns"] and r["moving"]]
    if not best:
        print("  没有同时满足『学会 + 仍在动』的 κ，相位检验无意义。")
        return {"rows": rows, "phase": None}

    kappa = max(r["kappa"] for r in best)
    print(f"  取 κ = {kappa}（唯一仍有极限环的较强耦合）\n")
    a = V.make_env(1, D)
    s1 = V.make_shift(2, D, amp=2.5, a_mat=a)
    s2 = V.make_shift(13, D, amp=2.5, a_mat=a)

    print(f"  {'seed':>5} {'相位A':>9} {'相位B':>9} {'|Δ相位|':>9} {'MSE_A':>8}")
    print("  " + "-" * 48)
    phases = []
    for sd in range(5):
        ra = develop(sd, kappa, t_total=60000, window=(300, 3000),
                     second=s1, b_start=0)
        rb = develop(sd, kappa, t_total=60000, window=(300, 3000),
                     second=s2, b_start=0)
        pa, pb = phase_of(ra["w"]), phase_of(rb["w"])
        dphi = abs((pa - pb + np.pi) % (2 * np.pi) - np.pi)
        phases.append({"seed": sd, "pA": pa, "pB": pb, "dphi": dphi,
                       "mse": ra["pred_mse"]})
        print(f"  {sd:>5} {pa:>+9.4f} {pb:>+9.4f} {dphi:>9.4f} {ra['pred_mse']:>8.5f}")

    dphis = [p["dphi"] for p in phases]
    print(f"\n  相位差中位 = {np.median(dphis):.4f} rad")
    print(f"  若相位差 ≈ 0 → 早期养育没写进相位（A3 失败）")
    print(f"  若相位差 > 0.3 → 早期养育锁定了相位（A3 的候选机制）")

    print("\n" + "=" * 82)
    print("结论")
    print("=" * 82)
    ok = [r for r in rows if r["learns"] and r["moving"]]
    print(f"""
1. 有界旋转解决了 E4c 的发散：{len(ok)}/{len(rows)} 组 κ 同时满足
   『pred MSE < 3×噪声地板』与『晚期仍在移动』。
2. 这意味着存在一个**稳定的极限环**：它学会了世界，但没有"训练完"。
   这一点与 LLM 训练有本质差别 —— 训练的终态是不动点。
3. 相位差中位 = {np.median(dphis):.4f} rad。这是关键：
   {'早期养育确实改变了相位' if np.median(dphis) > 0.3 else '早期养育没有改变相位，A3 仍未成立'}。
""")
    return {"rows": rows, "phase": phases,
            "phase_median": float(np.median(dphis))}


if __name__ == "__main__":
    main()
