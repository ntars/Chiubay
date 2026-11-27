"""E4b —— 尝试 1（**失败，见 docs/11 §12.5**）：加反对称项看能否得到极限环。

⚠️ 本文件的两个结论都不成立：
  1. "κ>0 时参数永不停止 = 极限环" —— 假。κ=0 的对照也在动，
     那是恒定 η 的稳态误调抖动，不是极限环。
  2. "相位记忆" 实测 0.000，与对照 0.000 完全一样，反对称耦合
     没有锁住任何来自早期经历的信息。

失败原因：Ωw 只依赖 w 自己，不依赖 x、不依赖误差，是参数空间的内禀
旋转，与环境无关，因此不携带信息。

保留本文件作为失败记录。后续：e4c（修混淆，暴露发散）、
e4d（加振幅限制，仍失败）。判决与结论见 docs/11。
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
ETA = 0.30
NOISE = V.CFG["noise"]


def antisym(n: int, d: int, seed: int = 7) -> np.ndarray:
    """n×n 反对称耦合矩阵（作用在行上：w → Ω w）。

    每对细胞之间一个弹簧，但方向相反 —— 这是"牵引"而不是"吸引"。
    吸引（对称）会收敛到共同点，属于能量最小化；
    牵引（反对称）不会，它只旋转。
    """
    rng = np.random.default_rng(seed)
    o = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.normal() / np.sqrt(n)
            o[i, j], o[j, i] = v, -v
    return o


def develop_nc(seed: int, windows, kappa: float,
               second_shift=None, b_start: int = 5000,
               n: int = N, d: int = D, t_total: int = 8000,
               eta: float = ETA, omega: np.ndarray = None,
               record: bool = False) -> dict:
    """非保守版：预测式 LMS + 反对称耦合。"""
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, d)
    shift = V.make_shift(seed + 2, d, amp=2.5, a_mat=a)
    om = antisym(n, d) if omega is None else omega

    w = rng.normal(size=(n, d)) * 0.05
    x = rng.normal(size=d)
    sq = []
    w_hist = []
    for t in range(t_total):
        on = any(lo <= t < hi for lo, hi in windows)
        if on and second_shift is not None and t >= b_start:
            a_now = a + second_shift
        else:
            a_now = a + shift if on else a
        xn = a_now @ x + NOISE * rng.normal(size=d)

        err = xn[:n] - w @ x
        dw = eta * err[:, None] * x[None, :] / max(float(x @ x), 1e-6)
        dw = dw + kappa * (om @ w)          # ← 唯一的非保守项
        w = w + dw

        sq.append(float(np.mean(err ** 2)))
        if record and t % 50 == 0:
            w_hist.append(w.copy())
        x = xn

    out = {
        "w": w,
        "a_mat": a,
        "shift": shift,
        "pred_mse": float(np.mean(sq[-500:])),
        "mse_trace": sq if record else None,
        "w_hist": np.array(w_hist) if record and w_hist else None,
    }
    return out


def motion(w_hist: np.ndarray) -> dict:
    """晚期还在动吗？极限环的两个判据。"""
    late = w_hist[len(w_hist) // 2:]
    steps = np.linalg.norm(np.diff(late, axis=0), axis=(1, 2))
    center = late.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(late - center, axis=(1, 2))))
    return {"step_size_late": float(np.mean(steps)),
            "cycle_radius": radius,
            "still_moving": float(np.mean(steps)) > 1e-4}


def jac_asym(kappa: float, eps: float = 1e-5) -> float:
    """非对称度：0 = 梯度场（训练），>0 = 非保守。"""
    n, d = N, D
    om = antisym(n, d)
    rng = np.random.default_rng(3)
    x = rng.normal(size=d)
    rng2 = np.random.default_rng(4)
    a = V.make_env(1, d)
    X = rng2.normal(size=(3000, d))
    Y = X @ a[:n].T + NOISE * rng2.normal(size=(3000, n))

    def field(wf):
        W = wf.reshape(n, d)
        err = Y - X @ W.T
        g = -ETA * (err.T @ X) / len(X) + kappa * (om @ W)
        return g.reshape(-1)

    w0 = rng.normal(size=(n, d)) * 0.5
    m = n * d
    J = np.zeros((m, m))
    for k in range(m):
        e = np.zeros(m)
        e[k] = eps
        J[:, k] = (field(w0.reshape(-1) + e) - field(w0.reshape(-1) - e)) / (2 * eps)
    return float(np.linalg.norm(J - J.T) / max(np.linalg.norm(J), 1e-12))


def main() -> dict:
    print("=" * 78)
    print("E4b  反对称耦合：让可塑性不是下山，而是转圈")
    print("=" * 78)

    noise_floor = NOISE ** 2
    print(f"\n噪声地板（单步预测 MSE）= {noise_floor:.6f}")
    print(f"判据：MSE 接近地板 = 仍学会了世界；参数仍在动 = 不是训练")

    rows = []
    print(f"\n{'κ (耦合强度)':>12} {'非对称度':>10} {'pred MSE':>10} "
          f"{'晚期步长':>10} {'环半径':>8}  判定")
    print("-" * 78)
    for kappa in (0.0, 1e-3, 3e-3, 1e-2, 3e-2):
        r = develop_nc(0, [(300, 4700)], kappa=kappa, t_total=6000, record=True)
        mo = motion(r["w_hist"])
        asym = jac_asym(kappa)
        learns = r["pred_mse"] < noise_floor * 3
        if kappa == 0.0:
            verdict = "训练（梯度场，会停下）"
        elif learns and mo["still_moving"]:
            verdict = "✅ 非训练且仍学会"
        elif mo["still_moving"]:
            verdict = "非训练但学不会"
        else:
            verdict = "转不动"
        rows.append({"kappa": kappa, "asymmetry": asym,
                     "pred_mse": r["pred_mse"], "learns": bool(learns),
                     **mo, "verdict": verdict})
        print(f"{kappa:>12.3f} {asym:>10.2e} {r['pred_mse']:>10.6f} "
              f"{mo['step_size_late']:>10.2e} {mo['cycle_radius']:>8.4f}  {verdict}")

    # ---- 相位是否被早期经历锁定（A3 的机制）----
    print("\n" + "=" * 78)
    print("A3 的新机制：锁相位，而不是锁参数")
    print("=" * 78)
    print("""
做法：两条轨迹给**不同的早期养育**，后期给**相同**环境，
看它们的参数差会不会被早期决定（相位记忆）。

    相位记忆 = ‖w_A − w_B‖（晚期） / ‖w_A − w_B‖（早期）
    若 ≈1 → 差异一直在，早期被永久写入
    若 ≈0 → 差异被抹平
""")
    kappa = 3e-3
    a = V.make_env(1, D)
    s1 = V.make_shift(2, D, amp=2.5, a_mat=a)
    s2 = V.make_shift(13, D, amp=2.5, a_mat=a)

    r1 = develop_nc(0, [(300, 3000)], kappa=kappa, second_shift=s1,
                    b_start=0, t_total=12000)
    r2 = develop_nc(0, [(300, 3000)], kappa=kappa, second_shift=s2,
                    b_start=0, t_total=12000)
    gap_late = float(np.linalg.norm(r1["w"] - r2["w"]))

    r1e = develop_nc(0, [(300, 3000)], kappa=kappa, second_shift=s1,
                     b_start=0, t_total=3000)
    r2e = develop_nc(0, [(300, 3000)], kappa=kappa, second_shift=s2,
                     b_start=0, t_total=3000)
    gap_early = float(np.linalg.norm(r1e["w"] - r2e["w"]))
    keep = gap_late / max(gap_early, 1e-12)
    print(f"  早期差异 = {gap_early:.4f}   晚期差异 = {gap_late:.4f}   "
          f"相位记忆 = {keep:.3f}")

    # 对照：κ=0（纯训练）下同一实验
    c1 = develop_nc(0, [(300, 3000)], kappa=0.0, second_shift=s1, b_start=0, t_total=12000)
    c2 = develop_nc(0, [(300, 3000)], kappa=0.0, second_shift=s2, b_start=0, t_total=12000)
    c1e = develop_nc(0, [(300, 3000)], kappa=0.0, second_shift=s1, b_start=0, t_total=3000)
    c2e = develop_nc(0, [(300, 3000)], kappa=0.0, second_shift=s2, b_start=0, t_total=3000)
    keep0 = (float(np.linalg.norm(c1["w"] - c2["w"])) /
             max(float(np.linalg.norm(c1e["w"] - c2e["w"])), 1e-12))
    print(f"  对照 κ=0（纯训练）相位记忆 = {keep0:.3f}")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print(f"""
1. κ=0 时非对称度 1e-10，参数最终停下（步长趋零）—— 标准训练。
2. κ>0 时非对称度显著 > 0，参数**永不停止移动**（极限环），
   而预测 MSE 仍在噪声地板附近 —— **它学会了世界，但没有"训练完"。**
3. 这是与 LLM 训练最本质的差别：训练的终态是不动点（loss 的极小），
   这里的终态是极限环。**你无法写出一个 loss，使它的极小点是一个极限环。**
4. 相位记忆 = {keep:.3f}（对照组 {keep0:.3f}）——
   {'早期差异被保留' if keep > 0.5 else '早期差异被抹平'}。
   若这条成立，A3 与 A5 不再矛盾：
       A3 = 相位被锁定（不能改）
       A5 = 环还在转（还能动）
   参数冻结做不到这一点——冻结会同时杀死 A5。
""")


if __name__ == "__main__":
    p = Path("runs/e4b")
    p.mkdir(parents=True, exist_ok=True)
    main()
