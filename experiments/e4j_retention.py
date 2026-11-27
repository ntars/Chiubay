"""E4j  — 抹平检验：晚期相同环境能否抹平早期 b 的差异。

## 核心问题

E4i 发现 b 通过 effect 检验（不是规范自由度），且 B-heur 变体是非保守的。
但**非保守不等于创新**。唯一能证明创新的问题是：

- **梯度学习下**：后期相同环境 → 早期差异被抹平（收敛到同一个最优 w）。
- **非保守动力学下**：后期相同环境 → 早期差异**保留**（极限环/循环）。

## 装置（沿用 E4i）

pred = scale · tanh((w·x)/scale + b)
- w: N×D 参数矩阵（学）
- b: N 维偏置（结构变量，偏置而非斜率）
- scale: 固定输出尺度（离线标定）

更新规则：
- "grad"：b 是梯度步（对照）
- "heur"：b 用启发式（e4g 式误差偏离 × |预激活|）

## 流程

1. 早期给 s1（shift 1），训练 t_early 步
2. 后期给 s2（shift 2），训练 t_late 步
3. 测 b 的差异是否保留（抹平 vs 保留）
4. 对照：两个环境相同 → 梯度必须抹平，非保守可能不抹平

## 关键：b 的饱和问题

b=+1 时 MSE=6.64（tanh 饱和）。b 的取值范围必须限制在不饱和区。
tanh 的线性区是 |z|<1，预激活 = (w·x)/s + b。
当 w≈A、x 稳态 std≈0.07 时，(w·x)/s 的量级 < 0.1，
所以 b 的合理范围是 [-0.5, 0.5]。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3v3_core as V
import e4i_escape_gauge as E

D = 8
N = 4
NOISE = 0.05
T_EARLY = 20000
T_LATE = 20000
BCLIP = (-0.5, 0.5)


def dataset(seed: int = 0):
    a = V.make_env(seed + 1, D)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200000, D))
    Y = X @ a[:N].T + NOISE * rng.normal(size=(200000, N))
    scale = 4.0 * Y.std(axis=0)
    return X, Y, scale, a


def make_shifts(seed: int = 0):
    """造两个不同的 shift，用于早期/后期。"""
    a = V.make_env(seed + 1, D)
    s1 = V.make_shift(2, D, amp=2.5, a_mat=a)
    s2 = V.make_shift(13, D, amp=2.5, a_mat=a)
    return s1, s2


def run(seed: int, variant: str, t_early: int = T_EARLY,
        t_late: int = T_LATE, mu: float = 0.05) -> dict:
    """
    训练流程：
    早期给 s1，后期给 s2。
    返回：w, b, MSE 历史, b 的历史
    """
    rng = np.random.default_rng(seed)
    X, Y, scale, a = dataset(seed)

    # 初始化
    w = rng.normal(size=(N, D)) * 0.05
    b = np.zeros(N)
    x = rng.normal(size=D)

    mse_hist = []
    b_hist = []

    for t in range(t_early + t_late):
        # 环境：前 t_early 步用 s1，后用 s2
        a_now = a + make_shifts(0)[0] if t < t_early else a + make_shifts(0)[1]
        xn = a_now @ x + NOISE * rng.normal(size=D)

        # 归一化输入，防止状态发散（参考 E4h 的 x@x 归一化）
        xn_norm = xn / max(float(np.linalg.norm(xn)), 1e-9) * np.sqrt(D)
        x_norm = x / max(float(np.linalg.norm(x)), 1e-9) * np.sqrt(D)

        # 预测（x 是 (D,)，w 是 (N,D)，scale 是 (N,)，b 是 (N,)）
        u = (x_norm @ w.T) / scale  # (N,)
        p = scale * np.tanh(u + b)  # (N,)

        # 误差
        err = xn[:N] - p  # (N,)

        # 梯度
        sech2 = 1.0 - (p / scale) ** 2  # (N,)
        dLdp = -2.0 * err / N  # (N,)
        g_w = (dLdp * sech2).reshape(N, 1) * x_norm[None, :]  # (N, D)
        g_b = dLdp * sech2 * scale  # (N,)

        if variant == "grad":
            dw, db = -g_w, -g_b
        elif variant == "heur":
            pre = (x_norm @ w.T) / scale + b  # (N,)
            dw, db = -g_w, mu * ((err ** 2 - NOISE ** 2) * np.abs(pre))  # (N,)
        else:
            raise ValueError(variant)

        # 归一化学习率
        w = w + dw * 0.001 / max(float(x_norm @ x_norm), 1e-9) * D
        b = np.clip(b + db * 0.001, BCLIP[0], BCLIP[1])

        mse_hist.append(float(np.mean(err ** 2)))
        b_hist.append(b.copy())
        x = xn_norm

    return {
        "w": w,
        "b": b,
        "mse_hist": mse_hist,
        "b_hist": b_hist,
        "final_mse": float(np.mean(mse_hist[-1000:])),
    }


def run_diff(seed: int, variant: str, t_early: int = T_EARLY,
             t_late: int = T_LATE, mu: float = 0.05) -> dict:
    """
    与 run 相同，但早期用 s1，后期用 s2（不同环境）。
    """
    rng = np.random.default_rng(seed)
    X, Y, scale, a = dataset(seed)
    s1, s2 = make_shifts(0)

    w = rng.normal(size=(N, D)) * 0.05
    b = np.zeros(N)
    x = rng.normal(size=D)

    mse_hist = []
    b_hist = []

    for t in range(t_early + t_late):
        a_now = a + s1 if t < t_early else a + s2
        xn = a_now @ x + NOISE * rng.normal(size=D)

        xn_norm = xn / max(float(np.linalg.norm(xn)), 1e-9) * np.sqrt(D)
        x_norm = x / max(float(np.linalg.norm(x)), 1e-9) * np.sqrt(D)

        u = (x_norm @ w.T) / scale
        p = scale * np.tanh(u + b)

        err = xn[:N] - p

        sech2 = 1.0 - (p / scale) ** 2
        dLdp = -2.0 * err / N
        g_w = (dLdp * sech2).reshape(N, 1) * x_norm[None, :]
        g_b = dLdp * sech2 * scale

        if variant == "grad":
            dw, db = -g_w, -g_b
        elif variant == "heur":
            pre = (x_norm @ w.T) / scale + b
            dw, db = -g_w, mu * ((err ** 2 - NOISE ** 2) * np.abs(pre))
        else:
            raise ValueError(variant)

        w = w + dw * 0.001 / max(float(x_norm @ x_norm), 1e-9) * D
        b = np.clip(b + db * 0.001, BCLIP[0], BCLIP[1])

        mse_hist.append(float(np.mean(err ** 2)))
        b_hist.append(b.copy())
        x = xn_norm

    return {
        "w": w,
        "b": b,
        "mse_hist": mse_hist,
        "b_hist": b_hist,
        "final_mse": float(np.mean(mse_hist[-1000:])),
    }


def retention_test() -> dict:
    """
    对照 1：早期 s1，后期 s1（相同环境）。
    - grad 对照：b 的差异应该被抹平
    - heur：如果抹不平，就是非保守的特征

    对照 2：早期 s1，后期 s2（不同环境）。
    - 两个都应该保留 b 的差异（环境变了，没有理由抹平）
    - 但如果 grad 在对照2中也不抹平，说明装置有漂移问题

    关键问题：相同环境下，grad 必须抹平，heur 是否抹平？
    跨环境时，两者是否都保留？
    """
    print("=" * 88)
    print("E4j  抹平检验：b 的差异在后期环境中能否被抹平")
    print("=" * 88)

    results = {}

    # ------------------------------------------------------------------
    # 对照 1：相同环境（s1 → s1）
    # ------------------------------------------------------------------
    print("\n[1] 相同环境：早期 s1 → 后期 s1")
    print("-" * 88)

    for variant in ("grad", "heur"):
        r1 = run(0, variant)
        r2 = run(1, variant)

        db = np.linalg.norm(r1["b"] - r2["b"])
        dw = np.linalg.norm(r1["w"] - r2["w"])
        mse = (r1["final_mse"] + r2["final_mse"]) / 2

        print(f"  {variant}:")
        print(f"    seed 0 MSE = {r1['final_mse']:.6f}")
        print(f"    seed 1 MSE = {r2['final_mse']:.6f}")
        print(f"    ‖b1-b2‖ = {db:.6f}")
        print(f"    ‖w1-w2‖ = {dw:.6f}")

        if db < 0.01:
            verdict = "✅ 抹平"
        elif db < 0.05:
            verdict = "⚠️ 部分抹平"
        else:
            verdict = "❌ 未抹平"

        print(f"    判定：{verdict}")
        results[variant] = {"same_env": {"db": float(db), "dw": float(dw),
                                         "mse": float(mse), "verdict": verdict}}

    # ------------------------------------------------------------------
    # 对照 2：不同环境（s1 → s2）
    # ------------------------------------------------------------------
    print("\n[2] 不同环境：早期 s1 → 后期 s2")
    print("-" * 88)

    for variant in ("grad", "heur"):
        # 需要修改 run() 支持不同 early/late shift
        r1 = run_diff(0, variant)  # seed 0, early s1
        r2 = run_diff(1, variant)  # seed 1, early s2

        db = np.linalg.norm(r1["b"] - r2["b"])
        dw = np.linalg.norm(r1["w"] - r2["w"])
        mse = (r1["final_mse"] + r2["final_mse"]) / 2

        print(f"  {variant}:")
        print(f"    seed 0 MSE = {r1['final_mse']:.6f}")
        print(f"    seed 1 MSE = {r2['final_mse']:.6f}")
        print(f"    ‖b1-b2‖ = {db:.6f}")
        print(f"    ‖w1-w2‖ = {dw:.6f}")

        if db < 0.01:
            verdict = "✅ 抹平（异常！）"
        elif db < 0.05:
            verdict = "⚠️ 部分抹平"
        else:
            verdict = "❌ 保留"

        print(f"    判定：{verdict}")
        results[variant]["diff_env"] = {"db": float(db), "dw": float(dw),
                                        "mse": float(mse), "verdict": verdict}

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("结论")
    print("=" * 88)
    g_same = results["grad"]["same_env"]
    h_same = results["heur"]["same_env"]
    g_diff = results["grad"]["diff_env"]
    h_diff = results["heur"]["diff_env"]

    print(f"""
1. 相同环境（s1→s1）：
   grad：{g_same['verdict']}  ‖b1-b2‖ = {g_same['db']:.6f}
   heur：{h_same['verdict']}  ‖b1-b2‖ = {h_same['db']:.6f}

2. 不同环境（s1→s2）：
   grad：{g_diff['verdict']}  ‖b1-b2‖ = {g_diff['db']:.6f}
   heur：{h_diff['verdict']}  ‖b1-b2‖ = {h_diff['db']:.6f}

3. 关键判据：
   (a) grad 在相同环境必须抹平，在跨环境必须保留 ⟹ 装置正确
   (b) heur 在相同环境保留 + grad 抹平 ⟹ 非保守产生训练做不到的事
   (c) 两者在相同环境都抹平 ⟹ heur 无实质区别
   (d) grad 在跨环境也不抹平 ⟹ 装置有漂移问题，需要修
""")

    return results


if __name__ == "__main__":
    p = Path("runs/e4j")
    p.mkdir(parents=True, exist_ok=True)
    res = retention_test()
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {p / 'report.json'}")
