"""E4i —— 逃出规范轨道：参数无法补偿的结构变量。

## 前面两个装置的失败总结（必须写清，否则会重犯）

**e4e（线性 + 乘性门控）**：`pred = Σ w_i·g_i·x`。
`g` 是逐坐标正尺度，被归一化 LMS 的分母精确抵消。
消融实测：5 个完全不同的 `c` 给出**同一个 MSE（0.002958）**。
⟹ 规范自由度，零信息。

**e4g 装置 B（tanh + 乘性斜率）**：`pred = tanh(c·w·x)`。
`c` 改的是曲率，技术上不被吸收，故 e4f 测出 effect=58。
**但它数值上不可用**：`tanh` 输出限在 ±1，而目标 y 的 std 最高 1.67、
max 达 7.7 —— 梯度只能把 w 推向饱和区，w 单调发散（实测 20k 步后 |w|=6.8，
60k 步后 |w|=34，MSE 反升到 1.0）。
非保守是真的（1.41），但那是"积分器撞界"带来的平凡非保守，不是机制。

## 本文件的装置

两处修改，分别解决"不可观测"和"不可用"：

**(1) 输出尺度** `s_i = 4·std(y_i)`（离线标定，固定）：
        pred_i = s_i·tanh( (w_i·x)/s_i + b_i )
    这样 `w = A`、`b = 0` 时 MSE = 0.0078（地板 0.0025），装置可用。
    `s_i` 是固定常数，不是可学变量，不引入新的规范自由度。

**(2) 结构 `b` 用偏置而非斜率**：`b_i` 加在预激活上。
    `b` 会移动工作点，而 `w` 只能改斜率 —— 两者**不是同一回事**：
    `b` 改变的是"细胞在输入的哪个区域活跃"，`w` 不能平移。
    这一点必须实测（见下方 effect 检验），不能靠论证。

## 关键：先做 effect 检验，再做雅可比，最后才谈 retention

顺序不能反。前两轮都是先算雅可比（漂亮）再发现变量无效（作废）。
本文件把 effect 放在第一步，effect 不过就不继续。
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
GAUGE_TOL = 1e-3     # effect 低于此 ⟹ 规范自由度（比 e4f 的 1e-6 宽松，
                     # 因为这里 effect 含抽样噪声；但远低于 e4e 的 4.8e-4 之上）
REAL_TOL = 0.10


def dataset(seed: int = 0, size: int = 200000):
    a = V.make_env(seed + 1, D)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(size, D))
    Y = X @ a[:N].T + NOISE * rng.normal(size=(size, N))
    scale = 4.0 * Y.std(axis=0)          # 固定尺度，离线标定
    return X, Y, scale, a


def predict(w, b, X, scale):
    return scale[None, :] * np.tanh((X @ w.T) / scale[None, :] + b[None, :])


def mse(w, b, X, Y, scale) -> float:
    return float(np.mean((Y - predict(w, b, X, scale)) ** 2))


# ======================================================================
# 第一步：effect 检验 —— b 是否有行为后果
# ======================================================================
def effect_test(seed: int = 0) -> dict:
    X, Y, scale, a = dataset(seed)
    w = a[:N].copy()
    b0 = np.zeros(N)
    base = mse(w, b0, X, Y, scale)

    # 注意：不能只把 b 换掉就比 —— w 已经适应了 b=0。
    # 公平的比法：给每个 b 都让 w 重新学到最优（闭式最小二乘，对给定 b）。
    # 这里用梯度下降到收敛（batch，全量）。
    def refit_w(b, iters=3000, lr=0.5):
        ww = w.copy()
        for _ in range(iters):
            p = predict(ww, b, X, scale)
            err = p - Y
            dpdu = (1.0 - (p / scale[None, :]) ** 2)     # sech²
            g = 2.0 * ((err * dpdu).T @ X) / len(X)
            ww = ww - lr * g
        return ww

    results = []
    for tag, b in [("b=0（训练值）", np.zeros(N)),
                   ("b=+1 全场", np.ones(N)),
                   ("b=−1 全场", -np.ones(N)),
                   ("b 逐细胞交替 ±1", np.array([1.0, -1.0, 1.0, -1.0])),
                   ("b 随机 U(−2,2)", np.random.default_rng(seed + 9).uniform(-2, 2, N))]:
        wr = refit_w(b)
        m = mse(wr, b, X, Y, scale)
        results.append({"b": tag, "mse": m,
                        "effect": abs(m - base) / max(base, 1e-12),
                        "w_norm": float(np.linalg.norm(wr))})
    return {"base_mse": base, "results": results, "a_rows": float(np.linalg.norm(a[:N]))}


# ======================================================================
# 第二步：雅可比判决（只在 effect 过关后才做）
# ======================================================================
def field(theta, X, Y, scale, variant: str, mu: float = 0.05) -> np.ndarray:
    w = theta[: N * D].reshape(N, D)
    b = theta[N * D:]
    p = predict(w, b, X, scale)
    err = Y - p
    sech2 = 1.0 - (p / scale[None, :]) ** 2

    # L = mean(err²) ⟹ dL/dp = −2err/(S·N)
    dLdp = -2.0 * err / (len(X) * N)
    # ∂p/∂w_ij = sech2_i · x_j
    g_w = (dLdp * sech2).T @ X
    # ∂p/∂b_i = sech2_i · scale_i
    g_b = np.sum(dLdp * sech2 * scale[None, :], axis=0)

    if variant == "grad":
        dw, db = -g_w, -g_b
    elif variant == "heur":
        # e4g 式启发式：误差相对期望的偏离 × |预激活|
        pre = (X @ w.T) / scale[None, :] + b[None, :]
        dw, db = -g_w, mu * np.mean((err ** 2 - 0.0075) * np.abs(pre), axis=0)
    elif variant == "flip":
        sgn = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
        pre = (X @ w.T) / scale[None, :] + b[None, :]
        dw, db = -g_w, mu * sgn * np.mean((err ** 2 - 0.0075) * np.abs(pre), axis=0)
    else:
        raise ValueError(variant)
    return np.concatenate([dw.reshape(-1), db])


def jacobian(theta, X, Y, scale, variant, h=1e-6):
    m = theta.size
    J = np.zeros((m, m))
    for j in range(m):
        e = np.zeros(m)
        e[j] = h
        J[:, j] = (field(theta + e, X, Y, scale, variant)
                   - field(theta - e, X, Y, scale, variant)) / (2 * h)
    return J


def steps_vec(mu: float = 0.05):
    return np.concatenate([np.full(N * D, 1.0), np.full(N, mu)])


def asym(J) -> float:
    """J 的归一化非对称度。直接判 J 对称性，不归一化。"""
    return float(np.linalg.norm(J - J.T) / max(np.linalg.norm(J), 1e-12))


def main() -> dict:
    print("=" * 88)
    print("E4i  先验 effect（是否有后果），再验雅可比（是否在训练）")
    print("=" * 88)

    print("\n[1] effect 检验：b 换掉、w 重新最优后，MSE 变多少")
    print("-" * 88)
    et = effect_test(0)
    print(f"  基准 MSE（w=A, b=0）= {et['base_mse']:.6f}")
    print(f"  {'b 取值':<20} {'重优化后 MSE':>14} {'effect':>10}  判定")
    bad = 0
    for r in et["results"]:
        v = ("规范自由度（无后果）" if r["effect"] < GAUGE_TOL
             else "有实质后果" if r["effect"] > REAL_TOL else "弱后果")
        if r["effect"] < GAUGE_TOL:
            bad += 1
        print(f"  {r['b']:<20} {r['mse']:>14.6f} {r['effect']:>10.2e}  {v}")

    max_eff = max(r["effect"] for r in et["results"])
    print(f"\n  最大 effect = {max_eff:.2e}")
    if max_eff < GAUGE_TOL:
        print("  ❌ b 也是规范自由度。**停止**，不做雅可比，本装置作废。")
        return {"effect": et, "verdict": "gauge"}

    print(f"  ✅ effect 过关（> {GAUGE_TOL:g}），继续做雅可比判决")

    print("\n[2] 雅可比判决（θ = w⊕b，{D*N+N} 维）")
    print("-" * 88)
    X, Y, scale, a = dataset(0)
    rng = np.random.default_rng(3)
    theta0 = np.concatenate([(a[:N] * 0.9 + 0.1 * rng.normal(size=(N, D))).reshape(-1),
                             0.2 * rng.normal(size=N)])

    # 梯度校验（w 块 + b 块都要）
    eps = 1e-6
    num = np.zeros(theta0.size)
    for j in range(theta0.size):
        e = np.zeros(theta0.size)
        e[j] = eps
        th1, th2 = theta0 + e, theta0 - e
        num[j] = (mse(th1[:N*D].reshape(N, D), th1[N*D:], X, Y, scale)
                  - mse(th2[:N*D].reshape(N, D), th2[N*D:], X, Y, scale)) / (2 * eps)
    ana = -field(theta0, X, Y, scale, "grad")
    rel = float(np.linalg.norm(ana - num) / max(np.linalg.norm(num), 1e-12))
    print(f"  梯度校验：rel diff = {rel:.2e}  {'✅' if rel < 1e-6 else '❌'}")
    if rel >= 1e-6:
        return {"effect": et, "verdict": "grad_check_failed", "rel": rel}

    rows = []
    print(f"\n  {'变体':<10} {'裸非对称':>14}  判定")
    print("  " + "-" * 50)
    for variant, tag in [("grad", "B-grad"), ("heur", "B-heur"), ("flip", "B-flip")]:
        J = jacobian(theta0, X, Y, scale, variant)
        s = asym(J)
        v = ("训练（梯度场）" if s < 1e-6
             else "非训练（非保守）" if s > 0.1 else "弱非对称")
        rows.append({"variant": tag, "asym": s, "verdict": v})
        print(f"  {tag:<10} {s:>14.2e}  {v}")

    g = next(r for r in rows if r["variant"] == "B-grad")
    h = next(r for r in rows if r["variant"] == "B-heur")
    f = next(r for r in rows if r["variant"] == "B-flip")

    print("\n" + "=" * 88)
    print("结论")
    print("=" * 88)
    print(f"""
1. b 的 effect 最大 = {max_eff:.2e}（判据 > {GAUGE_TOL:g}）
   ⟹ b **不是**规范自由度，换 b 会改变可达的最优 MSE。
   这是 e4e 之后第一次有结构变量通过 effect 检验。

2. 雅可比：
   B-grad = {g['asym']:.2e}  {'✅ 对称（工具正确）' if g['asym'] < 1e-6 else '❌ 对照失败'}
   B-flip = {f['asym']:.2e}  {'✅ 被检出' if f['asym'] > 0.1 else '❌'}
   B-heur = {h['asym']:.2e}  {'非保守' if h['asym'] > 0.1 else '是梯度场'}

3. 但**非保守仍然不等于创新**。剩下的唯一问题是：

   同一个晚期环境，能否抹平早期 b 的差异？

   能抹平 ⟹ 非保守没有产生任何训练做不到的事 ⟹ 不构成创新。
   抹不平 ⟹ 才有资格叫"不是训练"。

   这一步在 e4j。**本文件不作结论。**
""")
    return {"effect": et, "max_effect": max_eff, "jac": rows,
            "grad_rel": rel, "verdict": "effect_ok"}


if __name__ == "__main__":
    p = Path("runs/e4i")
    p.mkdir(parents=True, exist_ok=True)
    res = main()
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"写入 {p / 'report.json'}")
