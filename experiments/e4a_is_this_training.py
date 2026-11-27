"""E4a —— "这是训练吗？"一个可判定的数学检验。

为什么必须做这个
================

用户问："确定你这个方式和训练 AI 大模型不一样吗，我要的是创新。"

诚实的回答需要判据，不是修辞。判据在这里：

    一个更新规则如果在参数空间上存在**标量势函数 L**，
    使得  Δw ∝ −∂L/∂w，
    那么它在数学上就是"训练"——不管叫它什么、有没有 `loss` 这个词、
    有没有 `backward()`、是不是纯局部。

理由：优化一个 L 就是"沿某个方向下山"。任何下山过程都被同一个 L 度量，
因此它的行为完全由 L 的形状决定。换名字、换分层、换冻结策略，都只是在
换下山的步长或停步时机，**不改变"在训练"这个性质**。

反过来，若更新场**不存在**势函数（旋度非零），则系统不可能被任何单一目标
解释，它才会出现"训练不出来的行为"（持续循环、极限环、路径依赖）。

检验方法
========

对更新场 F(W) = E[ΔW | W] 在参数点 W0 处取雅可比 J，
其中 J[a,b] = ∂F_a/∂W_b（把 W 展平成向量）。

    J 对称  ⟺  F 是梯度场  ⟺  存在 L，F = −∂L/∂W   ⟺  这是训练
    J 非对称 ⟹  curl F ≠ 0 ⟹  不存在 L            ⟹  这不是训练

这是 Helmholtz 分解的直接应用：梯度场的雅可比必对称（因为
∂²L/∂w_a∂w_b 可交换）。
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
SAMPLES = 4000


def _sample_data(a: np.ndarray, n: int, seed: int = 0):
    """从环境的平稳分布采样 (x, x_next)。噪声置零，只看确定性部分。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=D)
    X, Y = [], []
    for _ in range(SAMPLES):
        xn = a @ x
        X.append(x.copy())
        Y.append(xn[:n].copy())
        x = xn
        if np.linalg.norm(x) > 1e6:      # 数值保护
            x = rng.normal(size=D)
    return np.array(X), np.array(Y)


def field_lms(w_flat: np.ndarray, X: np.ndarray, Y: np.ndarray,
              eta: float, n: int) -> np.ndarray:
    """预测式 LMS 的一步期望更新：ΔW = η·E[(y − Wx) xᵀ]。"""
    W = w_flat.reshape(n, D)
    err = Y - X @ W.T                 # (S, n)
    g = -eta * (err.T @ X) / len(X)   # (n, d)
    return g.reshape(-1)


def field_multi_eta(w_flat: np.ndarray, X: np.ndarray, Y: np.ndarray,
                    etas: np.ndarray, n: int) -> np.ndarray:
    """逐细胞不同 η。检验"可塑性分层"是否改变场的性质。"""
    W = w_flat.reshape(n, D)
    err = Y - X @ W.T
    g = -((err * etas[None, :]).T @ X) / len(X)
    return g.reshape(-1)


def field_nonconservative(w_flat: np.ndarray, X: np.ndarray, Y: np.ndarray,
                          eta: float, n: int, kappa: float = 0.5) -> np.ndarray:
    """**反例**：在 LMS 上加一个反对称旋转项。

    ΔW = η·E[(y−Wx)xᵀ] + κ·(ΩW − WΩ)，Ω 反对称。
    旋转项不可积，用来证明雅可比检验确实能区分"训练"与"非训练"。
    这个规则不对应任何 L，纯粹是为了验证工具有效性。
    """
    base = field_lms(w_flat, X, Y, eta, n)
    W = w_flat.reshape(n, D)
    # 旋转必须作用在 n×D 上：W·R − R'·W，其中 R 是 D×D 反对称（右乘），
    # R' 是 n×n 反对称（左乘）。两侧都取反对称才能保证不可积。
    Om = np.zeros((D, D))
    for i in range(0, D - 1, 2):
        Om[i, i + 1], Om[i + 1, i] = 1.0, -1.0
    Om_l = np.zeros((n, n))
    for i in range(0, n - 1, 2):
        Om_l[i, i + 1], Om_l[i + 1, i] = 1.0, -1.0
    rot = kappa * (W @ Om - Om_l @ W)
    return base + rot.reshape(-1)


def jacobian(field, w0: np.ndarray, eps: float = 1e-5, **kw) -> np.ndarray:
    """中心差分雅可比。"""
    m = len(w0)
    J = np.zeros((m, m))
    for k in range(m):
        e = np.zeros(m)
        e[k] = eps
        J[:, k] = (field(w0 + e, **kw) - field(w0 - e, **kw)) / (2 * eps)
    return J


def asymmetry(J: np.ndarray) -> float:
    """0 = 完美对称（梯度场，即训练）；越大越非保守。"""
    return float(np.linalg.norm(J - J.T) / max(np.linalg.norm(J), 1e-12))


def check(name: str, field, w0: np.ndarray, **kw) -> dict:
    J = jacobian(field, w0, **kw)
    s = asymmetry(J)
    return {"name": name, "asymmetry": s,
            "verdict": "训练（存在势函数）" if s < 1e-3 else "非训练（无势函数）"}


def main() -> dict:
    a = V.make_env(1, D)
    X, Y = _sample_data(a, N)
    w0 = np.random.default_rng(0).normal(size=(N, D)) * 0.5
    w0 = w0.reshape(-1)

    print("=" * 78)
    print("E4a  这是训练吗？—— 更新场的雅可比对称性检验")
    print("=" * 78)
    print(f"\n参数空间维度 = {N}×{D} = {len(w0)}")
    print("判据：雅可比非对称度 < 1e-3  ⟹  存在势函数  ⟹  在训练")

    rows = []
    print(f"\n{'规则':<34} {'非对称度':>10}  判定")
    print("-" * 78)
    for name, kw in [
        ("LMS (η=0.30 全场统一)", dict(eta=0.30, n=N)),
        ("LMS (η=0.05)", dict(eta=0.05, n=N)),
    ]:
        r = check(name, field_lms, w0, X=X, Y=Y, **kw)
        rows.append(r)
        print(f"{r['name']:<34} {r['asymmetry']:>10.2e}  {r['verdict']}")

    for tag, etas in [
        ("逐细胞 η 分层 (0.3/0.003 混合)",
         np.linspace(0.30, 0.003, N)),
        ("逐细胞 η 全零（=冻结）", np.zeros(N)),
    ]:
        r = check(tag, field_multi_eta, w0, X=X, Y=Y, etas=etas, n=N)
        rows.append(r)
        print(f"{r['name']:<34} {r['asymmetry']:>10.2e}  {r['verdict']}")

    r = check("LMS + 反对称旋转（人造反例）", field_nonconservative,
              w0, X=X, Y=Y, eta=0.30, n=N, kappa=0.5)
    rows.append(r)
    print(f"{r['name']:<34} {r['asymmetry']:>10.2e}  {r['verdict']}")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print("""
1. LMS 的非对称度 = 0（机器精度）。它是 L = ½·E‖y − Wx‖² 的精确负梯度。
   所谓"无全局损失、纯局部"，在数学上不成立：
       逐细胞 Δw_i = η·err_i·x  恰好等于  −η·∂/∂w_i [½Σ_j err_j²]
   局部性只是"梯度可以逐坐标算"，不是"不存在全局目标"。

2. 逐细胞不同的 η（可塑性分层）、η→0（冻结）、gate 衰减，
   全部只是**逐坐标的正缩放**：Δw_i = −η_i·∂L_i/∂w_i，η_i > 0。
   正缩放的梯度场仍是梯度场（雅可比只是被对角相似变换了一下，对称性不变）。
   **所以分层、退火、闭锁，一个都没让这个系统离开"训练"。**

3. 只有人造的反对称项能把非对称度推上去。它证明检验有区分力，
   同时说明：真实规则要离开"训练"，必须让更新方向**不与任何 L 的梯度共线**。

一句话：当前 ③ 就是"一层线性回归 + SGD + 学习率退火 + 冻结表"。
Widrow–Hoff 1960。不是创新。
""")
    return {"rows": rows}


if __name__ == "__main__":
    main()
