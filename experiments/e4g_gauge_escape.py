"""E4g —— 对装置 B（tanh 非线性）做雅可比判决。

## 为什么必须先做这一步

E4f 证明：线性预测下结构变量是规范自由度（effect ≈ 0）。
输出非线性 `pred = tanh(c·u)` 是第一个结构**有行为后果**的装置（effect = 58）。

但"有后果"≠"不是训练"。按 `docs/11` §12.1 的判据，唯一算数的检验是：
把 `(w, c)` 全体看成参数 `θ`，取更新场 `F(θ) = E[Δθ|θ]`，
**看它的雅可比 J 是否对称**。

    J 对称   ⟺  存在 L 使 F = −∂L/∂θ  ⟺  这是训练
    J 非对称 ⟹  curl F ≠ 0          ⟹  不存在任何 L ⟹ 不是训练

## 本文件的对照设计（关键）

三个变体，只改 `dc` 一项，其余完全相同：

    B-grad   dc = −∂L/∂c 的精确梯度       → J 应当对称（对照）
    B-heur   dc = μ·(err²−target)·|u|     → 待测（e4f 装置 B 原样）
    B-flip   dc = μ·(−1)^i · ...          → 人造反例，验证工具有区分力

**如果 B-heur 的 J 也对称，那它就是训练，只是换了个损失函数。**
**如果 B-heur 非对称，那是本项目第一次造出"结构走非保守方向"的装置。**

## 参数量

    θ = (w, c) ∈ R^{6×12 + 6} = R^{78}
    中心差分雅可比：156 次场求值，每次 4000 样本 —— 秒级。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3v3_core as V

D = 6          # 降维：雅可比可读，且与 e4a 的 72 维同量级
N = 4
S = 4000       # 样本数（求期望）
ETA = 0.30
MU = 0.05
TARGET = 0.02


def _dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, D)
    # 独立同分布输入：让 E 是真正的外期望，不含 x 的自相关（简化判决）
    X = rng.normal(size=(S, D))
    Y = X @ a[:N].T + 0.05 * rng.normal(size=(S, N))
    return X, Y


def unpack(theta: np.ndarray):
    w = theta[: N * D].reshape(N, D)
    c = theta[N * D:]
    return w, c


def pack(w: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.concatenate([w.reshape(-1), c])


def field(theta: np.ndarray, X: np.ndarray, Y: np.ndarray,
          variant: str) -> np.ndarray:
    """一步期望更新 F(θ) = E[Δθ]. 三个变体只改 dc。"""
    w, c = unpack(theta)
    u = X @ w.T                          # (S, N)
    p = np.tanh(c[None, :] * u)          # (S, N)
    err = Y - p                          # (S, N)
    dLdp = -2.0 * err / (S * N)          # L = mean(err²)
    dpdu = c[None, :] * (1.0 - p ** 2)   # (S, N)

    # ---- 参数 w：梯度 ----
    dw = -ETA * ((dLdp * dpdu).T @ X)    # (N, D)

    # ---- 结构 c ----
    if variant == "grad":
        # L = (1/(S·N))·Σ err²，故 ∂L/∂c_i = Σ_s dLdp_si·(1−p²)_si·u_si
        # （dp/dc_i = u·(1−p²)，不是 c·u·(1−p²) —— 那是 dp/du）
        dLdc = np.sum(dLdp * (1.0 - p ** 2) * u, axis=0)
        dc = -MU * dLdc
    elif variant == "heur":
        dc = MU * np.mean((err ** 2 - TARGET) * np.abs(u), axis=0)
    elif variant == "flip":
        sgn = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
        dc = MU * sgn * np.mean((err ** 2 - TARGET) * np.abs(u), axis=0)
    else:
        raise ValueError(variant)

    return pack(dw, dc)


def jacobian(theta: np.ndarray, X, Y, variant: str, h: float = 1e-6) -> np.ndarray:
    m = theta.size
    J = np.zeros((m, m))
    for j in range(m):
        e = np.zeros(m)
        e[j] = h
        J[:, j] = (field(theta + e, X, Y, variant)
                   - field(theta - e, X, Y, variant)) / (2 * h)
    return J


def asymmetry(J: np.ndarray) -> float:
    return float(np.linalg.norm(J - J.T) / max(np.linalg.norm(J), 1e-12))


def steps_vector() -> np.ndarray:
    """每个坐标的步长。w 用 ETA，c 用 MU。"""
    return np.concatenate([np.full(N * D, ETA), np.full(N, MU)])


def preconditioned_asymmetry(J: np.ndarray) -> float:
    """按步长归一化后再判对称 —— 这才是正确的判据。

    **这是一个修正，不是一个补丁。** 上一版（以及只看 ‖J−Jᵀ‖ 的所有版本）
    犯的错：

        真实更新是  F = −D·∇L,   D = diag(步长)
        其雅可比是  J = −D·H       （H = ∇²L 对称）

    `D·H` **一般不对称**，即使 `H` 对称。举例：D=diag(1,100)、H=[[1,1],[1,1]]，
    D·H = [[1,1],[100,100]]，转置是 [[1,100],[1,100]]，差得很远。

    所以"J 不对称 ⟹ 非保守"这个推论**只在 D 是单位阵倍数时成立**。
    正确判据是：把步长除掉，看 `D⁻¹J = −H` 是否对称。

    这与 e4a 的结论不冲突：那里所有 η 都是同一变量的行缩放，
    缩放索引恰好等于雅可比的块索引，所以块对角 × 对称仍是块对称。
    跨**不同变量**（w 与 c）用不同步长才会破坏它。
    """
    d = steps_vector()
    Jn = J / d[:, None]
    return float(np.linalg.norm(Jn - Jn.T) / max(np.linalg.norm(Jn), 1e-12))


def block_asymmetry(J: np.ndarray) -> dict:
    """分块看（先按步长归一化）：w-w、c-c、w-c 交叉。"""
    d = steps_vector()
    Jn = J / d[:, None]
    nw = N * D
    Jww = Jn[:nw, :nw]
    Jcc = Jn[nw:, nw:]
    Jwc = Jn[:nw, nw:]      # ∂(F_w/η)/∂c
    Jcw = Jn[nw:, :nw]      # ∂(F_c/μ)/∂w
    norm = max(float(np.linalg.norm(Jn)), 1e-12)
    return {
        "ww": float(np.linalg.norm(Jww - Jww.T) / norm),
        "cc": float(np.linalg.norm(Jcc - Jcc.T) / norm),
        # 交叉块：对称要求 Jwc == Jcwᵀ
        "cross": float(np.linalg.norm(Jwc - Jcw.T) / norm),
    }


def analytic_grad_check(theta, X, Y) -> dict:
    """B-grad 的解析梯度必须与数值梯度一致（证明 grad 变体实现正确）。

    ⚠️ 这里同时校验 w 和 c 两部分。上一版只校验 w，把 c 的错误放过去了
    —— 这是本文件第一次运行给出"对照也不对称"的直接原因。
    """
    w, c = unpack(theta)
    u = X @ w.T
    p = np.tanh(c[None, :] * u)
    err = Y - p
    dLdp = -2.0 * err / (S * N)
    dpdu = c[None, :] * (1.0 - p ** 2)

    g_w = ((dLdp * dpdu).T @ X).reshape(-1)
    g_c = np.sum(dLdp * (1.0 - p ** 2) * u, axis=0)

    eps = 1e-6

    def L(th):
        w_, c_ = unpack(th)
        p_ = np.tanh(c_[None, :] * (X @ w_.T))
        return float(np.mean((Y - p_) ** 2))

    num = np.zeros(theta.size)
    for j in range(theta.size):
        e = np.zeros(theta.size)
        e[j] = eps
        num[j] = (L(theta + e) - L(theta - e)) / (2 * eps)

    ana = np.concatenate([g_w, g_c])
    nw = N * D
    return {
        "all": float(np.linalg.norm(ana - num) / max(np.linalg.norm(num), 1e-12)),
        "w": float(np.linalg.norm(ana[:nw] - num[:nw])
                   / max(np.linalg.norm(num[:nw]), 1e-12)),
        "c": float(np.linalg.norm(ana[nw:] - num[nw:])
                   / max(np.linalg.norm(num[nw:]), 1e-12)),
    }


def main() -> dict:
    print("=" * 84)
    print("E4g  装置 B 的雅可比判决：结构项是不是在「下山」")
    print("=" * 84)
    print(f"θ ∈ R^{N*D + N}（w: {N}×{D}, c: {N}），样本 {S}，中心差分 h=1e-6\n")

    X, Y = _dataset(0)
    rng = np.random.default_rng(3)
    w0 = rng.normal(size=(N, D)) * 0.3
    c0 = np.ones(N) + 0.1 * rng.normal(size=N)
    theta0 = pack(w0, c0)

    rel = analytic_grad_check(theta0, X, Y)
    ok = rel["all"] < 1e-6
    print(f"[0] 解析梯度 vs 数值梯度（B-grad）："
          f"全部 {rel['all']:.2e} / w {rel['w']:.2e} / c {rel['c']:.2e}"
          f"  {'✅' if ok else '❌ 实现有误，以下结论全部作废'}\n")
    if not ok:
        return {"error": "gradient check failed", "rel": rel}


    print(f"{'变体':<10} {'‖J‖':>10} {'裸非对称':>10} {'归一后':>10} "
          f"{'ww':>9} {'cc':>9} {'交叉':>9}  判定")
    print("-" * 92)
    rows = []
    for variant, tag in [("grad", "B-grad"), ("heur", "B-heur"), ("flip", "B-flip")]:
        J = jacobian(theta0, X, Y, variant)
        raw = asymmetry(J)
        tot = preconditioned_asymmetry(J)
        blk = block_asymmetry(J)
        if tot < 1e-6:
            v = "训练（梯度场）"
        elif tot > 0.1:
            v = "**非训练（非保守场）**"
        else:
            v = "弱非对称（需复查）"
        rows.append({"variant": tag, "norm": float(np.linalg.norm(J)),
                     "asym_raw": raw, "asym": tot, **blk, "verdict": v})
        print(f"{tag:<10} {np.linalg.norm(J):>10.3e} {raw:>10.3e} {tot:>10.3e} "
              f"{blk['ww']:>9.2e} {blk['cc']:>9.2e} {blk['cross']:>9.2e}  {v}")

    print("\n多 seed 复现（归一化非对称度）：")
    print(f"  {'变体':<10} {'seeds':>6} {'中位':>12} {'最大':>12}")
    print("  " + "-" * 46)
    stable = {}
    for variant, tag in [("grad", "B-grad"), ("heur", "B-heur"), ("flip", "B-flip")]:
        vals = []
        for sd in range(4):
            Xs, Ys = _dataset(sd)
            th = pack(rng.normal(size=(N, D)) * 0.3,
                      np.ones(N) + 0.1 * rng.normal(size=N))
            vals.append(preconditioned_asymmetry(jacobian(th, Xs, Ys, variant)))
        stable[tag] = {"median": float(np.median(vals)), "max": float(np.max(vals))}
        print(f"  {tag:<10} {len(vals):>6} {np.median(vals):>12.3e} "
              f"{np.max(vals):>12.3e}")

    print("\n" + "=" * 84)
    print("结论")
    print("=" * 84)
    g = next(r for r in rows if r["variant"] == "B-grad")
    h = next(r for r in rows if r["variant"] == "B-heur")
    f = next(r for r in rows if r["variant"] == "B-flip")
    nonconserve = h["asym"] > 0.1
    print(f"""
1. 对照 B-grad（纯梯度）：非对称 = {g['asym']:.2e}
   {'✅ 对称 —— 证明判决工具有区分力，且实现正确' if g['asym'] < 1e-6 else '❌ 对照都不对称，工具/实现有问题，本轮结论全部作废'}

2. 人造反例 B-flip：非对称 = {f['asym']:.2e}
   {'✅ 被检出' if f['asym'] > 0.1 else '❌ 没检出，工具灵敏度不够'}

3. **e4f 装置 B 原样（B-heur）：非对称 = {h['asym']:.2e}**
   {'**⟹ 这是本项目第一次造出"结构走非保守方向"的装置。**' if nonconserve else '⟹ 它仍是梯度场，只是换了损失函数。"结构有后果"和"不是训练"是两件事。'}
   分块：w-w = {h['ww']:.2e}，c-c = {h['cc']:.2e}，w-c 交叉 = {h['cross']:.2e}
   {'非对称主要集中在结构项，符合设计意图' if h['cc'] + h['cross'] > h['ww'] else '非对称主要来自参数项，需查'}

## 这一步之后必须立刻做的（否则不算数）

非保守 ≠ 有用。非保守场必须带来**训练做不到的现象**，否则只是噪声：

    (a) 历史依赖：早期环境不同 → 终态不同，且晚期的相同环境抹不平
        （这才是 A3 的直接检验，docs/11 §12.6 第 1 条）
    (b) 不收敛：终态仍在动，且不是稳态误调抖动（e4b 踩过的坑）
    (c) 行为后果：结构差必须对应预测差（e4f 已建立 B 的 effect=58）

**(a) 是唯一有决定性的。** 若 (a) 不成立，非保守性只是"动力学更花哨"，
不构成创新。
""")
    return {"rows": rows, "stable": stable, "grad_rel": rel}


if __name__ == "__main__":
    p = Path("runs/e4g")
    p.mkdir(parents=True, exist_ok=True)
    res = main()
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"写入 {p / 'report.json'}")
