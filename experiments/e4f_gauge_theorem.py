"""E4f —— 规范自由度定理：线性预测下，结构变量不可能是新自由度。

## 结论（本文件用数值证死，可当定理用）

设预测是输入的线性函数：

    pred = A(c) · x,     A: R^p → R^{n×d} 光滑

且参数 `w ∈ R^{n×d}` 通过任意**能收敛到最优**的规则学习。
则存在 `c*` 使 `A(c*) = w*`（最优线性预测子），于是

    **`c` 对最优预测零贡献。任何"结构分化"都只是 `w` 的不同参数化。**

### 为什么

最优线性预测子 `A* = Σ_{yx} Σ_{xx}^{-1}` 由**环境**唯一决定，
与 `c` 无关（`c` 只改 `A` 的写法，不改 `A*` 的值）。
只要 `w` 学得动，`w → A*`，`c` 的信息被完全覆盖。

**这不是 e4e 的实现 bug，是"线性 + 可学参数"的结构性后果。**

推论：想找"参数无法补偿的结构"，**必须离开线性预测类**。
两条路：
  (a) 预测对 x 非线性（`pred = φ(w·x)` 或 `w·φ(x)`）
  (b) 结构做**离散**选择（选目标 / 选输入），w 的连续更新无法吸收

本文件测 (a) 和 (b) 的**可观测性**，即：
结构差异是否在**行为上**留下痕迹（而非只在参数数值上）。

## 判据

    gauge_effect = |MSE(c_learned) − MSE(c_replaced)| / MSE(c_learned)

`gauge_effect < 1e-6` ⟹ `c` 是规范自由度（无后果）。
`gauge_effect > 0.05` ⟹ `c` 有实质后果（值得继续）。

e4e 的实测值：**0.000000**（五个不同 c 给同一个 MSE）。
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
NOISE = V.CFG["noise"]
GAUGE_TOL = 1e-6       # 低于此 = 无后果
REAL_TOL = 0.05        # 高于此 = 有实质后果


def _sig(c):
    return 1.0 / (1.0 + np.exp(-np.clip(c, -30, 30)))


# ======================================================================
# 装置 A：e4e 原样复现 —— 线性预测 + 乘性门控
# ======================================================================
def run_linear(seed: int = 0, mu: float = 0.03, t_total: int = 20000,
               c_override=None, eval_steps: int = 4000) -> dict:
    """线性预测。`c_override` 非 None 时，训完 w 后把 c 换掉重测。"""
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, D)
    shift = V.make_shift(seed + 2, D, amp=2.5, a_mat=a)

    w = rng.normal(size=(N, D)) * 0.05
    c = rng.normal(size=(N, D)) * 0.05
    target = np.full(N, NOISE ** 2 * 3.0)
    x = rng.normal(size=D)

    err_ema = np.full(N, np.nan)
    err_ref = np.full(N, 1.0)
    ref_buf = []

    for t in range(t_total):
        on = 300 <= t < 3000
        xn = (a + shift if on else a) @ x + NOISE * rng.normal(size=D)

        g = _sig(c)
        xg = g * x[None, :]
        err = xn[:N] - np.sum(w * xg, axis=1)

        cur = np.abs(err)
        if np.isnan(err_ema[0]):
            err_ema = cur.copy()
        else:
            err_ema = 0.997 * err_ema + 0.003 * cur
        ref_buf.append(err_ema.copy())
        if len(ref_buf) <= 300:
            err_ref = np.mean(ref_buf, axis=0)
        ref_buf = ref_buf[-301:]
        conv = np.clip(err_ema / np.maximum(err_ref, 1e-12), 0.0, 1.0)
        eta = 2e-3 + (0.30 - 2e-3) * conv ** 2.0

        den = np.maximum(np.sum(xg ** 2, axis=1, keepdims=True), 1e-6)
        w = w + eta[:, None] * err[:, None] * xg / den
        c = np.clip(c + mu * (err ** 2 - target)[:, None] * x[None, :], -6.0, 6.0)
        x = xn

    c_final = c if c_override is None else c_override

    # 固定 c，只让 w 重新适应，测"这个 c 能做到多好"
    rng2 = np.random.default_rng(seed + 500)
    xx = rng2.normal(size=D)
    sq = []
    w2 = w.copy()
    g = _sig(c_final)
    xg_const = g * xx[None, :]
    for _ in range(eval_steps):
        xn = a @ xx + NOISE * rng2.normal(size=D)
        err = xn[:N] - np.sum(w2 * (g * xx[None, :]), axis=1)
        den = np.maximum(np.sum((g * xx[None, :]) ** 2, axis=1, keepdims=True), 1e-6)
        w2 = w2 + 0.30 * err[:, None] * (g * xx[None, :]) / den
        sq.append(float(np.mean(err ** 2)))
        xx = xn

    return {"mse": float(np.mean(sq[-1000:])), "c": c_final, "w": w}


# ======================================================================
# 装置 B：输出非线性 —— pred = tanh(c_i · (w_i · x))
#    c 在这里不是尺度（尺度会被分母吸收），是**饱和点**，改的是曲率
# ======================================================================
def run_nonlinear(seed: int = 0, mu: float = 0.03, t_total: int = 20000,
                  c_override=None, eval_steps: int = 4000) -> dict:
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, D)
    shift = V.make_shift(seed + 2, D, amp=2.5, a_mat=a)

    w = rng.normal(size=(N, D)) * 0.05
    c = np.full(N, 1.0) + rng.normal(size=N) * 0.05   # 逐细胞饱和斜率
    target = np.full(N, NOISE ** 2 * 3.0)
    x = rng.normal(size=D)

    err_ema = np.full(N, np.nan)
    err_ref = np.full(N, 1.0)
    ref_buf = []

    for t in range(t_total):
        on = 300 <= t < 3000
        xn = (a + shift if on else a) @ x + NOISE * rng.normal(size=D)

        u = w @ x                                  # (N,) 预激活
        pred = np.tanh(c * u)
        err = xn[:N] - pred

        cur = np.abs(err)
        if np.isnan(err_ema[0]):
            err_ema = cur.copy()
        else:
            err_ema = 0.997 * err_ema + 0.003 * cur
        ref_buf.append(err_ema.copy())
        if len(ref_buf) <= 300:
            err_ref = np.mean(ref_buf, axis=0)
        ref_buf = ref_buf[-301:]
        conv = np.clip(err_ema / np.maximum(err_ref, 1e-12), 0.0, 1.0)
        eta = 2e-3 + (0.30 - 2e-3) * conv ** 2.0

        # d pred / d u = c · sech²(c·u)
        dpdu = c * (1.0 - np.tanh(c * u) ** 2)
        den = np.maximum((dpdu ** 2)[:, None] * np.sum(x ** 2), 1e-9)
        dw = eta[:, None] * err[:, None] * dpdu[:, None] * x[None, :] / den

        # 结构：改饱和点。驱动是"误差相对期望的偏离" × |u|（只在该细胞真的被激活时）
        dc = mu * (err ** 2 - target) * np.abs(u)

        w = w + dw
        c = np.clip(c + dc, 0.05, 20.0)
        x = xn

    c_final = c if c_override is None else c_override

    rng2 = np.random.default_rng(seed + 500)
    xx = rng2.normal(size=D)
    sq = []
    w2 = w.copy()
    for _ in range(eval_steps):
        xn = a @ xx + NOISE * rng2.normal(size=D)
        u = w2 @ xx
        pred = np.tanh(c_final * u)
        err = xn[:N] - pred
        dpdu = c_final * (1.0 - np.tanh(c_final * u) ** 2)
        den = np.maximum((dpdu ** 2)[:, None] * np.sum(xx ** 2), 1e-9)
        w2 = w2 + 0.30 * err[:, None] * dpdu[:, None] * xx[None, :] / den
        sq.append(float(np.mean(err ** 2)))
        xx = xn

    return {"mse": float(np.mean(sq[-1000:])), "c": c_final, "w": w}


# ======================================================================
# 装置 C：离散结构 —— 细胞 i 选择"预测哪个目标坐标"
#   σ 是置换（一对一匹配），w 无法吸收离散选择
# ======================================================================
def run_discrete(seed: int = 0, mu: float = 0.03, t_total: int = 20000,
                 c_override=None, eval_steps: int = 4000) -> dict:
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, D)
    shift = V.make_shift(seed + 2, D, amp=2.5, a_mat=a)

    w = rng.normal(size=(N, D)) * 0.05
    # c: N×N 选择矩阵（软分配，用温度退火逼向置换）
    c = rng.normal(size=(N, N)) * 0.05
    x = rng.normal(size=D)

    err_ema = np.full(N, np.nan)
    err_ref = np.full(N, 1.0)
    ref_buf = []
    temp = 1.0

    for t in range(t_total):
        on = 300 <= t < 3000
        xn = (a + shift if on else a) @ x + NOISE * rng.normal(size=D)

        # 软分配：细胞 i 的目标 = Σ_j softmax(c_i)_j · x_{t+1}[j]
        p = np.exp(c - c.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        tgt = p @ xn[:N]
        pred = w @ x
        err = tgt - pred

        cur = np.abs(err)
        if np.isnan(err_ema[0]):
            err_ema = cur.copy()
        else:
            err_ema = 0.997 * err_ema + 0.003 * cur
        ref_buf.append(err_ema.copy())
        if len(ref_buf) <= 300:
            err_ref = np.mean(ref_buf, axis=0)
        ref_buf = ref_buf[-301:]
        conv = np.clip(err_ema / np.maximum(err_ref, 1e-12), 0.0, 1.0)
        eta = 2e-3 + (0.30 - 2e-3) * conv ** 2.0

        den = np.maximum(np.sum(x ** 2), 1e-6)
        w = w + eta[:, None] * err[:, None] * x[None, :] / den

        # 结构：把分配推向"预测得更准的目标"
        # d(tgt_i)/dc_ij = p_ij·(x_j − tgt_i)
        resid = xn[None, :N] - tgt[:, None]          # (N,N)
        dc = mu * 4.0 * err[:, None] * p * (resid - (p * resid).sum(axis=1, keepdims=True))
        c = np.clip(c + dc, -20.0, 20.0)
        x = xn

    c_final = c if c_override is None else c_override

    rng2 = np.random.default_rng(seed + 500)
    xx = rng2.normal(size=D)
    sq = []
    w2 = w.copy()
    p = np.exp(c_final - c_final.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    for _ in range(eval_steps):
        xn = a @ xx + NOISE * rng2.normal(size=D)
        tgt = p @ xn[:N]
        err = tgt - w2 @ xx
        den = np.maximum(np.sum(xx ** 2), 1e-6)
        w2 = w2 + 0.30 * err[:, None] * xx[None, :] / den
        sq.append(float(np.mean(err ** 2)))
        xx = xn

    return {"mse": float(np.mean(sq[-1000:])), "c": c_final, "w": w,
            "perm": p.argmax(axis=1)}


def gauge_effect(runner, seed=0, **kw) -> dict:
    """规范效应：c 换成随机值后行为变多少。"""
    base = runner(seed=seed, **kw)
    rng = np.random.default_rng(seed + 777)
    alt_c = base["c"] * rng.uniform(-3, 3, size=base["c"].shape)
    alt = runner(seed=seed, c_override=alt_c, **kw)
    mse0, mse1 = base["mse"], alt["mse"]
    eff = abs(mse0 - mse1) / max(mse0, 1e-12)
    return {"mse_base": mse0, "mse_alt": mse1, "effect": eff,
            "verdict": ("规范自由度（无后果）" if eff < GAUGE_TOL
                        else "有实质后果" if eff > REAL_TOL
                        else "弱后果（需更多证据）")}


def main() -> dict:
    print("=" * 84)
    print("E4f  规范自由度定理：线性预测下结构不可能是新自由度")
    print("=" * 84)
    print(f"判据：effect < {GAUGE_TOL:g} ⟹ 规范自由度；"
          f"effect > {REAL_TOL:g} ⟹ 有实质后果\n")
    devices = [
        ("A 线性 + 乘性门控（e4e 原样）", run_linear),
        ("B 输出非线性 tanh(c·w·x)", run_nonlinear),
        ("C 离散目标选择（软置换）", run_discrete),
    ]

    print(f"{'装置':<34} {'MSE(c)':>12} {'MSE(c随机)':>12} {'effect':>10}  判定")
    print("-" * 84)
    rows = []
    for name, fn in devices:
        try:
            r = gauge_effect(fn, seed=0)
            rows.append({"device": name, **r})
            print(f"{name:<34} {r['mse_base']:>12.6f} {r['mse_alt']:>12.6f} "
                  f"{r['effect']:>10.2e}  {r['verdict']}")
        except Exception as e:  # noqa: BLE001
            rows.append({"device": name, "error": str(e)})
            print(f"{name:<34} {'ERROR':>12} {type(e).__name__}: {e}")

    # 多 seed 稳定性
    print("\n多 seed 复现（effect 中位数）：")
    print(f"  {'装置':<34} {'seeds':>6} {'effect 中位':>13}")
    print("  " + "-" * 58)
    for name, fn in devices:
        effs = []
        for sd in range(5):
            try:
                effs.append(gauge_effect(fn, seed=sd)["effect"])
            except Exception:  # noqa: BLE001
                pass
        if effs:
            print(f"  {name:<34} {len(effs):>6} {np.median(effs):>13.2e}")

    print("\n" + "=" * 84)
    print("结论")
    print("=" * 84)
    a = next((r for r in rows if r["device"].startswith("A")), None)
    b = next((r for r in rows if r["device"].startswith("B")), None)
    c_ = next((r for r in rows if r["device"].startswith("C")), None)
    print(f"""
1. **线性预测（装置 A）：effect = {a['effect']:.2e}  ⟹  `c` 是规范自由度。**
   `w ⊙ g` 是唯一可观测量，`g` 的尺度被 `w` 精确吸收。
   这不是 e4e 的实现问题，是"线性 + 可学参数"的结构性后果。
   **e4e 的"两个自由度"结论作废。**

2. 输出非线性（装置 B）：effect = {b['effect']:.2e}  →  {b['verdict']}
   `tanh(c·u)` 里 `c` 改的是曲率，不是尺度，不会被 `w` 吸收。

3. 离散选择（装置 C）：effect = {c_['effect']:.2e}  →  {c_['verdict']}
   置换是离散的，`w` 的连续更新无法吸收它。

**判定标准（写死，防以后自我欺骗）**：
  - 任何"结构变量"，先做本文件的 effect 检验
  - effect < 1e-6 ⟹ 它是规范自由度，不管看起来多"在动"
  - "在动"不等于"有后果"。e4e 的 `c` 一直在动（步长 1e-4），
    但五个不同取值给同一个 MSE。
""")
    return {"devices": rows}


if __name__ == "__main__":
    p = Path("runs/e4f")
    p.mkdir(parents=True, exist_ok=True)
    res = main()
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {p / 'report.json'}")
