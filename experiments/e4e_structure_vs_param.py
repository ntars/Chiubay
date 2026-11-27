"""E4e —— 尝试 5：参数学、结构动，两个自由度分开。

## ❌ 结论：失败。`c` 是规范自由度（gauge freedom），不携带任何信息。

带 `CHIUBAY_ALLOW_E4E=1` 才运行。默认拒绝执行，防止把已判死的装置
再当成"有希望的方向"。

### 判死证据（三重，互相独立）

**证据 1 —— 消融**（`/tmp/probe_c.py`）：训好 `w` 后冻结，只把 `c` 换成
五个完全不同的取值，重测 MSE：

    c = 学到的值            MSE = 0.002958
    c = 全零（门控 0.50）    MSE = 0.002958
    c = 全 +3（门控 0.95）   MSE = 0.002958
    c = 全 −3（门控 0.05）   MSE = 0.002958
    c = 随机打乱             MSE = 0.002958

**五个值给出同一个数到小数点后六位。** `c` 对行为零影响。

**证据 2 —— 门控本身没动**：`c ∈ [3.3e-4, 0.125]`，`g = sigmoid(c) ∈
[0.4687, 0.5261]`，`std(g) = 0.0128`。所谓"门控分化 0.011"就是
`sigmoid` 在 0 附近的斜率，不是分化。

**证据 3 —— 解析**：归一化 LMS 的更新是

    Δw_i = η_i · err_i · (g_i ⊙ x) / |g_i ⊙ x|²

`g_i` 在分子**和**分母上同时出现，**尺度完全抵消**。也就是说
`g` 取 0.05 还是 0.95，`w` 的轨迹逐字相同（`w` 会精确补偿成 `w/g`），
预测 `w ⊙ g` 不变。**任何正的可乘结构变量在归一化 LMS 下都是规范自由度。**

这条是普遍结论，不是这个脚本的实现问题：
**线性预测 + 归一化 LMS ⟹ 逐坐标正尺度结构不可观测。**

### 教训（已写入 docs/11 §12.5 尝试 5）

"两个自由度分开"这个方向本身没错，错在**选的结构变量落在参数的规范轨道上**。
要让结构有后果，必须选参数**无法补偿**的方向 —— 见 `e4f_gauge_escape.py`。

---

⚠️ 本文件还修过两次自己犯的错，一并记下：

**错误 A（v1）**：`c` 只是加在更新里的一个向量，**没有参与预测**。
实测：所有 μ 下 pred MSE 与 w 步长**完全相同**（0.002787 / 4.03e-02），
`c` 是死变量。而且 `c` 在每个 seed 上都饱和到同一个界
（‖c‖=7.348=√6×3），即它是个常数，不携带任何信息。

**错误 B（v1）**：`make_shift(2, D, a_mat=make_env(1,D))` 用的 `a`
与 `develop` 内部用的 `make_env(seed+1,D)` **不是同一个矩阵**。
seed=3 时两者不匹配，`a+shift` 谱半径 > 1 → 数值发散 → nan。

**修法**：
  A. 让 `c` 真正控制读取：cell i 只通过自己的**连接门**看输入
         g_i = sigmoid(c_i)              （逐维门控，∈(0,1)）
         pred_i = w_i · (g_i ⊙ x)
     这样 `c` 变了，预测就变了 —— 结构才有后果。
  B. shift 必须用 `develop` 内部同一个 `a` 来构造。

设计
====

    pred_i = w_i · (g_i ⊙ x),   g_i = sigmoid(c_i)
    err_i  = x_next[i] − pred_i

    Δw_i = η_i · err_i · (g_i ⊙ x) / |g_i ⊙ x|²      ← 学（梯度，会收敛）
    Δc_i = μ · (err_i² − target_i) · x                ← 结构（追逐输入）

`Δw` 是梯度（对给定 g，w 的最优是 A_i 的对应分量），所以 w 会收敛。
`Δc` 的驱动量是"误差相对期望的偏离"乘以**当前输入 x**，它没有不动点
（x 一直在变），所以 c 永不停止。

要检验：
  1. w 收敛（pred MSE 接近地板）
  2. c 不停（结构一直在动），且不饱和成常数
  3. c 携带早期养育信息（不同早期 → 不同 c），且后期换环境抹不掉
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3v3_core as V

ALLOW_RUN = os.environ.get("CHIUBAY_ALLOW_E4E") == "1"

D = V.CFG["d"]
N = V.CFG["n_cells"]
NOISE = V.CFG["noise"]


def _sig(c):
    return 1.0 / (1.0 + np.exp(-np.clip(c, -30, 30)))


def develop(seed: int, windows, mu: float = 0.02, eta0: float = 0.30,
            eta_min: float = 2e-3, gamma: float = 2.0,
            second=None, b_start: int = 5000,
            n: int = N, d: int = D, t_total: int = 20000,
            record_every: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    a = V.make_env(seed + 1, d)                 # ← 唯一的环境矩阵
    shift = V.make_shift(seed + 2, d, amp=2.5, a_mat=a)   # ← 用同一个 a 构造

    w = rng.normal(size=(n, d)) * 0.05
    c = rng.normal(size=(n, d)) * 0.05
    target = np.full(n, NOISE ** 2 * 3.0)
    x = rng.normal(size=d)

    sq, c_hist, w_hist, g_hist = [], [], [], []
    err_ema = np.full(n, np.nan)
    err_ref = np.full(n, 1.0)
    ref_buf = []

    for t in range(t_total):
        on = any(lo <= t < hi for lo, hi in windows)
        if on and second is not None and t >= b_start:
            a_now = a + second
        else:
            a_now = a + shift if on else a
        xn = a_now @ x + NOISE * rng.normal(size=d)

        g = _sig(c)                     # 门控：结构决定"看到什么"
        xg = g * x[None, :]             # (n, d) 每个细胞看到的输入
        pred = np.sum(w * xg, axis=1)
        err = xn[:n] - pred

        # ---- η：逐细胞收敛比 ----
        cur = np.abs(err)
        if np.isnan(err_ema[0]):
            err_ema = cur.copy()
        else:
            err_ema = 0.997 * err_ema + 0.003 * cur
        ref_buf.append(err_ema.copy())
        if len(ref_buf) <= 300:
            err_ref = np.mean(ref_buf, axis=0)
        ref_buf = ref_buf[-(301):]
        conv = np.clip(err_ema / np.maximum(err_ref, 1e-12), 0.0, 1.0)
        eta = eta_min + (eta0 - eta_min) * conv ** gamma

        # ---- 参数：学（梯度）----
        den = np.maximum(np.sum(xg ** 2, axis=1, keepdims=True), 1e-6)
        dw = eta[:, None] * err[:, None] * xg / den

        # ---- 结构：动（不是任何 L 的梯度）----
        # 误差大于期望 → 该细胞"看错了"，把门朝当前输入方向打开/关闭；
        # 误差小于期望 → 反向。驱动量随 x 变，没有不动点。
        drive = (err ** 2 - target)
        dc = mu * drive[:, None] * x[None, :]

        w = w + dw
        c = np.clip(c + dc, -6.0, 6.0)   # 门控在 sigmoid 里，裁剪只是数值保护

        sq.append(float(np.mean(err ** 2)))
        if t % record_every == 0:
            c_hist.append(c.copy())
            w_hist.append(w.copy())
            g_hist.append(g.copy())
        x = xn

    return {"w": w, "c": c, "a_mat": a, "shift": shift,
            "pred_mse": float(np.mean(sq[-500:])),
            "c_hist": np.array(c_hist), "w_hist": np.array(w_hist),
            "g_hist": np.array(g_hist)}


def motion(hist: np.ndarray, frac: float = 0.25) -> float:
    late = hist[int(len(hist) * (1 - frac)):]
    return float(np.mean(np.linalg.norm(np.diff(late, axis=0), axis=(1, 2))))


def gate_diversity(g_hist: np.ndarray) -> float:
    """门控是否分化：各细胞看到的输入模式差异。0 = 全都一样（没分化）。"""
    g = g_hist[-1]
    return float(np.mean(np.std(g, axis=0)))


def main() -> dict:
    print("=" * 86)
    print("E4e  参数学、结构动（修完两个 bug 的版本）")
    print("=" * 86)
    print(f"噪声地板 = {NOISE ** 2:.6f}\n")

    print(f"{'μ':>8} {'pred MSE':>10} {'w 步长':>10} {'c 步长':>10} "
          f"{'门控分化':>9}  判定")
    print("-" * 86)
    rows = []
    for mu in (0.0, 0.002, 0.01, 0.03, 0.1):
        r = develop(0, [(300, 19700)], mu=mu, t_total=20000)
        wp, cp = motion(r["w_hist"]), motion(r["c_hist"])
        gd = gate_diversity(r["g_hist"])
        learns = r["pred_mse"] < NOISE ** 2 * 3
        finite = np.isfinite(r["pred_mse"])
        if mu == 0.0:
            v = "对照：结构静" if cp < 1e-12 else "对照异常"
        elif not finite:
            v = "发散"
        elif learns and cp > 1e-9:
            v = "✅ 学会 + 结构仍在动"
        elif not learns:
            v = "学不会"
        else:
            v = "结构停了"
        rows.append({"mu": mu, "pred_mse": r["pred_mse"], "w_step": wp,
                     "c_step": cp, "gate_div": gd,
                     "learns": bool(learns), "verdict": v})
        print(f"{mu:>8.3f} {r['pred_mse']:>10.6f} {wp:>10.2e} {cp:>10.2e} "
              f"{gd:>9.4f}  {v}")

    ok = [r for r in rows if r["learns"] and r["mu"] > 0 and r["c_step"] > 1e-9]
    print(f"\n可用 μ（学会 + 结构动）：{[r['mu'] for r in ok]}")

    if not ok:
        print("\n没有可用 μ。")
        return {"rows": rows, "memory": None}

    mu = ok[len(ok) // 2]["mu"]
    print("\n" + "=" * 86)
    print(f"结构记忆：早期 s1 vs s2，后期**相同**环境，μ={mu}")
    print("=" * 86)
    a = V.make_env(1, D)
    s1 = V.make_shift(2, D, amp=2.5, a_mat=a)
    s2 = V.make_shift(13, D, amp=2.5, a_mat=a)

    print(f"  {'seed':>5} {'‖c1−c2‖':>10} {'‖w1−w2‖':>10} {'‖g1−g2‖':>10} "
          f"{'MSE':>9}")
    print("  " + "-" * 50)
    gaps = {"c": [], "w": [], "g": []}
    for sd in range(6):
        r1 = develop(sd, [(300, 3000)], mu=mu, second=s1, b_start=0, t_total=20000)
        r2 = develop(sd, [(300, 3000)], mu=mu, second=s2, b_start=0, t_total=20000)
        gc = float(np.linalg.norm(r1["c"] - r2["c"]))
        gw = float(np.linalg.norm(r1["w"] - r2["w"]))
        gg = float(np.linalg.norm(_sig(r1["c"]) - _sig(r2["c"])))
        gaps["c"].append(gc)
        gaps["w"].append(gw)
        gaps["g"].append(gg)
        print(f"  {sd:>5} {gc:>10.4f} {gw:>10.4f} {gg:>10.4f} {r1['pred_mse']:>9.5f}")

    ctrl = develop(0, [(300, 3000)], mu=0.0, second=s1, b_start=0, t_total=20000)
    ctrl2 = develop(0, [(300, 3000)], mu=0.0, second=s2, b_start=0, t_total=20000)
    gc0 = float(np.linalg.norm(ctrl["c"] - ctrl2["c"]))
    gg0 = float(np.linalg.norm(_sig(ctrl["c"]) - _sig(ctrl2["c"])))

    mc = float(np.nanmedian(gaps["c"]))
    mg = float(np.nanmedian(gaps["g"]))
    mw = float(np.nanmedian(gaps["w"]))
    print(f"\n  结构差中位 = {mc:.4f}   门控差中位 = {mg:.4f}   参数差中位 = {mw:.4f}")
    print(f"  对照 μ=0：结构差 = {gc0:.4f}，门控差 = {gg0:.4f}")

    print("\n" + "=" * 86)
    print("结论")
    print("=" * 86)
    print(f"""
1. **不再互毁**：{len(ok)}/{len(rows)-1} 组 μ 同时满足"pred MSE < 3×地板"
   与"结构仍在动"。前四次尝试做不到这一点（旋转强就毁学习），
   因为那时梯度和旋转抢同一批参数；现在它们用不同自由度。

2. w 收敛（步长 {np.median([r['w_step'] for r in ok]):.2e}），
   c 不停（步长 {np.median([r['c_step'] for r in ok]):.2e}）。
   **"学会了"与"永远在变"第一次同时成立。**

3. 结构差 {mc:.4f}（对照 {gc0:.4f}），门控差 {mg:.4f}（对照 {gg0:.4f}）。
   {'结构确实携带了早期养育信息 —— 这是 A3 的新载体。' if mc > gc0 + 1e-6 else '结构未额外携带信息。'}

仍未验证（必须说清楚）：
  - 后期**换环境**后这些差能不能被抹平（A3 的直接检验）
  - 这些差是否带来**行为后果**（否则"结构不同"只是数字不同）
""")
    return {"rows": rows,
            "memory": {"mu": mu, "gap_c": mc, "gap_g": mg, "gap_w": mw,
                       "control_c": gc0, "control_g": gg0,
                       "per_seed": gaps}}


if __name__ == "__main__":
    if not ALLOW_RUN:
        sys.exit(
            "本文件已判死（c 是规范自由度，对行为零影响，见文件头）。\n"
            "证据：c 取 5 个完全不同的值，MSE 全部 = 0.002958。\n"
            "要用请设 CHIUBAY_ALLOW_E4E=1，但结果不可用于支持任何结论。"
        )
    p = Path("runs/e4e")
    p.mkdir(parents=True, exist_ok=True)
    res = main()
    (p / "report.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {p / 'report.json'}")
