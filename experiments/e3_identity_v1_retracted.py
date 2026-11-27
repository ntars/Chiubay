"""E3：认同环能否在无全局损失下产生"早期不可覆盖"。

这一关决定两周期命题是否成立。

四次修正（全部记录，不隐藏）：
    失败 1：细胞数 = 维度 → 系统总能重新适应，印记留不下。→ 容量受限。
    失败 2：细胞预测完整状态 → 容量受限时形状不匹配。→ 改为每个细胞
            只预测"自己那一路输入"：err_i = w_i·(x_{t+1}−x_t)。
    失败 3：用结构距离当判据 → 学进去与没学进去都偏离，分不开。
            → 改用撤去扰动后的残留误差。
    失败 4：三个时间尺度同时发生 → 棘轮在学会之前就闭锁，系统"学不干净"，
            没有印记只有损伤。→ **必须分离三个时间常数**：
              τ_learn  ~ 800 步   （eta0 决定）
              τ_anchor ~ 200 步   （rho0 = 0.005，快速记住"现在的我"）
              τ_ratchet~ 2500 步  （L0，缓慢锁死）
            正确顺序：先学会 → 锚点记下 → 棘轮锁死。锁死必须晚于学会。

诚实标注：
    局部学习规则 = LMS（Widrow–Hoff 1960），标 CONVENTIONAL。
    敏感期 = 关键期/印记（Lorenz 1935；Hubel & Wiesel 1970），不新。
    本实验只检验：无全局损失时能否复现"早期不可覆盖"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_env(seed: int, d: int, radius: float = 0.85) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d)) / np.sqrt(d)
    ev = float(np.max(np.abs(np.linalg.eigvals(a))))
    return a * (radius / max(ev, 1e-9))


def make_shift(seed: int, d: int, amp: float = 1.2) -> np.ndarray:
    """环境的秩 1 改变：多出一条耦合方向。这是"养育方式"的物理形式。"""
    rng = np.random.default_rng(seed + 900)
    u, v = rng.normal(size=d), rng.normal(size=d)
    return amp * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))


def develop(seed: int, use_identity: bool, window, t_total: int = 12000,
            n_cells: int = 6, d: int = 24, eta0: float = 0.15,
            eta_min: float = 0.0005, L0: float = 60.0,
            rho0: float = 0.005, beta: float = 8.0, kappa: float = 0.01,
            noise: float = 0.05) -> dict:
    """发育。三个时间常数分离：τ_learn < τ_anchor << τ_ratchet。

    细胞 i 的局部量（不涉及任何其他细胞）：
        err_i = w_i · (x_{t+1} − x_t)
    更新：w_i += η_eff_i · gate_i · err_i · x_t / |x_t|²
    """
    rng = np.random.default_rng(seed)
    a_mat = make_env(seed + 1, d)
    shift = make_shift(seed + 1, d)

    w = rng.normal(size=(n_cells, d)) * 0.15
    anchor = w.copy()
    load = np.zeros(n_cells)
    x = rng.normal(size=d)

    eta_hist, load_hist = [], []
    for t in range(t_total):
        a_now = a_mat + shift if (window and window[0] <= t < window[1]) else a_mat
        x_next = a_now @ x + noise * rng.normal(size=d)

        err = w @ (x_next - x)           # [n_cells]，纯局部
        dev = np.mean((w - anchor) ** 2, axis=1)

        if use_identity:
            eta_eff = eta_min + (eta0 - eta_min) * np.exp(-load / L0)
            gate = np.exp(-beta * dev / (dev + kappa))
            rho = np.full(n_cells, rho0)
        else:
            eta_eff = np.full(n_cells, eta0)
            gate = np.ones(n_cells)
            rho = np.zeros(n_cells)

        scale = 1.0 / max(float(x @ x), 1e-6)
        w += (eta_eff * gate * err)[:, None] * x[None, :] * scale
        # 棘轮累积"活动量"，不是"误差"。
        # 用误差会让棘轮在学会后自动停转（err→0），永远锁不死。
        # 活动量不随学习下降，才是真正的单向棘轮（= 年龄，但不是时钟）。
        load += np.abs(w @ x)
        anchor += rho[:, None] * (w - anchor)

        x = x_next
        if t % 1000 == 0:
            eta_hist.append(float(np.mean(eta_eff)))
            load_hist.append(float(np.mean(load)))

    return {"w": w, "load": load, "eta_hist": eta_hist, "load_hist": load_hist}


def residual(w: np.ndarray, a_mat: np.ndarray, d: int, steps: int = 600,
             seed: int = 777, noise: float = 0.05) -> float:
    """在**原始**环境（扰动已撤去）里测本地预测误差。不更新权重。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=d)
    tot = 0.0
    for _ in range(steps):
        x_next = a_mat @ x + noise * rng.normal(size=d)
        tot += float(np.mean(np.abs(w @ (x_next - x))))
        x = x_next
    return tot / steps


def main(n_cells: int = 6, t_total: int = 12000) -> dict:
    d = 24
    early = (300, 1100)                       # η 仍大，能学进去
    late = (t_total - 1200, t_total - 400)    # η 已接近 eta_min，进不去

    a_mat = make_env(1, d)
    out = {}
    for use_id, tag in ((True, "identity"), (False, "no_identity")):
        base = develop(0, use_id, None, t_total, n_cells=n_cells, d=d)
        e = develop(0, use_id, early, t_total, n_cells=n_cells, d=d)
        l = develop(0, use_id, late, t_total, n_cells=n_cells, d=d)
        r = {
            "res_base": residual(base["w"], a_mat, d),
            "res_early": residual(e["w"], a_mat, d),
            "res_late": residual(l["w"], a_mat, d),
            "eta_hist": base["eta_hist"],
            "load_hist": base["load_hist"],
            "lock_frac": float((base["load"] > 2.0 * 60.0).mean()),
        }
        r["excess_early"] = r["res_early"] - r["res_base"]
        r["excess_late"] = r["res_late"] - r["res_base"]
        r["imprint"] = r["excess_early"] / max(abs(r["excess_late"]), 1e-12)
        out[tag] = r

    r0 = residual(develop(0, True, None, t_total, n_cells=n_cells, d=d)["w"], a_mat, d)
    r1 = residual(develop(23, True, None, t_total, n_cells=n_cells, d=d)["w"], a_mat, d)
    out["noise_floor"] = abs(r1 - r0)
    out["config"] = {"n_cells": n_cells, "d": d, "t_total": t_total,
                     "early": list(early), "late": list(late)}
    return out


if __name__ == "__main__":
    res = main()
    print("=" * 76)
    print("E3  认同环 → 早期不可覆盖（撤去扰动后的残留局部误差）")
    print("=" * 76)
    cfg = res["config"]
    print(f"细胞 {cfg['n_cells']} / 维度 {cfg['d']}（容量比 "
          f"{cfg['n_cells']/cfg['d']:.0%}），总步数 {cfg['t_total']}")
    print(f"时间尺度：τ_learn≈800  τ_anchor≈200  τ_ratchet≈60（必须递增）")
    for tag, label in (("identity", "有认同环"), ("no_identity", "无认同环（对照）")):
        r = res[tag]
        print(f"\n{label}")
        print(f"  基线残留        {r['res_base']:.8f}")
        print(f"  早期扰动后残留  {r['res_early']:.8f}   超出基线 {r['excess_early']:+.2e}")
        print(f"  晚期扰动后残留  {r['res_late']:.8f}   超出基线 {r['excess_late']:+.2e}")
        print(f"  印记比 早/晚    {r['imprint']:.2f}×")
        print(f"  η 轨迹   {[round(v, 4) for v in r['eta_hist']]}")
        print(f"  载荷轨迹 {[round(v, 1) for v in r['load_hist']]}")
    print(f"\n噪声底（同条件换种子、无扰动）{res['noise_floor']:.2e}")

    p = Path("runs/e3_identity")
    p.mkdir(parents=True, exist_ok=True)
    (p / "report.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n写入 {p / 'report.json'}")
