"""E3c：载荷棘轮是否必要？—— **结论不可引用（2026-11-27）**

RETRACTED
=========
本实验的全部对照都建立在 e3_identity.develop_custom 上，
而那个学习规则的不动点是 w=0（见 e3_identity.py 顶部横幅）。
装置无效，则"六类环境五类分离"这个结论无效。

保留下来的价值是两条**方法**结论，它们与装置无关：
  1. 跨机制比 "late/early" 不公平：full 的 de 比 soft_only 大 78 倍，
     分母不同不可比。正确判据是同一机制下的绝对值 + 噪声地板参照
     （full 晚期效应仅为噪声的 5.9 倍 = 不可测；soft=None 为 6552 倍 = 可测）。
  2. 环境设计必须有"有效性"判定：若早期增量精确为 0，说明养育
     根本没进入系统（C3_slow_env 就是这种情形），此时比值不是"晚期成功"，
     而是"两件事都没发生"，必须显式判为无效而非混进均值。

替代实验：experiments/e3d_emergence.py。

---

原说明（历史记录）：

E3 v2 的消融（T6）测出反例——
    η 恒定 0.30 + 只有 gate 阻抗（无 η 衰减、无硬冻结）→ 晚/早 = 0.0000
也就是说"固定学习率 + 偏离惩罚"就够了，不需要"载荷驱动学习率"这一层。
若这是普遍现象，③ 认同环不能自称创新，只能叫"局部偏离阻抗"。

本实验的任务是**证伪自己**：尽力找场景让「固定 η + gate」失败。
判据在跑之前写死（见 docs/09 §7）：

    找到环境 E，使
        固定η+gate  : 晚/早 ≥ 0.20
        载荷棘轮+gate: 晚/早 < 0.20
    即"必须有棘轮才能不可逆"。

若 5 类环境都找不到，判定棘轮冗余，③ 改名。
无论结果如何都如实写入报告——这是一个**可以失败**的实验。

五类候选环境（都需让"晚期还有学习动力"，否则固定 η 也会过）：
    C1 大容量      细胞数 = 维度（8/8）：系统总能重新适应
    C2 高噪声      noise=0.40：偏离被噪声掩盖，gate 挡不住
    C3 慢速环境    谱半径 0.99：信号弱、需要长时间才能学
    C4 强扰动      amp 拉满：单次偏离大，gate 的 exp(-β·d/(d+κ)) 迅速饱和
    C5 交替养育    A/B 每 2000 步轮换：持续有"新东西"

运行：.venv/bin/python experiments/e3c_ratchet_necessity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3_identity as E


def build_env(kind: str, seed: int = 1, d: int = 24):
    """返回 (a_mat, shift, n_cells, noise, extra)。

    extra 可含 "second_shift"（A/B 交替的 B 扰动）与 "period"（交替周期）。
    """
    if kind == "baseline":
        a = E.make_env(seed, d)
        return a, E.make_shift(seed + 1, d, a_mat=a), 6, 0.05, {}
    if kind == "C1_large_capacity":
        a = E.make_env(seed, d)
        return a, E.make_shift(seed + 1, d, a_mat=a), d, 0.05, {}
    if kind == "C2_high_noise":
        a = E.make_env(seed, d)
        return a, E.make_shift(seed + 1, d, a_mat=a), 6, 0.40, {}
    if kind == "C3_weak_signal":
        # C3 原设"慢环境 = 谱半径 0.99"，那是不可行的设计：
        # make_shift 要求 spec(a) < max_radius(0.88)，radius≥0.88 时扰动
        # 会被收缩到零，整类环境静默失效（de 精确为 0，报 nan）。
        # "慢"的正确物理形式不是"环境自身快发散"，而是"扰动弱、需长期才学得进"。
        # 这里保持基准环境，把扰动幅度降到 0.30（谱半径约束内可解析），
        # 于是每步信号只有基准的 1/4，但确实存在、确实可学。
        a = E.make_env(seed, d)
        return a, E.make_shift(seed + 1, d, amp=0.30, a_mat=a), 6, 0.05, {}
    if kind == "C4_strong_shift":
        # 旧版写 s*1.6 —— 绕过了谱半径约束，可能把 spec(a+s) 推到 1 以上，
        # 长程跑出 nan。改为把放大做进 amp，让约束在构造期生效。
        a = E.make_env(seed, d)
        return a, E.make_shift(seed + 1, d, amp=1.6, a_mat=a), 6, 0.05, {}
    if kind == "C5_alternate":
        # 持续有新东西：A/B 每 2000 步轮换。这是"固定 η 应该失败"的最强场景——
        # 系统永远处于"刚换环境、还没学会"的状态，晚期学习动力不可能衰减。
        a = E.make_env(seed, d)
        sA = E.make_shift(seed + 1, d, a_mat=a)
        sB = E.make_shift(seed + 13, d, a_mat=a)
        return a, sA, 6, 0.05, {"second_shift": sB, "period": 2000}
    raise ValueError(kind)


def develop_custom(seed, windows, a_mat, shift, n_cells, noise, t_total,
                   mode, d=24, second_shift_matrix=None, alternate=None):
    """独立实现，便于传自定义环境。逻辑与 E3.develop 一致。

    alternate=(another_shift, period)：养育窗口内按 period 步在 shift 与
    another_shift 之间轮换，用于 C5。
    """
    cfg = dict(E.CFG)
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n_cells, d)) * 0.15
    anchor = w.copy()
    frozen = np.zeros(n_cells, dtype=bool)
    thr = cfg["D_freeze"] * (1.0 + cfg["spread"] * rng.uniform(-1, 1, size=n_cells))
    order = np.argsort(thr)
    n_id = max(1, int(round(cfg["identity_frac"] * n_cells)))
    id_mask = np.zeros(n_cells, dtype=bool)
    id_mask[order[:n_id]] = True
    load = 0.0
    x = rng.normal(size=d)
    eta0, eta_min = cfg["eta0"], cfg["eta_min"]

    # 模式定义（必须严格区分，否则会得出错误结论）：
    #   full       η载荷衰减 + 锚点钉住 + gate + 分层硬冻结（棘轮在）
    #   soft_only  η恒定 + 锚点钉住 + gate，**不硬冻结**（棘轮不在）
    #   none       纯 LMS
    use_gate = mode in ("full", "soft_only")
    use_pin = mode != "none"
    use_decay = mode == "full"
    use_layer = mode in ("full", "no_gate")   # 硬冻结只属于棘轮模式

    for t in range(t_total):
        on = any(a <= t < b for a, b in windows)
        a_now = a_mat
        if on:
            if alternate is not None:
                a_now = a_mat + (shift if (t // alternate[1]) % 2 == 0 else alternate[0])
            elif second_shift_matrix is not None:
                a_now = a_mat + second_shift_matrix
            else:
                a_now = a_mat + shift
        xn = a_now @ x + noise * rng.normal(size=d)
        dx = xn - x
        err = w @ dx
        dev = np.mean((w - anchor) ** 2, axis=1)

        if use_decay:
            eta_eff = eta_min + (eta0 - eta_min) * np.exp(-load / cfg["L_load"])
        else:
            eta_eff = np.full(n_cells, eta0)
        gate = np.where(frozen, np.exp(-cfg["beta"] * dev / (dev + cfg["kappa"])), 1.0) \
            if use_gate else np.ones(n_cells)

        if use_layer:
            eta_eff = np.where(frozen & id_mask, 0.0, eta_eff)
            gate = np.where(id_mask, gate, 1.0)

        w = w + (eta_eff * gate * err)[:, None] * x[None, :] / max(float(x @ x), 1e-6)
        load += float(np.mean(np.abs(dx)))
        if use_pin:
            newly = (~frozen) & (load >= thr)
            if newly.any():
                anchor[newly] = w[newly]
                frozen |= newly
        x = xn
    return w


def measure(kind, mode, n_seeds=3):
    """返回晚期/早期比值（投影指标）。

    额外返回诊断量：若 de 精确为 0，说明"养育"根本没进入系统
    （扰动为零 / 窗口未生效 / 学习率被稀释），此时比值不是"晚期成功"，
    而是"两件事都没发生"。必须让它显式失败，不能混进均值。
    """
    d = 24
    a_mat, shift, n_cells, noise, extra = build_env(kind)
    second = extra.get("second_shift")
    period = extra.get("period")
    alt = (second, period) if second is not None else None
    u = np.linalg.svd(shift)[0][:, 0]
    TT = 14000
    ratios, diag = [], []
    for s in range(n_seeds):
        def M(w):
            return float(np.mean(np.abs(w @ u)))
        b = M(develop_custom(s, [], a_mat, shift, n_cells, noise, TT, mode))
        e = M(develop_custom(s, [(300, 4300)], a_mat, shift, n_cells, noise, TT,
                             mode, alternate=alt))
        l = M(develop_custom(s, [(TT - 4800, TT - 800)], a_mat, shift, n_cells, noise,
                             TT, mode, alternate=alt))
        de, dl = e - b, l - b
        diag.append({"seed": s, "de": float(de), "dl": float(dl)})
        ratios.append(abs(dl / de) if abs(de) > 1e-6 else float("nan"))
    return ratios, diag


def main():
    kinds = ["baseline", "C1_large_capacity", "C2_high_noise",
             "C3_weak_signal", "C4_strong_shift", "C5_alternate"]
    out = {"design": "找场景使 固定η+gate 晚/早 >=0.20 而 载荷棘轮+gate <0.20",
           "prediction": "若棘轮真必要，至少一类环境应出现分离",
           "invalid_environments": [],
           "results": {}}

    print(f"{'环境':20s} {'soft_only(无棘轮)':>18s} {'full(棘轮+gate)':>17s} {'分离?':>7s}")
    any_sep = False
    for k in kinds:
        r_fixed, d_fixed = measure(k, "soft_only")
        r_full, d_full = measure(k, "full")
        m_fixed = float(np.nanmean(r_fixed))
        m_full = float(np.nanmean(r_full))
        valid = bool(np.all(np.isfinite(r_fixed)) and np.all(np.isfinite(r_full)))
        sep = valid and (m_fixed >= 0.20) and (m_full < 0.20)
        any_sep = any_sep or sep
        if not valid:
            out["invalid_environments"].append(k)
        out["results"][k] = {
            "soft_only_ratios": r_fixed, "ratchet_ratios": r_full,
            "soft_only_mean": m_fixed, "ratchet_mean": m_full,
            "separated": bool(sep), "valid": valid,
            "diag_soft": d_fixed, "diag_full": d_full,
        }
        mark = "" if valid else "  <- 无效（早期增量为0，环境未进入系统）"
        print(f"{k:20s} {m_fixed:18.4f} {m_full:17.4f} {str(sep):>7s}{mark}")

    n_valid = len(kinds) - len(out["invalid_environments"])
    out["n_valid_environments"] = n_valid
    out["any_environment_separates"] = bool(any_sep)
    out["verdict"] = ("棘轮必要，③ 可保留机制主张" if any_sep
                      else "在全部有效环境上，棘轮未显示必要性；③ 需重新设计对照")
    print(f"\n有效环境 {n_valid}/{len(kinds)}")
    print(f"判定：{out['verdict']}")

    p = Path("runs/e3c_ratchet")
    p.mkdir(parents=True, exist_ok=True)
    (p / "report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"写入 {p / 'report.json'}")


if __name__ == "__main__":
    main()
