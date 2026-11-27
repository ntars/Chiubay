"""E3 v3 —— **结论作废（2026-11-27 晚），被 e3d_emergence.py 取代**

RETRACTED
=========
本文件的装置（e3v3_core 的预测式 LMS）现在是对的，但本文件的**判据和
参数标定是坏的**，三个原因：

  1. T0 用 cfg={"D_lock": 1e9} 试图"关掉闭锁"，但判据是 conv <= thr，
     thr 变大会让所有细胞在第 400 步立刻闭锁（实测 frozen=24、η=0），
     于是 T0_learns 报 False —— 这个 False 是自伤的，不是装置坏。
  2. T1/T2 用的 D_lock/L_load 是旧参数名，e3v3_core 已改为 conv 驱动的
     D_lock(收敛比) + gamma，本文件的 CFG 已不对应。
  3. 最关键：本文件没有测信噪比。实测噪声底（0.24–0.35）大于养育信号
     （0.03–0.25），所以 T1/T2/T4 的所有数字都不能用。

替代：experiments/e3d_emergence.py（复用 core，不写第二份 develop）。
本文件保留作为"判据设计错误"的记录，报告 report_v3.json 不再作为证据引用。

---

原 v3 说明（历史记录）：

E3 v3：认同环 —— 预测式学习下的"早期不可覆盖"。

本版为什么不是 v2 的补丁
------------------------
v2 的学习规则是 Δw = η·(w·dx)·x/|x|²，即让 w·dx → 0。
但 dx 张满 R^d、(A-I) 可逆（实测最小奇异值 0.2601），
该规则唯一不动点是 w = 0。实测 20000 步后 ||w|| = 2.4e-28（初始 1.728）。
所以 v1/v2 测到的"印记"是权重向零崩塌途中的残留量，不是学到的内容。

v3 改为预测式：err_i = x_next[i] - w_i·x，不动点是 w_i = A[i]（环境第 i 行）。
这才是"学会世界"，也才使得"养育写进权重"有处可写。

  指标也换：v2 用 M=mean|w·u|，无上界、跨条件不可比，且 w→0 时天然趋零。
  v3 用有界指数 imprint=(d0-d1)/(d0+d1)，d0=||w-A[i]||，d1=||w-(A+S)[i]||。
  +1 完全吸收养育，0 无偏好，-1 反向。有界、无量纲、跨条件可比。

三个已被证伪的旧结论，本版重测
------------------------------
1. v2 说"gate 单独就够、分层冗余" —— 那是 use_layer 的 bug（frozen_eta
   误含分层）。E3c 修正后 6 类环境 5 类分离，棘轮必要。
2. v2 说"软阻抗 60k 后仍保住印记" —— 同样受污染，且 w→0 使该结论无意义。
3. v2 说"外围层还能学（A5 活着）" —— 实测外围层 eta_eff 也是 1.01e-4，
   因为 load 是全局标量。分层从未实现。v3 改为 load_i 逐细胞。

运行：.venv/bin/python experiments/e3_identity_v3.py
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

D = 24
N = 6
TT = 14000

# 参数名已对齐 e3v3_core（L_load 废弃，改为 conv 驱动的 gamma/ema）。
# 但**取值仍是旧标定的**：D_lock=1.55 会让 conv <= thr 恒真
# （conv 被 clip 到 [0,1]），于是所有细胞在第 lock_delay 步一次性全锁。
# 这就是 banner 第 1 条说的"自伤"。保留旧值只为让文件能跑出
# 与 report_v3.json 一致的历史轨迹；不要用这些值做新实验。
CFG = {"gamma": 2.0, "ema": 0.003, "ref_steps": 300, "D_lock": 1.55}

ALLOW_RUN = os.environ.get("CHIUBAY_ALLOW_RETRACTED") == "1"


def env_of(seed):
    a = V.make_env(seed + 1, D)
    s = V.make_shift(seed + 2, D, a_mat=a)
    sB = V.make_shift(seed + 13, D, a_mat=a)
    return a, s, sB


def imprint_vec(w, a, s):
    """逐细胞印记指数。"""
    d0 = np.linalg.norm(w - a[:N], axis=1)
    d1 = np.linalg.norm(w - (a + s)[:N], axis=1)
    den = d0 + d1
    return np.where(den > 1e-12, (d0 - d1) / np.maximum(den, 1e-12), 0.0)


def run(seed, windows, mode="full", second=None, t_total=TT, **kw):
    return V.develop(seed, windows, second_shift=second, mode=mode,
                     cfg={**CFG, **kw}, n_cells=N, d=D, t_total=t_total)


def boot_ci(vals, n_boot=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    m = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return [float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))]


def main(n_seeds=5):
    out = {"version": "v3", "design": "预测式LMS + 逐细胞棘轮 + 认同层硬锁",
           "cfg": {**V.CFG, **CFG}, "metric": "有界印记指数 (d0-d1)/(d0+d1)"}

    # --- T0 装置自检：学习是否真的在学（不是塌向零）
    a0, s0, _ = env_of(0)
    r = V.develop(0, [(300, 4300)], cfg={**CFG, "D_lock": 1e9}, n_cells=D, d=D)
    out["T0_sanity"] = {
        "pred_mse": r["pred_mse"],
        "noise_floor": V.CFG["noise"] ** 2,
        "rel_err_to_A": float(np.linalg.norm(r["w"] - r["a_mat"]) / np.linalg.norm(r["a_mat"])),
        "weight_norm": float(np.linalg.norm(r["w"])),
        "note": "若 rel_err_to_A 在噪声地板附近则可学；v2 在同样时长下 ||w||→0",
    }

    # --- T1 关键期：早期 vs 晚期（等剂量，各 4000 步）
    rows = []
    for sd in range(n_seeds):
        a, s, _ = env_of(sd)
        b = imprint_vec(run(sd, [])["w"], a, s)
        e = imprint_vec(run(sd, [(300, 4300)])["w"], a, s)
        l = imprint_vec(run(sd, [(TT - 4800, TT - 800)])["w"], a, s)
        de, dl = (e - b).mean(), (l - b).mean()
        rows.append({"seed": sd, "base": float(b.mean()), "early": float(de),
                     "late": float(dl),
                     "late_over_early": float(dl / de) if abs(de) > 1e-9 else float("nan"),
                     "t_lock": run(sd, [])["t_lock"].tolist()})
    de = np.array([abs(r_["early"]) for r_ in rows])
    dl = np.array([abs(r_["late"]) for r_ in rows])
    out["T1_equal_dose"] = {
        "per_seed": rows,
        "early_abs_mean": float(de.mean()), "late_abs_mean": float(dl.mean()),
        "late_over_early": float(dl.mean() / de.mean()),
        "A3_pass": bool(dl.mean() / de.mean() < 0.20),
    }

    # --- T2 纵向（A3 正字面）：养育 A 后换养育 B，A 的印记还在不在
    t2 = []
    for sd in range(n_seeds):
        a, s, sB = env_of(sd)
        b = imprint_vec(run(sd, [])["w"], a, s)
        pA = imprint_vec(run(sd, [(300, 4300)])["w"], a, s)
        pAB = imprint_vec(run(sd, [(300, 4300), (5000, TT - 500)], second=sB)["w"], a, s)
        eA, eAB = (pA - b).mean(), (pAB - b).mean()
        # B 的印记：检验 A 之后能否写入 B
        impB = imprint_vec(run(sd, [(300, 4300), (5000, TT - 500)], second=sB)["w"],
                           a, sB).mean()
        t2.append({"seed": sd, "imprint_A": float(eA), "imprint_A_after_B": float(eAB),
                   "keep": float(eAB / eA) if abs(eA) > 1e-9 else float("nan"),
                   "imprint_B_after": float(impB)})
    keeps = [r_["keep"] for r_ in t2 if np.isfinite(r_["keep"])]
    out["T2_imprint_survives_new_nurture"] = {
        "per_seed": t2, "keep_mean": float(np.mean(keeps)),
        "keep_ci95": boot_ci(keeps), "A3_literal_pass": bool(np.mean(keeps) >= 0.8),
    }

    # --- T3 A5：认同层锁死后，外围层能否学新方向 B
    t3 = {}
    for sd in (0, 1, 2):
        a, s, sB = env_of(sd)
        rA = run(sd, [(300, 4300)])
        rAB = run(sd, [(300, 4300), (5000, TT - 500)], second=sB)
        idm = rA["identity_mask"]
        per = ~idm
        impB_A = imprint_vec(rA["w"], a, sB)[per].mean()
        impB_AB = imprint_vec(rAB["w"], a, sB)[per].mean()
        impA_A = imprint_vec(rA["w"], a, s)[idm].mean()
        impA_AB = imprint_vec(rAB["w"], a, s)[idm].mean()
        t3[f"seed{sd}"] = {
            "n_identity": int(idm.sum()), "n_periphery": int(per.sum()),
            "identity_imprintA_before": float(impA_A),
            "identity_imprintA_after": float(impA_AB),
            "periphery_imprintB_before": float(impB_A),
            "periphery_imprintB_after": float(impB_AB),
            "A3_identity_held": bool(abs(impA_AB - impA_A) < 0.05),
            "A5_periphery_learned": bool(impB_AB > impB_A + 0.05),
        }
    out["T3_A5_layered"] = t3
    out["T3_A5_pass"] = bool(all(v["A5_periphery_learned"] for v in t3.values()))
    out["T3_A3_held"] = bool(all(v["A3_identity_held"] for v in t3.values()))

    # --- T4 消融
    t4 = {}
    for mode in ("full", "no_gate", "eta_only", "soft", "none"):
        e = imprint_vec(run(0, [(300, 4300)], mode=mode)["w"], *env_of(0)[:2]).mean()
        l = imprint_vec(run(0, [(TT - 4800, TT - 800)], mode=mode)["w"], *env_of(0)[:2]).mean()
        b = imprint_vec(run(0, [], mode=mode)["w"], *env_of(0)[:2]).mean()
        t4[mode] = {"early": float(e - b), "late": float(l - b),
                    "late_over_early": float((l - b) / (e - b)) if abs(e - b) > 1e-9 else None,
                    "t_lock": run(0, [], mode=mode)["t_lock"].tolist()}
    out["T4_ablation"] = t4

    # --- T5 剂量 vs 时钟（棘轮驱动量的必要性）
    t5 = {}
    for sname, kw in (("stationary", {}), ("quiet_0_3000", {})):
        row = {}
        for drv in ("dose", "clock"):
            cfg = {**CFG}
            if drv == "clock":
                cfg = {**CFG, "L_load": CFG["L_load"]}
            e = imprint_vec(run(0, [(300, 4300)], **kw)["w"], *env_of(0)[:2]).mean()
            row[drv] = {"early": float(e)}
        t5[sname] = row
    out["T5_dose_vs_clock"] = t5

    # --- T6 关键期扫描（不同起始时刻，各 800 步等剂量）
    out["T6_critical_scan"] = {}
    a, s, _ = env_of(0)
    base = imprint_vec(run(0, [])["w"], a, s).mean()
    for t0 in (0, 300, 800, 1500, 2500, 3500, 5000, 7000, 9000, 11000, 12500):
        v = imprint_vec(run(0, [(t0, t0 + 800)])["w"], a, s).mean() - base
        out["T6_critical_scan"][str(t0)] = float(v)

    out["summary"] = {
        "T0_learns": out["T0_sanity"]["rel_err_to_A"] < 0.15,
        "A3_equal_dose": out["T1_equal_dose"]["A3_pass"],
        "A3_literal": out["T2_imprint_survives_new_nurture"]["A3_literal_pass"],
        "A5": out["T3_A5_pass"],
        "honest": (
            "v3 换了学习装置：v1/v2 的规则不动点是 w=0（已证实 ||w||→2.4e-28），"
            "测到的是崩塌残余；v3 用预测式 LMS，不动点是环境行 A[i]，"
            "才有'学会世界'可言。指标也换成有界指数，v2 的 mean|w·u| 无上界且随 w→0 趋零。"
            "本版仍未测：结构距离定义、死亡判据、'环'的反向边。"
        ),
    }
    return out


if __name__ == "__main__":
    if not ALLOW_RUN:
        sys.exit(
            "本文件已 RETRACTED（见文件头）。它的判据与参数标定是坏的：\n"
            "  - T0 的 D_lock=1e9 让所有细胞在第 400 步全锁，T0_learns=False 是自伤的；\n"
            "  - mean|w·u| 无上界、跨条件不可比；\n"
            "  - 未测信噪比（当时噪声底 > 信号）。\n"
            "替代实现：experiments/e3d_emergence.py。\n"
            "若确要复现历史轨迹，设 CHIUBAY_ALLOW_RETRACTED=1。"
        )
    res = main()
    W = 78
    print("=" * W)
    print("E3 v3  认同环（预测式学习装置）")
    print("=" * W)

    t0 = res["T0_sanity"]
    print(f"\nT0  装置自检")
    print(f"    预测MSE {t0['pred_mse']:.5f}  噪声地板 {t0['noise_floor']:.5f}")
    print(f"    ||W-A||/||A|| = {t0['rel_err_to_A']*100:.2f}%   ||w||={t0['weight_norm']:.3f}")
    print(f"    -> 学习{'有效' if res['summary']['T0_learns'] else '无效'}")

    t1 = res["T1_equal_dose"]
    print(f"\nT1  等剂量（各 4000 步）—— A3 主判据")
    for r_ in t1["per_seed"]:
        print(f"    seed {r_['seed']}  基线 {r_['base']:+.4f}  早 {r_['early']:+.4f}  "
              f"晚 {r_['late']:+.4f}  晚/早 {r_['late_over_early']:+.4f}")
    print(f"    |早| {t1['early_abs_mean']:.4f}  |晚| {t1['late_abs_mean']:.4f}  "
          f"晚/早 {t1['late_over_early']:.4f}  A3 "
          f"{'通过' if t1['A3_pass'] else '不通过'}")

    t2 = res["T2_imprint_survives_new_nurture"]
    print(f"\nT2  印记穿越新养育（A3 正字面）")
    for r_ in t2["per_seed"]:
        print(f"    seed {r_['seed']}  仅A {r_['imprint_A']:+.4f}  A后B {r_['imprint_A_after_B']:+.4f}  "
              f"保留 {r_['keep']:+.3f}  B印记 {r_['imprint_B_after']:+.4f}")
    ci = t2["keep_ci95"]
    print(f"    保留 {t2['keep_mean']:.3f} 95%CI [{ci[0]:.3f},{ci[1]:.3f}]  "
          f"{'通过' if t2['A3_literal_pass'] else '不通过'}")

    print(f"\nT3  A3 与 A5 能否并存（分层）")
    for k, v in res["T3_A5_layered"].items():
        print(f"    {k}  认同{v['n_identity']}/外围{v['n_periphery']}  "
              f"认同层A印记 {v['identity_imprintA_before']:+.4f}→{v['identity_imprintA_after']:+.4f}"
              f"({v['A3_identity_held']})  "
              f"外围层B印记 {v['periphery_imprintB_before']:+.4f}→{v['periphery_imprintB_after']:+.4f}"
              f"({v['A5_periphery_learned']})")

    t4 = res["T4_ablation"]
    print(f"\nT4  消融")
    for m, v in t4.items():
        loe = v["late_over_early"]
        print(f"    {m:>9s}  早 {v['early']:+.4f}  晚 {v['late']:+.4f}  "
              f"晚/早 {f'{loe:+.4f}' if loe is not None else '   n/a'}")

    print(f"\nT6  关键期扫描（800 步窗口）")
    for k, v in res["T6_critical_scan"].items():
        bar = "#" * int(max(0, v) * 100)
        print(f"    t0={int(k):5d}  {v:+.4f}  {bar}")

    print(f"\n结论")
    for k, v in res["summary"].items():
        print(f"    {k}: {v}")

    p = Path("runs/e3_identity")
    p.mkdir(parents=True, exist_ok=True)
    (p / "report_v3.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"\n写入 {p / 'report_v3.json'}")
