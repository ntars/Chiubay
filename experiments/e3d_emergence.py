"""E3d：A3 与 A5 能否在同一装置上同时成立？

前置：本实验的装置必须过信噪比关（`test_signal_exceeds_noise_floor`）。
      d=12、amp=2.5、gamma=2.0 时实测 S/N=1.70（信号 0.287 / 误调 0.169）。
      在 S/N<1 时，任何"印记"数字都是误调，不是信号——v1/v2/v3 全部栽在这里。

判据（**跑之前写死，不许事后改**）
---------------------------------
A3（不可逆）：同一批细胞，等剂量 4000 步，
    认同层 `late_effect / early_effect < 0.20`
A5（可生长）：认同层闭锁后，外围层对新养育 B 的印记
    `imprint_B_after > +0.15`（正值，不是"变化 > 1e-6"）
并存：同一装置的同一批细胞上，A3 与 A5 **同时**成立。

对照（缺一不可，否则结论无意义）
------------------------------
1. 纯 LMS（none）：不看收敛比、不闭锁、无 gate。若无约束系统也能给 A3，
   则 ③ 的全部机制都是冗余的。
2. 只学 B（参照上界）：同一装置、同等剂量只喂 B。A5 的绝对上界，
   若这个参照本身为负，A5 判据不成立（v2 栽在这里）。

若 A3 与 A5 无法并存，判定：**"用同一批参数的可塑性同时承担不可逆与可生长"
这一设计不成立**，③ 需重新设计，不允许靠调参绕过。

运行：.venv/bin/python experiments/e3d_emergence.py
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
TT = 30000
AMP = 2.5
# 锁定时刻实测（D_lock=0.25）：576–1291，中位 781。
# 所以"早期养育窗"必须从 300 起（含闭锁前），"晚期养育窗"必须整段
# 落在 1300 之后。旧版的早期窗 300-4300 有 3/4 落在闭锁之后，等于
# "早期"和"晚期"都在闭锁后 —— A3 测的是"容量耗尽"，不是"不可逆"。
EARLY = (300, 4300)
LATE = (TT - 4800, TT - 800)
CFG_LOCK = {"D_lock": 0.25}


def env_of(seed):
    a = V.make_env(seed + 1, D)
    sA = V.make_shift(seed + 2, D, amp=AMP, a_mat=a)
    sB = V.make_shift(seed + 13, D, amp=AMP, a_mat=a)
    return a, sA, sB


def run(seed, windows, mode="full", second=None, b_start=5000, t_total=TT, **cfg):
    merged = {**CFG_LOCK, **cfg}
    return V.develop(seed, windows, second_shift=second, b_start=b_start,
                     mode=mode, n_cells=N, d=D, t_total=t_total, cfg=merged)


def imp(w, a, s, mask=None):
    """有界印记指数。+1 完全吸收该养育，0 无偏好，-1 完全拒绝。"""
    if mask is None:
        mask = np.ones(N, dtype=bool)
    if not mask.any():
        return float("nan")
    d0 = np.linalg.norm(w[mask] - a[:N][mask], axis=1)
    d1 = np.linalg.norm(w[mask] - (a + s)[:N][mask], axis=1)
    den = d0 + d1
    return float(np.where(den > 1e-12, (d0 - d1) / np.maximum(den, 1e-12), 0.0).mean())


def snr_check(seed=0):
    """返回 (signal, noise, ratio)。signal/noise < 1 则整个实验无效。"""
    a, sA, _ = env_of(seed)
    r = run(seed, [EARLY], second=sA, b_start=0)
    sig = float(np.mean(np.linalg.norm(sA[:N], axis=1)))
    noise = float(np.mean(np.linalg.norm(r["w"] - a[:N], axis=1)))
    return sig, noise, sig / max(noise, 1e-12)


def test_a3_a5(seed, mode="full"):
    a, sA, sB = env_of(seed)
    base = run(seed, [], mode=mode)
    early = run(seed, [EARLY], mode=mode)
    late = run(seed, [LATE], mode=mode)
    # B 必须在闭锁之后才出现（5000 > 锁定中位 781），否则测的是"闭锁前能否改"
    ab = run(seed, [EARLY, (5000, TT - 500)], mode=mode, second=sB)

    m = early["identity_mask"]
    per = ~m
    de = imp(early["w"], a, sA, m) - imp(base["w"], a, sA, m)
    dl = imp(late["w"], a, sA, m) - imp(base["w"], a, sA, m)
    ratio = abs(dl / de) if abs(de) > 1e-6 else float("nan")

    # A5：外围层对 B 的印记（绝对值，不是"变化"）
    b_after = imp(ab["w"], a, sB, per)
    b_before = imp(early["w"], a, sB, per)
    # A3 污染检查：认同层对 A 的印记在见了 B 之后是否变了
    a_pollute = abs(imp(ab["w"], a, sA, m) - imp(early["w"], a, sA, m))

    return {"seed": seed, "mode": mode,
            "early_effect": float(de), "late_effect": float(dl),
            "late_over_early": float(ratio),
            "periphery_B_before": b_before, "periphery_B_after": b_after,
            "identity_pollution": float(a_pollute),
            "n_identity": int(m.sum()), "n_periphery": int(per.sum()),
            "t_lock": early["t_lock"].tolist()}


def ref_only_B(seed):
    """参照上界：同等剂量只喂 B，测外围层能到多少。"""
    a, sA, sB = env_of(seed)
    r = run(seed, [EARLY], second=sB, b_start=0)
    m = r["identity_mask"]
    return {"periphery_B": imp(r["w"], a, sB, ~m),
            "identity_B": imp(r["w"], a, sB, m)}



def lock_vs_learning(n_seeds=3):
    """闭锁时机 vs 学到程度：证明闭锁与"学会"直接冲突。

    这是 ③ 的根本矛盾，比 A3/A5 更靠前：
      - 闭锁越早，系统越早停止学习，||w-(A+B)|| 越大
      - 但闭锁越晚，"不可逆"越弱
    也就是说"A3（不可逆）"与"学会世界"在同一个参数上争夺，
    而这两个都是方案要求的东西。
    """
    rows = []
    for dl in (0.05, 0.10, 0.15, 0.25, 0.40):
        dists, imps, locks = [], [], []
        for s in range(n_seeds):
            a, sA, sB = env_of(s)
            r = run(s, [EARLY], second=sB, b_start=0, D_lock=dl)
            dists.append(float(np.mean(np.linalg.norm(r["w"] - (a + sB)[:N], axis=1))))
            imps.append(imp(r["w"], a, sB))
            locks.append(r["t_lock"])
        lt = [t for row in locks for t in row if t >= 0]
        rows.append({"D_lock": dl,
                     "dist_to_target": float(np.mean(dists)),
                     "imprint_B": float(np.mean(imps)),
                     "lock_median": int(np.median(lt)) if lt else -1})
    return rows


def main(n_seeds=5):
    sig, noise, ratio = snr_check()
    out = {"design": {"d": D, "n_cells": N, "t_total": TT, "amp": AMP,
                      "cfg": {k: v for k, v in V.CFG.items()}},
           "snr": {"signal": sig, "noise": noise, "ratio": ratio,
                   "valid": bool(ratio > 1.0)},
           "criteria": {"A3": "late/early < 0.20（认同层，等剂量）",
                        "A5": "periphery_B_after > +0.15（绝对值）",
                        "coexist": "同一装置同一批细胞上同时成立"}}

    print("=" * 78)
    print("E3d  A3 与 A5 能否并存")
    print("=" * 78)
    print(f"\n信噪比：信号 {sig:.4f} / 误调 {noise:.4f} = {ratio:.2f}  "
          f"{'有效' if ratio > 1.0 else '无效（S/N<1，结论不可用）'}")
    if ratio <= 1.0:
        print("!! 信噪比不足，本次运行不得产出任何结论")
        p = Path("runs/e3d")
        p.mkdir(parents=True, exist_ok=True)
        (p / "report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return out

    refs = [ref_only_B(s) for s in range(n_seeds)]
    out["reference_only_B"] = refs
    ref_mean = float(np.mean([r["periphery_B"] for r in refs]))
    print(f"\n参照上界（只喂 B）：外围层 B 印记 = {ref_mean:+.4f}  "
          f"{'可用' if ref_mean > 0 else '参照本身为负 → A5 判据不可用'}")

    rows = {}
    for mode in ("full", "soft", "none"):
        res = [test_a3_a5(s, mode) for s in range(n_seeds)]
        rows[mode] = res
        r3 = float(np.nanmean([x["late_over_early"] for x in res]))
        r5 = float(np.mean([x["periphery_B_after"] for x in res]))
        poll = float(np.mean([x["identity_pollution"] for x in res]))
        a3 = r3 < 0.20
        a5 = r5 > 0.15
        out[f"results_{mode}"] = {"per_seed": res,
                                  "late_over_early_mean": r3,
                                  "periphery_B_mean": r5,
                                  "identity_pollution_mean": poll,
                                  "A3_pass": bool(a3), "A5_pass": bool(a5),
                                  "coexist": bool(a3 and a5)}
        print(f"\n{mode:>5s}  A3 晚/早={r3:+.4f} ({'过' if a3 else '不过'})   "
              f"A5 外围B={r5:+.4f} ({'过' if a5 else '不过'})   "
              f"认同层污染={poll:+.4f}   并存={'是' if (a3 and a5) else '否'}")

    coexist_any = any(out[f"results_{m}"]["coexist"] for m in ("full", "soft"))
    out["coexist_any"] = bool(coexist_any)
    out["verdict"] = ("A3 与 A5 可并存于同一装置" if coexist_any
                      else "A3 与 A5 无法并存 → 「同一批参数的可塑性同时承担"
                           "不可逆与可生长」这一设计不成立，③ 需重新设计")
    print(f"\n判定：{out['verdict']}")

    p = Path("runs/e3d")
    p.mkdir(parents=True, exist_ok=True)
    (p / "report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n写入 {p / 'report.json'}")
    return out


if __name__ == "__main__":
    main()
