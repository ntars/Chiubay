"""M2 实验：S0 网格房端到端 + 5 seeds 复现 + 父母塑形消融。

运行：.venv/bin/python experiments/m2_s0.py
产出：runs/m2_s0/report.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.child.gridworld import GridWorld
from src.child.lse import LSELayer
from src.child.trainer import REINFORCETrainer

SIZE, OBSTACLES, MAX_STEPS = 8, 3, 80
N_EXPERTS, EXPERT_DIM, TOP_K = 8, 16, 2
EPISODES, LR, GAMMA = 500, 1e-3, 0.99
SEEDS = [0, 1, 2, 3, 4]


def run(seed: int, shape_coef: float) -> dict:
    model = LSELayer(dim=16, out_dim=5, n_experts=N_EXPERTS,
                     expert_dim=EXPERT_DIM, top_k=TOP_K,
                     use_bridge=False, seed=seed)
    env = GridWorld(size=SIZE, n_obstacles=OBSTACLES,
                    max_steps=MAX_STEPS, seed=seed)
    trainer = REINFORCETrainer(model, env, lr=LR, gamma=GAMMA,
                               shape_coef=shape_coef, seed=seed)
    report = trainer.train(episodes=EPISODES, log_every=100, eval_every=200)
    report["seed"] = seed
    report["shape_coef"] = shape_coef
    report["params"] = model.n_parameters()
    return report


def iqm(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    k = max(1, len(v) // 4)
    return float(v[:k].mean())


def summarize(name: str, reports: list) -> dict:
    ev = [r["final_eval"] for r in reports]
    keys = ev[0].keys()
    out = {
        "name": name,
        "n_seeds": len(reports),
        "episodes_each": EPISODES,
        "steps_per_sec": round(float(np.mean([r["steps_per_sec"] for r in reports])), 1),
        "params": reports[0]["params"],
    }
    for k in keys:
        vals = [e[k] for e in ev]
        out[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "iqm": round(iqm(vals), 4),
        }
    out["seeds"] = [r["seed"] for r in reports]
    out["learning_curves"] = [
        [h.get("episode_return") for h in r["history"]] for r in reports
    ]
    return out


def main() -> int:
    print(f"S0: {SIZE}×{SIZE}, {OBSTACLES} 障碍, {N_EXPERTS} 专家, "
          f"{EPISODES} episodes × {len(SEEDS)} seeds")

    shaped = [run(s, 0.3) for s in SEEDS]
    plain = [run(s, 0.0) for s in SEEDS[:3]]

    a = summarize("with_parent_shaping", shaped)
    b = summarize("no_shaping", plain)

    print(f"\n{'配置':<22}{'回报均值':>10}{'成功率':>9}{'步数':>8}"
          f"{'MI(bit)':>9}{'增益':>8}{'死专家':>8}{'steps/s':>10}")
    print("-" * 84)
    for r in (a, b):
        print(f"{r['name']:<22}"
              f"{r['mean_return']['mean']:>10.3f}"
              f"{r['success_rate']['mean']:>9.3f}"
              f"{r['mean_length']['mean']:>8.1f}"
              f"{r['mutual_info_bits']['mean']:>9.4f}"
              f"{r['routing_gain']['mean']:>8.4f}"
              f"{r['dead_experts']['mean']:>8.1f}"
              f"{r['steps_per_sec']:>10.1f}")

    print("\nwith_parent_shaping 各 seed 回报:")
    for r in shaped:
        print(f"  seed {r['seed']}: mean_return={r['final_eval']['mean_return']:.3f} "
              f"success={r['final_eval']['success_rate']:.2f} "
              f"steps/s={r['steps_per_sec']:.0f}")

    out = ROOT / "runs" / "m2_s0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps({"shaped": a, "plain": b}, ensure_ascii=False, indent=2))
    print(f"\n报告写入 {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
