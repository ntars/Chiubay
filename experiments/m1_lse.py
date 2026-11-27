"""M1 实验：LSE 分化能力、联动桥消融、Top-K 门控梯度、吞吐。

运行：.venv/bin/python experiments/m1_lse.py
产出：runs/m1_lse/report.json + 终端表格
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.child.lse import LSELayer, build_default_lse
from src.child.metrics import summarize_route
from src.runtime.losses import cross_entropy
from src.runtime.optim import Adam
from src.runtime.tensor import Tensor

DIM, N_CLASSES, N_EXPERTS, EXPERT_DIM = 12, 6, 8, 16
STEPS, BATCH, LR = 1500, 64, 0.01


def make_task(seed=0, n_per_class=300, spread=0.9, noise=1.0):
    """3 个高斯簇；spread 越小越难分，用来暴露初始化偏差。"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, spread, size=(N_CLASSES, DIM))
    labels = np.repeat(np.arange(N_CLASSES), n_per_class)
    x = centers[labels] + rng.normal(0.0, noise, size=(labels.size, DIM))
    return x, labels


def evaluate(model, x, labels, top_k):
    _, stats = model(Tensor(x), return_stats=True)
    metrics = summarize_route(stats["gate_probs"], labels, top_k=top_k)
    out = model(Tensor(x))
    pred = out.data.argmax(axis=1)
    metrics["accuracy"] = float((pred == labels).mean())
    return metrics


def run(name, cfg, x, labels, steps=STEPS, seed=1, record_curve=False):
    model = LSELayer(dim=DIM, out_dim=N_CLASSES, n_experts=N_EXPERTS,
                     expert_dim=EXPERT_DIM, top_k=cfg.pop("top_k", 2),
                     bridge_dim=EXPERT_DIM, seed=seed, **cfg)
    opt = Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(1234)
    gate_before = model.gate.weight.data.copy()
    history = []
    start = time.perf_counter()

    for step in range(1, steps + 1):
        idx = rng.integers(0, labels.size, BATCH)
        opt.zero_grad()
        loss = cross_entropy(model(Tensor(x[idx])), labels[idx])
        loss.backward()
        opt.step()
        if record_curve and step % 250 == 0:
            m = evaluate(model, x, labels, model.top_k)
            history.append({"step": step, "loss": float(loss.item()),
                            "mi": m["mutual_info_bits"],
                            "consistency": m["routing_consistency"],
                            "acc": m["accuracy"]})

    elapsed = time.perf_counter() - start
    final = evaluate(model, x, labels, model.top_k)
    gate_drift = float(np.abs(model.gate.weight.data - gate_before).mean())

    result = {
        "name": name,
        "steps": steps,
        "batch": BATCH,
        "params": model.n_parameters(),
        "steps_per_sec": round(steps / elapsed, 1),
        "final_loss": round(float(loss.item()), 4),
        "accuracy": round(final["accuracy"], 4),
        "mutual_info_bits": round(final["mutual_info_bits"], 4),
        "routing_consistency": round(final["routing_consistency"], 4),
        "routing_gain": round(final["routing_gain"], 4),
        "entropy_over_k": round(final["entropy_mean"], 4),
        "dead_experts": final["dead_experts"],
        "gate_drift": round(gate_drift, 6),
        "history": history,
    }
    return result


def main() -> int:
    x, labels = make_task()
    print(f"任务: {labels.size} 样本 / {N_CLASSES} 类 / {DIM} 维；"
          f"LSE: {N_EXPERTS} 专家 × {EXPERT_DIM}")

    baseline = evaluate(
        LSELayer(dim=DIM, out_dim=N_CLASSES, n_experts=N_EXPERTS,
                 expert_dim=EXPERT_DIM, top_k=2, bridge_dim=EXPERT_DIM, seed=1),
        x, labels, 2)
    print(f"初始化: MI={baseline['mutual_info_bits']:.3f}bit "
          f"consistency={baseline['routing_consistency']:.3f} "
          f"acc={baseline['accuracy']:.3f}\n")

    runs = [
        run("full", dict(use_bridge=True, use_gate_scaled_qk=True,
                         mask_inactive=True), x, labels, record_curve=True),
        run("no_bridge", dict(use_bridge=False), x, labels),
        run("bridge_no_scale", dict(use_bridge=True, use_gate_scaled_qk=False),
            x, labels),
        run("bridge_no_mask", dict(use_bridge=True, mask_inactive=False),
            x, labels),
        run("top1", dict(use_bridge=True, top_k=1), x, labels),
    ]

    header = f"{'配置':<16}{'loss':>8}{'acc':>7}{'MI(bit)':>9}{'一致':>7}{'增益':>7}{'H/logk':>9}{'死专家':>7}{'门控漂移':>10}{'steps/s':>9}"
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['name']:<16}{r['final_loss']:>8.4f}{r['accuracy']:>7.3f}"
              f"{r['mutual_info_bits']:>9.3f}{r['routing_consistency']:>7.3f}"
              f"{r['routing_gain']:>7.3f}"
              f"{r['entropy_over_k']:>9.3f}{r['dead_experts']:>7d}"
              f"{r['gate_drift']:>10.6f}{r['steps_per_sec']:>9.1f}")

    print("\nfull 曲线:")
    for h in runs[0]["history"]:
        print(f"  step {h['step']:>4}  loss {h['loss']:.3f}  "
              f"MI {h['mi']:.3f}  一致 {h['consistency']:.3f}  acc {h['acc']:.3f}")

    # 吞吐：默认配置 batch 64
    model = build_default_lse(seed=0)
    opt = Adam(model.parameters(), lr=1e-3)
    xt = Tensor(np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32))
    yt = np.random.default_rng(1).integers(0, 64, 64)
    for _ in range(5):
        opt.zero_grad(); cross_entropy(model(xt), yt).backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(100):
        opt.zero_grad(); cross_entropy(model(xt), yt).backward(); opt.step()
    throughput = round(100 / (time.perf_counter() - t0), 1)
    print(f"\n默认 LSE（88k 参数, batch 64）: {throughput} steps/s")

    out = ROOT / "runs" / "m1_lse"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": {"samples": int(labels.size), "classes": N_CLASSES, "dim": DIM},
        "baseline": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in baseline.items()},
        "runs": runs,
        "default_lse_params": model.n_parameters(),
        "default_lse_steps_per_sec": throughput,
    }
    (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"报告写入 {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())