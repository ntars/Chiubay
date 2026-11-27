"""自研 checkpoint：JSON 元数据 + NPZ 权重。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_checkpoint(model, path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name.replace(".", "__"): p.data for name, p in model.named_parameters()}
    np.savez(path, **state)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, ensure_ascii=False, indent=2)


def load_checkpoint(model, path) -> dict:
    path = Path(path)
    with np.load(path) as data:
        for name, p in model.named_parameters():
            key = name.replace(".", "__")
            if key not in data:
                raise KeyError(f"checkpoint 缺少参数: {name}")
            if data[key].shape != p.data.shape:
                raise ValueError(f"形状不匹配: {name} {data[key].shape} != {p.data.shape}")
            p.data[...] = data[key]
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}