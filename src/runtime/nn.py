"""最小神经网络层：全部基于自研 Tensor。"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:
    def named_parameters(self, prefix: str = ""):
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                yield f"{prefix}{name}", value
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{name}.")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{i}.")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for _, p in self.named_parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed=None):
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(
            rng.uniform(-bound, bound, (out_features, in_features)),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LayerNorm(Module):
    """手写层归一化：只用 Tensor 的 mean / pow 组合，不调用外部实现。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gain = Tensor(np.ones(dim), requires_grad=True)
        self.bias = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = float(eps)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        inv_std = (var + self.eps) ** -0.5
        return centered * inv_std * self.gain + self.bias