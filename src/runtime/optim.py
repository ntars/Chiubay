"""自研优化器：SGD / Adam。"""
from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, params, lr: float = 0.1, momentum: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            p.data -= self.lr * self.velocity[i]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class Adam:
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[i] / (1.0 - self.b1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()