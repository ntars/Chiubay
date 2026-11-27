"""世界模型：只学"做了什么会发生什么"，不学"该做什么"。

输入 `(s, a)` 拼成的向量，输出 `Δs = s′ − s`。
目标量是差分而非绝对状态，因为多数维度每步只动一格，绝对值的方差远大于差分，
直接回归绝对值会让模型把容量浪费在拟合常量上。

**这个模块里没有任何奖励。** 训练信号只有预测误差。
"""
from __future__ import annotations

import numpy as np

from ..runtime.losses import mse
from ..runtime.optim import Adam
from ..runtime.tensor import Tensor
from .lse import LSELayer


class WorldModel:
    """LSE 主干 + 差分预测头。"""

    def __init__(self, obs_dim: int = 16, n_actions: int = 5,
                 n_experts: int = 8, expert_dim: int = 32, top_k: int = 2,
                 lr: float = 2e-3, seed: int | None = None):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.in_dim = self.obs_dim + self.n_actions
        rng = np.random.default_rng(seed)
        self.net = LSELayer(
            dim=self.in_dim, out_dim=self.obs_dim,
            n_experts=n_experts, expert_dim=expert_dim, top_k=top_k,
            use_bridge=False, norm=True, seed=int(rng.integers(1 << 30)),
        )
        self.opt = Adam(self.net.parameters(), lr=lr)
        self.last_loss = float("nan")

    # ---------- 输入构造 ----------
    def encode(self, obs, actions) -> np.ndarray:
        """obs [B, D] + actions [B] → [B, D + A] 的 one-hot 拼接。"""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        a = np.asarray(actions, dtype=np.int64).ravel()
        if a.size != obs.shape[0]:
            raise ValueError("obs 与 actions 行数不一致")
        onehot = np.zeros((a.size, self.n_actions), dtype=np.float32)
        onehot[np.arange(a.size), a] = 1.0
        return np.concatenate([obs, onehot], axis=1)

    # ---------- 预测 ----------
    def predict_delta(self, obs, actions) -> np.ndarray:
        """纯前向，不建图；返回 Δs [B, D]。"""
        x = Tensor(self.encode(obs, actions))
        return self.net(x).data

    def predict_next(self, obs, actions) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        return obs + self.predict_delta(obs, actions)

    # ---------- 学习 ----------
    def train_batch(self, obs, actions, next_obs) -> float:
        """一步 MSE 回归。目标 = next_obs − obs，梯度只到这里为止。"""
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        target = next_obs - obs
        x = Tensor(self.encode(obs, actions))
        pred = self.net(x)
        loss = mse(pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.item())
        return self.last_loss

    def n_parameters(self) -> int:
        return self.net.n_parameters()

    def route_stats(self, obs, actions) -> dict:
        """世界模型自身的路由状态，用于观测专家是否分化。"""
        x = Tensor(self.encode(obs, actions))
        _, stats = self.net(x, return_stats=True)
        return stats


class ReplayBuffer:
    """定长环形缓冲。只存转移，不存奖励——奖励不参与学习。"""

    def __init__(self, capacity: int = 20000, obs_dim: int = 16):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.size = 0
        self.ptr = 0

    def add(self, obs, action: int, next_obs) -> None:
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = int(action)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        if self.size == 0:
            raise ValueError("缓冲区为空")
        idx = rng.integers(0, self.size, size=min(batch, self.size))
        return self.obs[idx], self.actions[idx], self.next_obs[idx]