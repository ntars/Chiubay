"""S0 训练循环：REINFORCE（**已知算法对照基线，非主线**）。

警告（ADR-002）：
    本模块用的是 1992 年的策略梯度：折扣回报 G_t、log π(a|s) × advantage、
    对策略反向传播。它不属于本项目的自研机制，只用于报告中的对照组。
    **不要在此文件上继续演进。** 主线见 `world_model.py` / `planner.py`。

不依赖任何 RL 框架。log_probs 由自研 Tensor 产生，梯度经手写反向传播回 LSE。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..runtime.losses import log_softmax
from ..runtime.optim import Adam
from ..runtime.tensor import Tensor
from .metrics import summarize_route

KNOWN_BASELINE = True


@dataclass
class Trajectory:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    raw_rewards: list = field(default_factory=list)
    shaped_rewards: list = field(default_factory=list)


def goal_distance(obs: np.ndarray) -> float:
    """状态索引 7、8 是到目标的归一化相对坐标。"""
    return float(np.hypot(obs[7], obs[8]))


def state_label(obs: np.ndarray) -> int:
    """把连续状态离散成 16 个空间簇，作为分化的代理标签。

    agent 所在象限 ×4 + box 所在象限；无 box 时 box 象限取 0。
    """
    ax, ay = obs[0], obs[1]
    bx = obs[4]
    qa = int(ax >= 0.5) * 2 + int(ay >= 0.5)
    # box 的 y 不在独立维度里，用 d_box_goal 近似
    qb = int(bx >= 0.5) * 2 + int(obs[9] >= 0.5)
    return qa * 4 + qb


class REINFORCETrainer:
    """一步一采样、一 episode 一更新的 on-policy 训练器。"""

    def __init__(self, model, env, lr: float = 1e-3, gamma: float = 0.99,
                 shape_coef: float = 0.3, seed: int | None = None):
        self.model = model
        self.env = env
        self.opt = Adam(model.parameters(), lr=lr)
        self.gamma = float(gamma)
        self.shape_coef = float(shape_coef)
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def rollout(self) -> Trajectory:
        traj = Trajectory()
        obs = self.env.reset()
        n_actions = len(self.model.decoder.bias.data)

        for _ in range(self.env.max_steps):
            logits = self.model(Tensor(obs.reshape(1, -1)))
            lp = log_softmax(logits)
            probs = np.exp(lp.data[0])
            probs = np.clip(probs, 0.0, 1.0)
            probs /= probs.sum()
            action = int(self.rng.choice(n_actions, p=probs))

            onehot = np.zeros((1, n_actions), dtype=lp.data.dtype)
            onehot[0, action] = 1.0
            selected = (lp * Tensor(onehot, dtype=lp.data.dtype)).sum(axis=1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            shaped = reward + self.shape_coef * (
                goal_distance(obs) - goal_distance(next_obs))

            traj.states.append(obs)
            traj.actions.append(action)
            traj.log_probs.append(selected)
            traj.raw_rewards.append(reward)
            traj.shaped_rewards.append(shaped)

            obs = next_obs
            if terminated or truncated:
                break
        return traj

    @staticmethod
    def discounts(rewards, gamma: float, standardize: bool = True) -> np.ndarray:
        g, acc = [], 0.0
        for r in reversed(rewards):
            acc = float(r) + gamma * acc
            g.append(acc)
        out = np.array(g[::-1], dtype=np.float32)
        if standardize and out.size > 1:
            out = (out - out.mean()) / (out.std() + 1e-8)
        return out

    def update(self, traj: Trajectory) -> float:
        returns = self.discounts(traj.shaped_rewards, self.gamma)
        self.opt.zero_grad()
        total = None
        for lp, g in zip(traj.log_probs, returns):
            term = lp * float(g)
            total = term if total is None else total + term
        loss = -(total / max(len(returns), 1))
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def evaluate(self, n_episodes: int = 20) -> dict:
        """采样评估：原始回报 + 路由分化指标。"""
        returns, lengths, probs, labels = [], [], [], []
        for _ in range(n_episodes):
            obs = self.env.reset()
            total, steps = 0.0, 0
            for _ in range(self.env.max_steps):
                out, stats = self.model(Tensor(obs.reshape(1, -1)),
                                        return_stats=True)
                probs.append(stats["gate_probs"][0])
                labels.append(state_label(obs))
                p = np.exp(out.data[0])
                p /= max(p.sum(), 1e-12)
                action = int(self.rng.choice(len(p), p=p))
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                steps += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(steps)

        probs = np.asarray(probs)
        route = summarize_route(probs, np.asarray(labels), top_k=self.model.top_k)
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean([r > 0 for r in returns])),
            "entropy_k": route["entropy_mean"],
            "mutual_info_bits": route["mutual_info_bits"],
            "routing_consistency": route["routing_consistency"],
            "routing_gain": route["routing_gain"],
            "dead_experts": route["dead_experts"],
        }

    def train(self, episodes: int, log_every: int = 50,
              eval_every: int = 100) -> dict:
        start = time.perf_counter()
        total_steps = 0
        last_loss = 0.0

        for ep in range(1, episodes + 1):
            traj = self.rollout()
            total_steps += len(traj.raw_rewards)
            last_loss = self.update(traj)
            if ep % log_every == 0:
                record = {
                    "episode": ep,
                    "loss": round(last_loss, 5),
                    "episode_return": round(float(sum(traj.raw_rewards)), 3),
                    "episode_length": len(traj.raw_rewards),
                }
                if ep % eval_every == 0:
                    record.update({k: round(v, 4) for k, v in
                                   self.evaluate().items()})
                self.history.append(record)

        elapsed = time.perf_counter() - start
        return {
            "episodes": episodes,
            "env_steps": total_steps,
            "steps_per_sec": round(total_steps / elapsed, 1),
            "elapsed_sec": round(elapsed, 2),
            "final_loss": round(last_loss, 5),
            "history": self.history,
            "final_eval": {k: round(v, 4) for k, v in self.evaluate().items()},
        }
