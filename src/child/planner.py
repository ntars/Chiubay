"""内屏：把候选动作在模型里预演一遍，按先天驱力挑一个。

**本模块没有任何可训练参数，也不是策略网络。**
它不产生动作概率分布，也不接收梯度——选动作是一个 `np.argmax`。
因此这里不存在策略梯度，也没有"提高被奖励动作概率"的路径（ADR-002 §3）。

预演深度 P0 = 1：只看一步。P1（深度 3 贪心 + 剪枝）留给 M3，见方案 §6。
"""
from __future__ import annotations

import numpy as np

from .drives import Drives
from .world_model import WorldModel

STILL, MOVED = 0, 1


class InnerScreen:
    """单步预演决策器。"""

    def __init__(self, model: WorldModel, drives: Drives, n_actions: int = 5,
                 epsilon: float = 0.2, epsilon_min: float = 0.02,
                 epsilon_decay: float = 0.9995, seed: int | None = None,
                 use_goal: bool = True, use_novelty: bool = True,
                 use_still: bool = True, use_sight: bool = True):
        self.model = model
        self.drives = drives
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)
        self.use_goal = bool(use_goal)
        self.use_novelty = bool(use_novelty)
        self.use_still = bool(use_still)
        self.use_sight = bool(use_sight)
        self.last_scores = None

    # ---------- 预演 ----------
    def predict_all(self, obs) -> np.ndarray:
        """对全部候选动作各预演一次，返回 [A, D] 的预测下一状态。"""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        batch = np.repeat(obs[None, :], self.n_actions, axis=0)
        actions = np.arange(self.n_actions, dtype=np.int64)
        return self.model.predict_next(batch, actions)

    def score_actions(self, obs) -> np.ndarray:
        """驱力打分。开关为关时该项权重置 0，用于消融。"""
        pred = self.predict_all(obs)
        d = self.drives
        parts = d.breakdown(obs, pred)
        s = np.zeros(self.n_actions, dtype=np.float64)
        if self.use_goal:
            s += parts["goal"] + parts["reached"]
        if self.use_novelty:
            s += parts["novelty"]
        if self.use_still:
            s += parts["still"]
        if self.use_sight:
            s += parts["sight"]
        return s

    # ---------- 决策 ----------
    def act(self, obs, explore: bool = True) -> int:
        scores = self.score_actions(obs)
        self.last_scores = scores
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(scores))

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    # ---------- 诊断 ----------
    def score_gap(self, obs) -> float:
        """最优与次优的分差。分差接近 0 说明驱力在该状态下无法区分动作。"""
        s = np.sort(self.score_actions(obs))[::-1]
        return float(s[0] - s[1]) if s.size > 1 else 0.0