"""先天驱力：给"预演出的未来"打分。

**本模块没有任何可训练参数，也不接收奖励。**
它是价值判断，不是学习结果——这一取舍是有意的（ADR-002 §5 风险 1）：
价值观留在人手里，不交给梯度去猜。

三个驱力：
- 趋近目标：预测状态离目标更近就更高分。这是唯一的"目标"来源。
- 好奇：去得少的格子更高分。打破局部最优，也让智能体覆盖地图。
- 省力：预测位置没变（撞墙了）就扣分，抑制原地打转。

三者的量纲都归一到 [0, 1] 附近，权重才可解释。
"""
from __future__ import annotations

import numpy as np

# 观测向量的字段索引（与 gridworld.observe 严格对应）
IDX_AX, IDX_AY = 0, 1
IDX_DGX, IDX_DGY = 7, 8
IDX_AT_GOAL = 11
IDX_PASSABLE = (12, 13, 14, 15)   # 上、下、左、右、交互前四项
ACTION_ORDER = ("up", "down", "left", "right", "interact")


class Drives:
    """打分器。无参数，`score` 是纯函数。"""

    def __init__(self, w_goal: float = 1.0, w_novelty: float = 0.15,
                 w_still: float = 0.3, w_sight: float = 0.2, size: int = 8):
        self.w_goal = float(w_goal)
        self.w_novelty = float(w_novelty)
        self.w_still = float(w_still)
        self.w_sight = float(w_sight)
        self.size = int(size)
        self.visit = np.zeros((self.size, self.size), dtype=np.float64)
        self.n_visits = 0

    # ---------- 状态簿记 ----------
    def observe_cell(self, obs) -> tuple[int, int]:
        """从观测还原格子坐标。观测里是 /n 归一化后的值。"""
        n = max(self.size - 1, 1)
        x = int(round(float(obs[IDX_AX]) * n))
        y = int(round(float(obs[IDX_AY]) * n))
        return (min(max(x, 0), self.size - 1), min(max(y, 0), self.size - 1))

    def visit_current(self, obs) -> None:
        """每走一步记一次访问。只影响好奇分，不影响任何参数。"""
        cx, cy = self.observe_cell(obs)
        self.visit[cx, cy] += 1.0
        self.n_visits += 1

    def reset_visits(self, decay: float = 1.0) -> None:
        """清零或按比例衰减访问计数。

        `decay=0` 等价于清零（评估时用，保证每个 episode 起点一致）；
        `decay=0.5` 用于训练：保留"最近去过哪"的记忆，但让旧计数不再
        把新奇度压成一条平线。全程累加会让训练后期 novelty 恒为 0，
        等于悄悄关掉了好奇驱力。
        """
        if decay <= 0.0:
            self.visit[:] = 0.0
            self.n_visits = 0
        else:
            self.visit *= float(decay)
            self.n_visits = int(self.n_visits * decay)

    # ---------- 单项驱动 ----------
    def goal_score(self, pred_obs) -> np.ndarray:
        """负的目标距离。pred_obs [B, D] → 分数 [B]，越接近 0 越好。"""
        p = np.asarray(pred_obs, dtype=np.float64)
        dist = np.hypot(p[:, IDX_DGX], p[:, IDX_DGY])
        return -dist

    def novelty_score(self, pred_obs) -> np.ndarray:
        """1/√(1+n)。没去过的格子得 1，去过 100 次得 0.1 量级。

        用倒数而非负计数：新奇度衰减得快，避免智能体在边角反复蹭分。
        """
        p = np.asarray(pred_obs, dtype=np.float64)
        n = max(self.size - 1, 1)
        out = np.empty(p.shape[0], dtype=np.float64)
        for i in range(p.shape[0]):
            x = int(round(float(p[i, IDX_AX]) * n))
            y = int(round(float(p[i, IDX_AY]) * n))
            x = min(max(x, 0), self.size - 1)
            y = min(max(y, 0), self.size - 1)
            out[i] = 1.0 / np.sqrt(1.0 + self.visit[x, y])
        return out

    def still_penalty(self, obs, pred_obs) -> np.ndarray:
        """预测位置未变 → 1（要扣分），否则 0。"""
        cur = np.asarray(obs, dtype=np.float64)
        p = np.asarray(pred_obs, dtype=np.float64)
        if p.ndim == 1:
            p = p[None, :]
        n = max(self.size - 1, 1)
        cx = int(round(float(cur[IDX_AX]) * n))
        cy = int(round(float(cur[IDX_AY]) * n))
        px = np.round(p[:, IDX_AX] * n).astype(int)
        py = np.round(p[:, IDX_AY] * n).astype(int)
        return ((px == cx) & (py == cy)).astype(np.float64)

    def reached_goal(self, pred_obs) -> np.ndarray:
        """预测到达目标 → 给一个大加分，压过好奇与省力。"""
        p = np.asarray(pred_obs, dtype=np.float64)
        if p.ndim == 1:
            p = p[None, :]
        return (p[:, IDX_AT_GOAL] > 0.5).astype(np.float64)

    def sight_penalty(self, obs) -> np.ndarray:
        """先天回避：视野里明确是墙的方向，直接扣分。

        观测第 12–15 维是四向可通行标志，属于当前就能看见的信息，
        不是预测结果。世界模型对"撞墙"的判断准确率约 84%，若完全依赖
        预测，智能体会在两个格子间来回撞墙。用视野兜住这一层，
        模型只负责"会动的那部分"。
        """
        o = np.asarray(obs, dtype=np.float64).reshape(-1)
        visible = np.array([o[i] for i in IDX_PASSABLE], dtype=np.float64)
        # 前四项对应上下左右；交互永远允许
        return np.concatenate([1.0 - visible, [0.0]])

    # ---------- 合成 ----------
    def score(self, obs, pred_obs) -> np.ndarray:
        """总驱力分。obs 是当前状态（单条），pred_obs 是 [B, D] 的预演结果。"""
        g = self.goal_score(pred_obs)
        # 目标距离归一到 [0,1]：除以 √2 是对角线上限
        g = g / np.sqrt(2.0)
        s = (self.w_goal * g
             + self.w_novelty * self.novelty_score(pred_obs)
             - self.w_still * self.still_penalty(obs, pred_obs)
             - self.w_sight * self.sight_penalty(obs)
             + 2.0 * self.reached_goal(pred_obs))
        return s

    def breakdown(self, obs, pred_obs) -> dict:
        """消融实验要能分别关掉每项，所以给逐项明细。"""
        g = self.goal_score(pred_obs) / np.sqrt(2.0)
        return {
            "goal": self.w_goal * g,
            "novelty": self.w_novelty * self.novelty_score(pred_obs),
            "still": -self.w_still * self.still_penalty(obs, pred_obs),
            "sight": -self.w_sight * self.sight_penalty(obs),
            "reached": 2.0 * self.reached_goal(pred_obs),
        }