"""自研 GridWorld（S0 网格房）。

不依赖 gymnasium；接口形似但完全自己实现。
8×8 网格，1 目标、0–3 障碍、1 可推物体，16 维状态向量，5 个离散动作。

M2.1 变更：
1. **连通性保证**：随机采样后做 BFS，目标不可达则重采。原实现有概率生成
   被障碍围死的目标，这类 episode 无论策略多好都必然失败，是上一轮 seed 方差
   的主要来源之一。
2. **固定地图**：`fixed_layout=True` 时首次 reset 采样一次后冻结，供课程学习
   前期使用——先在同一张地图上学会导航，再泛化到随机地图。
3. **难度档位**：`difficulty` 0/1/2 控制障碍数；`require_box` 时目标格被围住，
   必须推箱才能进入。
"""
from __future__ import annotations

from collections import deque

import numpy as np

UP, DOWN, LEFT, RIGHT, INTERACT = range(5)
ACTIONS = ("up", "down", "left", "right", "interact")
_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
OBS_DIM = 16


class GridWorld:
    """8×8 网格房。agent 从左上角出发，目标在右下角。"""

    def __init__(self, size: int = 8, n_obstacles: int = 3,
                 max_steps: int = 80, has_box: bool = True,
                 fixed_layout: bool = False, require_box: bool = False,
                 seed: int | None = None):
        self.size = int(size)
        self.n_obstacles = int(min(max(n_obstacles, 0), (size - 2) ** 2 - 2))
        self.max_steps = int(max_steps)
        self.has_box = bool(has_box)
        self.fixed_layout = bool(fixed_layout)
        self.require_box = bool(require_box) and self.has_box
        self.rng = np.random.default_rng(seed)
        self._frozen: dict | None = None
        self.reset()

    # ---------- 生命周期 ----------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._frozen = None

        self.pos = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.steps = 0
        self.box_delivered = False

        if self.fixed_layout and self._frozen is not None:
            self.box = self._frozen["box"]
            self.obstacles = set(self._frozen["obstacles"])
            return self.observe()

        for _ in range(200):
            self.box = self._sample_box() if self.has_box else None
            self.obstacles = self._sample_obstacles()
            if self.require_box:
                self.obstacles |= self._ring_around_goal()
            if self._layout_valid():
                break
        else:
            # 极端参数下退化为无障碍，保证环境永远可用
            self.obstacles = set()
            self.box = (1, 1) if self.has_box else None

        if self.fixed_layout:
            self._frozen = {"box": self.box,
                            "obstacles": tuple(sorted(self.obstacles))}
        return self.observe()

    def _sample_box(self):
        for _ in range(100):
            p = (int(self.rng.integers(1, self.size - 1)),
                 int(self.rng.integers(1, self.size - 1)))
            if p != self.goal and p != self.pos:
                return p
        return (1, 1)

    def _sample_obstacles(self):
        occupied = {self.pos, self.goal}
        if self.box is not None:
            occupied.add(self.box)
        cells = [(x, y) for x in range(self.size) for y in range(self.size)
                 if (x, y) not in occupied]
        n = min(self.n_obstacles, max(len(cells) - 1, 0))
        if n == 0:
            return set()
        idx = self.rng.choice(len(cells), size=n, replace=False)
        return {cells[int(i)] for i in idx}

    def _ring_around_goal(self) -> set:
        """把目标四邻封死（越界的跳过），逼 agent 必须推箱才能进。"""
        gx, gy = self.goal
        ring = set()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = gx + dx, gy + dy
            if 0 <= x < self.size and 0 <= y < self.size and (x, y) != self.pos:
                ring.add((x, y))
        # 留一个入口，否则箱子和 agent 都进不去
        if ring:
            ring.discard(max(ring, key=lambda c: (c[0] + c[1])))
        return ring

    # ---------- 连通性 ----------
    def _passable(self, x: int, y: int) -> bool:
        return (0 <= x < self.size and 0 <= y < self.size
                and (x, y) not in self.obstacles)

    def _reachable(self, start, goal, blocked=()) -> bool:
        if start == goal:
            return True
        blocked = set(blocked)
        seen = {start}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in _DELTAS[:4]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen or (nx, ny) in blocked:
                    continue
                if not self._passable(nx, ny):
                    continue
                if (nx, ny) == goal:
                    return True
                seen.add((nx, ny))
                q.append((nx, ny))
        return False

    def _layout_valid(self) -> bool:
        """目标可达；有箱子时箱子也必须可达，且能推到目标旁。"""
        if not self._reachable(self.pos, self.goal):
            return False
        if self.box is None:
            return True
        if not self._reachable(self.pos, self.box):
            return False
        # 箱子到目标：允许箱子占位，检查目标邻格是否可达
        gx, gy = self.goal
        for dx, dy in _DELTAS[:4]:
            x, y = gx + dx, gy + dy
            if 0 <= x < self.size and 0 <= y < self.size and self._passable(x, y):
                if self._reachable(self.pos, (x, y), blocked={self.box}):
                    return True
        return False

    # ---------- 状态 ----------
    def observe(self) -> np.ndarray:
        ax, ay = self.pos
        gx, gy = self.goal
        n = max(self.size - 1, 1)
        bx, by = self.box if self.box is not None else (gx, gy)
        diag = float(np.sqrt(2) * n)
        box_active = 1.0 if self.box is not None else 0.0
        d_box = float(np.hypot(ax - bx, ay - by)) / diag
        d_box_goal = float(np.hypot(bx - gx, by - gy)) / diag

        return np.array([
            ax / n, ay / n,
            gx / n, gy / n,
            bx / n, by / n,
            box_active,
            (gx - ax) / n, (gy - ay) / n,
            d_box, d_box_goal,
            1.0 if ax == gx and ay == gy else 0.0,
            float(self._passable(ax - 1, ay)),
            float(self._passable(ax + 1, ay)),
            float(self._passable(ax, ay - 1)),
            float(self._passable(ax, ay + 1)),
        ], dtype=np.float32)

    # ---------- 转移 ----------
    def step(self, action: int):
        a = int(action)
        if a < 0 or a >= len(ACTIONS):
            raise ValueError(f"非法动作 {a}")
        self.steps += 1
        reward = -0.01
        dx, dy = _DELTAS[a]

        if a == INTERACT:
            reward += self._try_push()
        else:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if self._passable(nx, ny) and (nx, ny) != self.box:
                self.pos = (nx, ny)
            else:
                reward -= 0.05

        terminated = self.pos == self.goal and (
            not self.require_box or self.box_delivered)
        if terminated:
            reward += 1.0
        truncated = self.steps >= self.max_steps
        info = {
            "pos": self.pos,
            "box": self.box,
            "box_delivered": self.box_delivered,
            "action": ACTIONS[a],
        }
        return self.observe(), reward, terminated, truncated, info

    def _try_push(self) -> float:
        """相邻则推箱一格；成功 +0.1，箱到目标额外 +0.5。"""
        if self.box is None:
            return -0.02
        bx, by = self.box
        ax, ay = self.pos
        if abs(bx - ax) + abs(by - ay) != 1:
            return -0.02
        nx, ny = bx + (bx - ax), by + (by - ay)
        if not self._passable(nx, ny):
            return -0.02
        self.pos = (bx, by)
        self.box = (nx, ny)
        reward = 0.1
        if self.box == self.goal and not self.box_delivered:
            self.box_delivered = True
            reward += 0.5
        return reward

    # ---------- 展示 ----------
    def render(self) -> str:
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for x, y in self.obstacles:
            grid[x][y] = "#"
        grid[self.goal[0]][self.goal[1]] = "G"
        if self.box is not None:
            grid[self.box[0]][self.box[1]] = "*" if self.box == self.goal else "B"
        grid[self.pos[0]][self.pos[1]] = "A"
        return "\n".join("".join(row) for row in grid)
