"""智能体：世界模型 + 驱力 + 内屏的闭环。

训练循环里**只有一个损失**：世界模型的预测误差（`model.train_batch`）。
奖励 `r` 只被累加进统计，从不传给任何 `backward()`。

这是 ADR-002 §3.2 的可执行形式：把 `env.step` 返回的 `r` 换成常数 0，
训练过程（模型参数轨迹）完全不变。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .drives import Drives
from .gridworld import GridWorld
from .metrics import summarize_route
from .planner import InnerScreen
from .stats import iqm
from .world_model import ReplayBuffer, WorldModel


@dataclass
class EpisodeLog:
    returns: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    successes: list = field(default_factory=list)
    model_losses: list = field(default_factory=list)


class Agent:
    """S0 阶段的完整智能体。"""

    def __init__(self, env: GridWorld, model: WorldModel | None = None,
                 drives: Drives | None = None, screen: InnerScreen | None = None,
                 buffer_size: int = 20000, batch: int = 64,
                 train_every: int = 4, warmup: int = 500,
                 seed: int | None = None):
        self.env = env
        self.model = model or WorldModel(seed=seed)
        self.drives = drives or Drives(size=env.size)
        self.screen = screen or InnerScreen(self.model, self.drives,
                                            n_actions=5, seed=seed)
        self.buffer = ReplayBuffer(buffer_size, obs_dim=self.model.obs_dim)
        self.batch = int(batch)
        self.train_every = int(train_every)
        self.warmup = int(warmup)
        self.rng = np.random.default_rng(seed)
        self.global_step = 0
        self.history: list[dict] = []

    # ---------- 单 episode ----------
    def run_episode(self, train: bool = True) -> dict:
        obs = self.env.reset()
        # 训练时衰减而非清零：保留"最近去过哪"，但防止计数累积把好奇压平
        self.drives.reset_visits(decay=0.5 if train else 0.0)
        self.drives.visit_current(obs)
        total_r, steps, hit = 0.0, 0, False
        losses = []

        for _ in range(self.env.max_steps):
            if train and self.buffer.size < self.warmup:
                action = int(self.rng.integers(self.screen.n_actions))
            else:
                action = self.screen.act(obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            # 奖励只进统计，不进学习
            total_r += reward
            steps += 1
            self.global_step += 1

            if train:
                self.buffer.add(obs, action, next_obs)
                if (self.buffer.size >= self.warmup
                        and self.global_step % self.train_every == 0):
                    o, a, n = self.buffer.sample(self.batch, self.rng)
                    losses.append(self.model.train_batch(o, a, n))

            obs = next_obs
            self.drives.visit_current(obs)
            if terminated:
                hit = True
                break
            if truncated:
                break

        if train:
            self.screen.decay_epsilon()

        return {"return": total_r, "length": steps, "success": hit,
                "model_loss": float(np.mean(losses)) if losses else float("nan")}

    # ---------- 评估（不训练、不探索） ----------
    def evaluate(self, n_episodes: int = 20, epsilon: float = 0.0) -> dict:
        keep, self.screen.epsilon = self.screen.epsilon, epsilon
        returns, lengths, successes = [], [], []
        try:
            for _ in range(n_episodes):
                obs = self.env.reset()
                total, steps, hit = 0.0, 0, False
                for _ in range(self.env.max_steps):
                    action = self.screen.act(obs, explore=False)
                    obs, r, terminated, truncated, _ = self.env.step(action)
                    total += r
                    steps += 1
                    if terminated:
                        hit = True
                        break
                    if truncated:
                        break
                returns.append(total)
                lengths.append(steps)
                successes.append(1.0 if hit else 0.0)
        finally:
            self.screen.epsilon = keep

        return {
            "success_rate": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "iqm_return": iqm(returns),
        }

    # ---------- 模型质量 ----------
    def probe_model(self, n_samples: int = 2000, seed: int = 12345) -> dict:
        """在 held-out 随机转移上测预测精度。

        不用训练缓冲，避免"记住了就是学会了"。
        RNG 独立于训练 RNG，保证可复现。
        """
        env = GridWorld(size=self.env.size, n_obstacles=self.env.n_obstacles,
                        max_steps=self.env.max_steps, has_box=self.env.has_box,
                        seed=seed)
        rng = np.random.default_rng(seed)
        obs_l, act_l, nxt_l = [], [], []
        while len(obs_l) < n_samples:
            o = env.reset()
            for _ in range(env.max_steps):
                a = int(rng.integers(5))
                o2, _, term, trunc, _ = env.step(a)
                obs_l.append(o)
                act_l.append(a)
                nxt_l.append(o2)
                o = o2
                if term or trunc or len(obs_l) >= n_samples:
                    break
        obs = np.asarray(obs_l[:n_samples], dtype=np.float32)
        act = np.asarray(act_l[:n_samples], dtype=np.int64)
        nxt = np.asarray(nxt_l[:n_samples], dtype=np.float32)

        pred = self.model.predict_next(obs, act)
        cells = max(self.env.size - 1, 1)
        pos_err = np.hypot(pred[:, 0] - nxt[:, 0], pred[:, 1] - nxt[:, 1]) / cells

        # 撞墙预测：真实位置没动的样本里，预测是否也没动
        moved = np.hypot(nxt[:, 0] - obs[:, 0], nxt[:, 1] - obs[:, 1]) * cells
        still = moved < 0.5
        pred_still = (np.hypot(pred[:, 0] - obs[:, 0],
                               pred[:, 1] - obs[:, 1]) * cells) < 0.5
        block_acc = float((pred_still[still]).mean()) if still.any() else float("nan")

        stats = self.model.route_stats(obs, act)
        labels = (np.round(obs[:, 0] * cells).astype(int) * self.env.size
                  + np.round(obs[:, 1] * cells).astype(int))
        route = summarize_route(stats["gate_probs"], labels,
                                top_k=self.model.net.top_k)
        return {
            "pos_mae": float(np.mean(pos_err)),
            "block_acc": block_acc,
            "n": int(obs.shape[0]),
            "routing_gain": route["routing_gain"],
            "dead_experts": route["dead_experts"],
            "entropy_k": route["entropy_mean"],
        }

    # ---------- 训练 ----------
    def probe_decisions(self, n: int = 200, seed: int = 777) -> dict:
        """决策一致率：模型选的动作 vs 真值选的动作。

        这是比位置 MAE 更贴近行为的指标。位置误差小不代表选动作对，
        位置误差大也不代表选错——驱力打分对小幅误差本来就不敏感。
        G0 的模型门用这个指标，不用 MAE。
        """
        import copy

        env = GridWorld(size=self.env.size, n_obstacles=self.env.n_obstacles,
                        max_steps=self.env.max_steps, has_box=self.env.has_box,
                        seed=seed)
        rng = np.random.default_rng(seed)
        drives = Drives(size=env.size, w_goal=self.drives.w_goal,
                        w_novelty=self.drives.w_novelty,
                        w_still=self.drives.w_still)
        agree, total = 0, 0
        for _ in range(n):
            obs = env.reset()
            drives.reset_visits()
            drives.visit_current(obs)
            for _ in range(env.max_steps):
                truth = []
                for a in range(self.screen.n_actions):
                    probe = copy.deepcopy(env)
                    truth.append(probe.step(a)[0])
                truth = np.asarray(truth, dtype=np.float32)
                pred = self.model.predict_next(
                    np.repeat(obs[None, :], self.screen.n_actions, axis=0),
                    np.arange(self.screen.n_actions))
                a_true = int(np.argmax(drives.score(obs, truth)))
                a_pred = int(np.argmax(drives.score(obs, pred)))
                agree += int(a_true == a_pred)
                total += 1

                obs, _, term, trunc, _ = env.step(int(rng.integers(
                    self.screen.n_actions)))
                drives.visit_current(obs)
                if term or trunc:
                    break
        return {"decision_agreement": agree / max(total, 1), "samples": total}

    # ---------- 训练 ----------
    def train(self, episodes: int, log_every: int = 100,
              eval_every: int = 500, eval_n: int = 20) -> dict:
        start = time.perf_counter()
        steps = 0
        for ep in range(1, episodes + 1):
            info = self.run_episode(train=True)
            steps += info["length"]
            if ep % log_every == 0:
                rec = {"episode": ep, **{k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in info.items()},
                       "epsilon": round(self.screen.epsilon, 4)}
                if ep % eval_every == 0:
                    rec.update({f"eval_{k}": round(v, 4)
                                for k, v in self.evaluate(eval_n).items()})
                    rec.update({f"probe_{k}": (round(v, 4) if isinstance(v, float) else v)
                                for k, v in self.probe_model(400).items()})
                self.history.append(rec)
        elapsed = time.perf_counter() - start
        return {
            "episodes": episodes,
            "env_steps": steps,
            "steps_per_sec": round(steps / elapsed, 1),
            "elapsed_sec": round(elapsed, 2),
            "history": self.history,
            "final_eval": self.evaluate(eval_n * 2),
            "model": self.probe_model(),
            "params": self.model.n_parameters(),
        }