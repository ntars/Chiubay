"""M2：log_softmax、REINFORCE 轨迹、梯度与吞吐。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.child.gridworld import GridWorld
from src.child.lse import LSELayer
from src.child.trainer import REINFORCETrainer, state_label
from src.runtime.losses import log_softmax
from src.runtime.tensor import Tensor


def small_model(seed=0):
    return LSELayer(dim=16, out_dim=5, n_experts=4, expert_dim=8, top_k=2,
                    use_bridge=False, seed=seed)


def small_env(seed=0):
    return GridWorld(size=6, n_obstacles=1, max_steps=40, seed=seed)


class TestLogSoftmax(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3, 5)).astype(np.float64)
        t = Tensor(x)
        t.data = t.data.astype(np.float64)
        out = log_softmax(t)
        shifted = x - x.max(axis=1, keepdims=True)
        expected = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        np.testing.assert_allclose(out.data, expected, atol=1e-6)

    def test_gradient_exists(self):
        x = Tensor(np.random.randn(2, 5).astype(np.float64), requires_grad=True)
        out = log_softmax(x)
        (out * out).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(np.abs(x.grad).sum()), 0.0)


class TestRollout(unittest.TestCase):
    def test_trajectory_length(self):
        tr = REINFORCETrainer(small_model(), small_env(), seed=0)
        traj = tr.rollout()
        n = len(traj.raw_rewards)
        self.assertGreater(n, 0)
        self.assertLessEqual(n, 40)
        self.assertEqual(len(traj.log_probs), n)
        self.assertEqual(len(traj.states), n)
        self.assertEqual(len(traj.actions), n)

    def test_discounts_shape(self):
        g = REINFORCETrainer.discounts([1.0, 0.0, -1.0], 0.9)
        self.assertEqual(g.shape, (3,))
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertAlmostEqual(float(g.mean()), 0.0, places=6)

    def test_update_produces_gradients(self):
        tr = REINFORCETrainer(small_model(), small_env(), seed=1)
        traj = tr.rollout()
        loss = tr.update(traj)
        self.assertTrue(np.isfinite(loss))
        grads = [p.grad for p in tr.model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(any(float(np.abs(g).sum()) > 0 for g in grads))

    def test_state_label_range(self):
        obs = small_env().reset()
        self.assertIn(state_label(obs), range(16))


class TestTraining(unittest.TestCase):
    def test_short_run_is_stable(self):
        tr = REINFORCETrainer(small_model(seed=2), small_env(seed=2),
                              lr=1e-3, seed=2)
        report = tr.train(episodes=60, log_every=30, eval_every=60)
        self.assertTrue(np.isfinite(report["final_loss"]))
        self.assertGreater(report["env_steps"], 0)
        self.assertIn("mean_return", report["final_eval"])
        self.assertIn("mutual_info_bits", report["final_eval"])

    def test_throughput_above_g0_gate(self):
        tr = REINFORCETrainer(small_model(seed=3), small_env(seed=3),
                              lr=1e-3, seed=3)
        report = tr.train(episodes=40, log_every=100, eval_every=100)
        print(f"\n[M2] 训练吞吐（含环境+反向）: {report['steps_per_sec']} steps/s")
        self.assertGreater(report["steps_per_sec"], 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)