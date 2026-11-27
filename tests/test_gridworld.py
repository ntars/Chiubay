"""M2：自研 GridWorld 的形状、边界、终止、推箱与复现性。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.child.gridworld import DOWN, INTERACT, LEFT, RIGHT, UP, GridWorld, OBS_DIM


class TestGridWorld(unittest.TestCase):
    def test_observation_shape_and_range(self):
        env = GridWorld(size=8, seed=0)
        obs = env.reset(seed=1)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all((obs >= -0.2) & (obs <= 1.2)))

    def test_reset_is_deterministic(self):
        a = GridWorld(size=8, seed=7).reset()
        b = GridWorld(size=8, seed=7).reset()
        np.testing.assert_array_equal(a, b)

    def test_step_signature(self):
        env = GridWorld(size=8, seed=0)
        result = env.step(RIGHT)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("pos", info)

    def test_wall_blocks_and_penalizes(self):
        env = GridWorld(size=4, n_obstacles=0, seed=0)
        env.reset()
        env.pos = (0, 0)
        _, reward, _, _, info = env.step(UP)
        self.assertEqual(info["pos"], (0, 0))
        self.assertLessEqual(reward, -0.05)

    def test_obstacle_blocks(self):
        env = GridWorld(size=4, n_obstacles=0, seed=0)
        env.reset()
        env.obstacles = {(1, 0)}
        _, _, _, _, info = env.step(RIGHT)
        self.assertEqual(info["pos"], (0, 0))

    def test_goal_terminates_with_reward(self):
        env = GridWorld(size=4, n_obstacles=0, has_box=False, seed=0)
        env.reset()
        env.pos = (2, 3)
        _, reward, terminated, truncated, _ = env.step(RIGHT)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreaterEqual(reward, 0.99)

    def test_push_box(self):
        env = GridWorld(size=6, n_obstacles=0, seed=0)
        env.reset()
        env.pos = (0, 0)
        env.box = (1, 0)
        _, reward, _, _, info = env.step(INTERACT)
        self.assertEqual(info["pos"], (1, 0))
        self.assertEqual(info["box"], (2, 0))
        self.assertGreaterEqual(reward, 0.09)

    def test_push_box_into_goal(self):
        env = GridWorld(size=4, n_obstacles=0, seed=0)
        env.reset()
        env.pos = (3, 1)
        env.box = (3, 2)
        _, reward, _, _, info = env.step(INTERACT)
        self.assertEqual(info["box"], (3, 3))
        self.assertTrue(info["box_delivered"])
        self.assertGreaterEqual(reward, 0.59)

    def test_truncation(self):
        env = GridWorld(size=4, n_obstacles=0, max_steps=3, seed=0)
        env.reset()
        for i in range(3):
            _, _, terminated, truncated, _ = env.step(LEFT)
            if i < 2:
                self.assertFalse(truncated)
        self.assertTrue(truncated)

    def test_render(self):
        env = GridWorld(size=4, n_obstacles=0, has_box=False, seed=0)
        env.reset()
        text = env.render()
        self.assertEqual(len(text.splitlines()), 4)
        self.assertIn("A", text)
        self.assertIn("G", text)

    def test_invalid_action(self):
        env = GridWorld(size=4, seed=0)
        with self.assertRaises(ValueError):
            env.step(9)


if __name__ == "__main__":
    unittest.main(verbosity=2)