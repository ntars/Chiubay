"""吞吐验收：小 MLP 的 forward+backward 至少 200 steps/s。"""
import time
import unittest

import numpy as np

from src.runtime.nn import Linear, ReLU, Sequential
from src.runtime.optim import SGD
from src.runtime.tensor import Tensor


class TestThroughput(unittest.TestCase):
    def test_steps_per_second(self):
        np.random.seed(0)
        model = Sequential(Linear(64, 64, seed=1), ReLU(), Linear(64, 10, seed=2))
        opt = SGD(model.parameters(), lr=1e-3)
        x = Tensor(np.random.randn(64, 64))

        for _ in range(10):
            opt.zero_grad()
            out = model(x)
            loss = (out * out).mean()
            loss.backward()
            opt.step()

        n = 200
        start = time.perf_counter()
        for _ in range(n):
            opt.zero_grad()
            out = model(x)
            loss = (out * out).mean()
            loss.backward()
            opt.step()
        elapsed = time.perf_counter() - start
        steps_per_sec = n / elapsed
        self.assertGreaterEqual(
            steps_per_sec, 200.0,
            msg=f"吞吐不足: {steps_per_sec:.1f} steps/s",
        )


if __name__ == "__main__":
    unittest.main()