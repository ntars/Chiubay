"""M0 验收：两层 MLP 从随机初始化学会 XOR。"""
import unittest

import numpy as np

from src.runtime.nn import Linear, Sequential, Sigmoid
from src.runtime.optim import Adam
from src.runtime.tensor import Tensor


class TestXOR(unittest.TestCase):
    def test_learn_xor(self):
        np.random.seed(0)
        x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = Tensor([[0.0], [1.0], [1.0], [0.0]])

        model = Sequential(
            Linear(2, 8, seed=1),
            Sigmoid(),
            Linear(8, 1, seed=2),
            Sigmoid(),
        )
        opt = Adam(model.parameters(), lr=0.05)

        loss = None
        for _ in range(3000):
            opt.zero_grad()
            pred = model(x)
            diff = pred - y
            loss = (diff * diff).mean()
            loss.backward()
            opt.step()

        pred = model(x).data
        accuracy = float(((pred > 0.5) == (y.data > 0.5)).mean())
        self.assertEqual(accuracy, 1.0, msg=f"XOR 未学会，预测={pred.ravel()}")
        self.assertLess(loss.item(), 0.01, msg=f"loss 过高: {loss.item()}")


if __name__ == "__main__":
    unittest.main()