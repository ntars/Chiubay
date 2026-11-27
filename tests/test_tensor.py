"""张量内核梯度校验：解析梯度 vs 有限差分。"""
import unittest

import numpy as np

from src.runtime.tensor import Tensor


def numeric_grad(param: Tensor, build_loss, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(param.data)
    it = np.nditer(param.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = param.data[idx]
        param.data[idx] = old + eps
        loss_plus = build_loss().item()
        param.data[idx] = old - eps
        loss_minus = build_loss().item()
        param.data[idx] = old
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


class TestTensorGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_add_broadcast(self):
        x = Tensor(np.random.randn(2, 3), requires_grad=True, dtype=np.float64)
        b = Tensor(np.random.randn(3), requires_grad=True, dtype=np.float64)
        c = Tensor(np.random.randn(2, 3), dtype=np.float64)

        def loss_fn():
            return ((x + b) * c).sum()

        loss_fn().backward()
        np.testing.assert_allclose(b.grad, numeric_grad(b, loss_fn), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(x.grad, numeric_grad(x, loss_fn), atol=1e-6, rtol=1e-5)

    def test_matmul(self):
        a = Tensor(np.random.randn(2, 3), requires_grad=True, dtype=np.float64)
        b = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
        c = Tensor(np.random.randn(2, 4), dtype=np.float64)

        def loss_fn():
            return ((a @ b) * c).sum()

        loss_fn().backward()
        np.testing.assert_allclose(a.grad, numeric_grad(a, loss_fn), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(b.grad, numeric_grad(b, loss_fn), atol=1e-6, rtol=1e-5)

    def test_softmax(self):
        x = Tensor(np.random.randn(3, 5), requires_grad=True, dtype=np.float64)
        c = Tensor(np.random.randn(3, 5), dtype=np.float64)

        def loss_fn():
            return (x.softmax(axis=-1) * c).sum()

        loss_fn().backward()
        np.testing.assert_allclose(x.grad, numeric_grad(x, loss_fn), atol=1e-6, rtol=1e-5)

    def test_sigmoid_tanh_relu(self):
        x = Tensor(np.random.randn(4, 6), requires_grad=True, dtype=np.float64)
        c = Tensor(np.random.randn(4, 6), dtype=np.float64)

        def loss_fn():
            return ((x.sigmoid() + x.tanh() + x.relu()) * c).sum()

        loss_fn().backward()
        np.testing.assert_allclose(x.grad, numeric_grad(x, loss_fn), atol=1e-6, rtol=1e-5)

    def test_reshape_transpose(self):
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True, dtype=np.float64)

        def loss_fn():
            y = x.transpose(2, 1, 0).reshape(4, 6)
            w = Tensor(np.ones((6, 1)), dtype=np.float64)
            return (y @ w).sum()

        loss_fn().backward()
        np.testing.assert_allclose(x.grad, numeric_grad(x, loss_fn), atol=1e-6, rtol=1e-5)

    def test_mean_sum(self):
        x = Tensor(np.random.randn(3, 4), requires_grad=True, dtype=np.float64)
        c = Tensor(np.random.randn(3, 4), dtype=np.float64)

        def loss_fn():
            return (x.mean(axis=0) * c.mean(axis=0)).sum() + x.sum()

        loss_fn().backward()
        np.testing.assert_allclose(x.grad, numeric_grad(x, loss_fn), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()