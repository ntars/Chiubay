"""M1：LSE 层形状、路由稀疏性、梯度正确性、指标口径与专家分化。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.child.lse import LSELayer, build_default_lse
from src.child.metrics import (mutual_information, normalized_entropy,
                               routing_consistency, routing_gain,
                               soft_mutual_information, summarize_route)
from src.runtime.losses import cross_entropy
from src.runtime.optim import Adam
from src.runtime.tensor import Tensor


def numeric_grad(param, loss_fn, eps=1e-6):
    """中心差分，要求参数为 float64。"""
    base = param.data.copy()
    grad = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = base[idx]
        param.data[idx] = orig + eps
        lp = loss_fn().item()
        param.data[idx] = orig - eps
        lm = loss_fn().item()
        param.data[idx] = orig
        grad[idx] = (lp - lm) / (2.0 * eps)
    return grad


def make_model(**kwargs):
    cfg = dict(dim=4, out_dim=2, n_experts=3, expert_dim=3, top_k=3,
               bridge_dim=3, seed=7)
    cfg.update(kwargs)
    return LSELayer(**cfg)


def to_float64(model):
    for p in model.parameters():
        p.data = p.data.astype(np.float64)
        p.grad = None
    return model


class TestLSEForward(unittest.TestCase):
    def test_output_shape(self):
        model = make_model(top_k=2)
        out = model(Tensor(np.random.randn(5, 4)))
        self.assertEqual(out.shape, (5, 2))
        self.assertTrue(np.all(np.isfinite(out.data)))

    def test_topk_sparsity(self):
        model = make_model(n_experts=4, top_k=2)
        _, stats = model(Tensor(np.random.randn(8, 4)), return_stats=True)
        probs = stats["gate_probs"]
        self.assertEqual(probs.shape, (8, 4))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal((probs > 0).sum(axis=1), np.full(8, 2))
        np.testing.assert_array_equal(stats["active"], np.full(8, 2))

    def test_entropy_k1_is_zero(self):
        model = make_model(top_k=1)
        _, stats = model(Tensor(np.random.randn(6, 4)), return_stats=True)
        np.testing.assert_allclose(stats["entropy_k"], 0.0, atol=1e-9)

    def test_entropy_uniform_is_one(self):
        model = make_model(top_k=3)
        model.gate.weight.data[:] = 0.0
        _, stats = model(Tensor(np.random.randn(4, 4)), return_stats=True)
        np.testing.assert_allclose(stats["entropy_k"], 1.0, atol=1e-6)

    def test_gate_usage_sums_to_one(self):
        model = make_model(n_experts=4, top_k=2)
        _, stats = model(Tensor(np.random.randn(32, 4)), return_stats=True)
        self.assertAlmostEqual(float(stats["expert_usage"].sum()), 1.0, places=5)


class TestLSEGradient(unittest.TestCase):
    def test_gradients_full_routing(self):
        """top_k = N 时路由连续，可对全部参数做有限差分。"""
        model = to_float64(make_model(top_k=3))
        x = Tensor(np.random.default_rng(3).normal(size=(2, 4)))
        labels = np.array([0, 1])

        def loss_fn():
            return cross_entropy(model(x), labels)

        loss_fn().backward()

        for name, p in model.named_parameters():
            numeric = numeric_grad(p, loss_fn)
            denom = np.maximum(np.abs(numeric), 1e-4)
            err = float(np.max(np.abs(p.grad - numeric) / denom))
            self.assertLess(err, 5e-4, f"{name} 梯度误差 {err:.2e}")

    def test_gradients_bridge_off(self):
        model = to_float64(make_model(top_k=3, use_bridge=False))
        x = Tensor(np.random.default_rng(5).normal(size=(2, 4)))
        labels = np.array([0, 1])

        def loss_fn():
            return cross_entropy(model(x), labels)

        loss_fn().backward()
        for name, p in model.named_parameters():
            numeric = numeric_grad(p, loss_fn)
            denom = np.maximum(np.abs(numeric), 1e-4)
            err = float(np.max(np.abs(p.grad - numeric) / denom))
            self.assertLess(err, 5e-4, f"{name} 梯度误差 {err:.2e}")

    def test_top2_gate_has_gradient(self):
        """k≥2 时 top-k 内部 softmax 仍有非零 jacobian，门控可训练。"""
        model = make_model(n_experts=4, top_k=2)
        loss = cross_entropy(model(Tensor(np.random.default_rng(1).normal(size=(4, 4)))),
                             np.array([0, 1, 0, 1]))
        loss.backward()
        grad = model.gate.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(np.abs(grad).sum()), 1e-8)

    def test_top1_gate_gradient_is_zero(self):
        """结构限制：k=1 时 softmax 输出为 one-hot，jacobian 恒为 0。

        这不是 bug，而是硬单路由的数学必然。要让 k=1 可训练，
        必须引入 straight-through 或负载均衡辅助损失（M2 处理）。
        """
        model = make_model(n_experts=4, top_k=1)
        loss = cross_entropy(model(Tensor(np.random.default_rng(1).normal(size=(4, 4)))),
                             np.array([0, 1, 0, 1]))
        loss.backward()
        grad = model.gate.weight.grad
        self.assertTrue(grad is None or float(np.abs(grad).sum()) == 0.0)


class TestBridgeSemantics(unittest.TestCase):
    def test_bridge_off_is_plain_weighted_sum(self):
        model = make_model(use_bridge=False, norm=False, top_k=2)
        x = Tensor(np.random.default_rng(11).normal(size=(3, 4)))
        h = model.experts(x).data
        probs, _ = model.gate(x)
        agg = np.einsum("bn,bnm->bm", probs.data, h)
        expected = agg @ model.decoder.weight.data.T + model.decoder.bias.data
        np.testing.assert_allclose(model(x).data, expected, atol=1e-5)

    def test_bridge_switch_changes_output(self):
        kw = dict(n_experts=4, top_k=2, use_bridge=True)
        a = make_model(use_gate_scaled_qk=True, **kw)
        b = make_model(use_gate_scaled_qk=False, **kw)
        for pa, pb in zip(a.parameters(), b.parameters()):
            np.testing.assert_allclose(pa.data, pb.data)
        x = Tensor(np.random.default_rng(2).normal(size=(6, 4)))
        self.assertGreater(float(np.max(np.abs(a(x).data - b(x).data))), 1e-4)

    def test_bridge_residual_preserves_shape(self):
        model = make_model(n_experts=5, top_k=2)
        x = Tensor(np.random.randn(7, 4))
        h = model.experts(x)
        probs, mask = model.gate(x)
        self.assertEqual(model.bridge(h, probs, mask).shape, (7, 5, 3))


class TestMetrics(unittest.TestCase):
    def test_entropy_extremes(self):
        self.assertAlmostEqual(float(normalized_entropy(np.array([[1.0, 0, 0]]))[0]), 0.0)
        self.assertAlmostEqual(float(normalized_entropy(np.array([[0.25] * 4]))[0]), 1.0)

    def test_entropy_gauge_is_topk_independent(self):
        probs = np.zeros((4, 6))
        probs[:, :2] = 0.5
        np.testing.assert_allclose(normalized_entropy(probs, top_k=2), 1.0, atol=1e-9)
        self.assertLess(float(normalized_entropy(probs)[0]), 0.45)

    def test_discrete_mutual_information(self):
        assign = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(mutual_information(assign, assign), np.log2(3), places=6)
        rng = np.random.default_rng(0)
        self.assertLess(mutual_information(rng.integers(0, 3, 6000),
                                           rng.integers(0, 3, 6000)), 0.01)

    def test_soft_mutual_information(self):
        probs = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(soft_mutual_information(probs, labels), 1.0, places=6)
        probs2 = np.full((8, 2), 0.5)
        self.assertLess(abs(soft_mutual_information(probs2, labels.repeat(2))), 1e-9)

    def test_routing_consistency(self):
        sep = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0]])
        self.assertAlmostEqual(routing_consistency(sep, np.array([0, 0, 1, 1])), 1.0)
        mixed = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        self.assertAlmostEqual(routing_consistency(mixed, np.array([0, 0])),
                               1 / np.sqrt(2), places=6)

    def test_routing_gain_rejects_uniform_routing(self):
        uniform = np.full((8, 4), 0.25)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.assertAlmostEqual(routing_gain(uniform, labels), 0.0, places=6)
        sep = np.zeros((8, 4))
        sep[:4, 0], sep[4:, 1] = 1.0, 1.0
        self.assertGreater(routing_gain(sep, labels), 0.2)

    def test_summarize_route_keys(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        out = summarize_route(probs, np.array([0, 1]))
        for k in ("entropy_mean", "entropy_over_n", "mutual_info_bits",
                  "routing_consistency", "routing_gain", "expert_usage",
                  "dead_experts"):
            self.assertIn(k, out)


class TestDifferentiation(unittest.TestCase):
    """3 个可分簇，训练后路由应与状态对齐。"""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.centers = rng.normal(0.0, 3.0, size=(3, 8))
        self.labels = np.repeat(np.arange(3), 300)
        self.x = self.centers[self.labels] + rng.normal(0.0, 0.6, size=(900, 8))

    def test_experts_specialize(self):
        model = LSELayer(dim=8, out_dim=3, n_experts=6, expert_dim=8, top_k=2,
                         bridge_dim=8, seed=1)
        opt = Adam(model.parameters(), lr=0.02)
        rng = np.random.default_rng(42)

        def evaluate():
            _, stats = model(Tensor(self.x), return_stats=True)
            return summarize_route(stats["gate_probs"], self.labels, top_k=2)

        before = evaluate()
        for _ in range(2000):
            idx = rng.integers(0, len(self.labels), 64)
            opt.zero_grad()
            loss = cross_entropy(model(Tensor(self.x[idx])), self.labels[idx])
            loss.backward()
            opt.step()
        after = evaluate()

        print(f"\n[M1] 训练前: consistency={before['routing_consistency']:.3f} "
              f"MI={before['mutual_info_bits']:.3f}bit")
        print(f"[M1] 训练后: consistency={after['routing_consistency']:.3f} "
              f"MI={after['mutual_info_bits']:.3f}bit "
              f"dead={after['dead_experts']} "
              f"usage={np.round(after['expert_usage'], 3)}")

        self.assertLess(float(loss.item()), 0.3, "分类未收敛")
        self.assertGreater(after["routing_consistency"], 0.85, f"分化不足: {after}")
        self.assertGreater(after["mutual_info_bits"], 1.0, f"互信息不足: {after}")
        self.assertGreaterEqual(after["mutual_info_bits"],
                               before["mutual_info_bits"] - 1e-6,
                               f"互信息倒退: before={before}")


class TestBudget(unittest.TestCase):
    def test_default_lse_under_1m(self):
        n = build_default_lse().n_parameters()
        print(f"\n[M1] 默认 LSE 参数: {n:,}")
        self.assertLess(n, 1_000_000)
        self.assertGreater(n, 40_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)