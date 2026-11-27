"""checkpoint 保存 / 加载一致性。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.runtime.nn import Linear, Sequential, Sigmoid
from src.runtime.serialization import load_checkpoint, save_checkpoint
from src.runtime.tensor import Tensor


class TestCheckpoint(unittest.TestCase):
    def test_roundtrip(self):
        np.random.seed(0)
        model = Sequential(Linear(2, 4, seed=1), Sigmoid(), Linear(4, 1, seed=2), Sigmoid())
        x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        before = model(x).data.copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            save_checkpoint(model, path, meta={"stage": "test"})
            clone = Sequential(Linear(2, 4, seed=9), Sigmoid(), Linear(4, 1, seed=9), Sigmoid())
            meta = load_checkpoint(clone, path)
            after = clone(x).data

        np.testing.assert_allclose(before, after, atol=1e-6)
        self.assertEqual(meta["stage"], "test")


if __name__ == "__main__":
    unittest.main()