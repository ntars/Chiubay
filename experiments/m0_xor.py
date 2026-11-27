"""M0：两层 MLP 学 XOR，验证自研张量栈可训练。"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.nn import Linear, Sequential, Sigmoid
from src.runtime.optim import Adam
from src.runtime.serialization import save_checkpoint
from src.runtime.tensor import Tensor


def main() -> int:
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

    epochs = 3000
    start = time.perf_counter()
    loss = None
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(x)
        diff = pred - y
        loss = (diff * diff).mean()
        loss.backward()
        opt.step()
        if epoch % 500 == 0:
            print(f"epoch={epoch:4d} loss={loss.item():.6f}")

    elapsed = time.perf_counter() - start
    pred = model(x).data
    accuracy = float(((pred > 0.5) == (y.data > 0.5)).mean())

    print("\n输入      目标   预测")
    for row, target, got in zip(x.data, y.data.ravel(), pred.ravel()):
        print(f"{row}  {target:.0f}      {got:.4f}")
    print(f"\nloss={loss.item():.6f} accuracy={accuracy:.0%}")
    print(f"steps/s={epochs / elapsed:.1f}  elapsed={elapsed:.2f}s")

    save_checkpoint(
        model,
        "runs/m0_xor/model.npz",
        meta={"stage": "M0", "epochs": epochs, "loss": loss.item(), "accuracy": accuracy},
    )
    print("checkpoint -> runs/m0_xor/model.npz")
    return 0 if accuracy == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())