"""玖白（Chiubay）自研张量内核（M0）。

约束：
- 不依赖任何机器学习 / 自动微分库。
- NumPy 仅用于内存块、索引、逐元素运算与底层 BLAS。
- 反向传播由本文件手写实现。
"""
from __future__ import annotations

import numpy as np


def as_tensor(value):
    """把 Python 标量 / NumPy 数组 / Tensor 统一成 Tensor。"""
    return value if isinstance(value, Tensor) else Tensor(value)


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """把广播后的梯度缩回原始形状。"""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad: bool = False, dtype=np.float32,
                 _children=(), _op: str = ""):
        self.data = np.asarray(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    # ---------- 基础属性 ----------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.item())

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False, dtype=self.data.dtype)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # ---------- 反向传播 ----------
    def _accumulate(self, grad) -> None:
        if not self.requires_grad:
            return
        grad = np.asarray(grad, dtype=self.data.dtype)
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad = self.grad + grad

    def backward(self) -> None:
        topo, seen = [], set()

        def build(t: "Tensor"):
            if id(t) in seen:
                return
            seen.add(id(t))
            for parent in t._prev:
                build(parent)
            topo.append(t)

        build(self)
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            # 常量分支不参与梯度，跳过其闭包，避免访问 None
            if t.grad is None:
                continue
            t._backward()

    # ---------- 算术 ----------
    def __add__(self, other):
        other = as_tensor(other)
        dtype = np.result_type(self.data, other.data)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     dtype=dtype, _children=(self, other), _op="add")

        def _backward():
            self._accumulate(_unbroadcast(out.grad, self.shape))
            other._accumulate(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __rsub__(self, other):
        return as_tensor(other) + (-self)

    def __mul__(self, other):
        other = as_tensor(other)
        dtype = np.result_type(self.data, other.data)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     dtype=dtype, _children=(self, other), _op="mul")

        def _backward():
            self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            other._accumulate(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = as_tensor(other)
        dtype = np.result_type(self.data, other.data)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     dtype=dtype, _children=(self, other), _op="matmul")

        def _backward():
            self._accumulate(_unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.shape))
            other._accumulate(_unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.shape))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = as_tensor(other)
        return self * other.reciprocal()

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            raise TypeError("Tensor ** Tensor 未实现；M0 只支持标量幂")
        out = Tensor(self.data ** exponent, self.requires_grad,
                     dtype=self.data.dtype, _children=(self,), _op="pow")

        def _backward():
            self._accumulate(out.grad * exponent * (self.data ** (exponent - 1)))

        out._backward = _backward
        return out

    # ---------- 逐元素 ----------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad,
                     dtype=self.data.dtype, _children=(self,), _op="relu")

        def _backward():
            self._accumulate(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    def sigmoid(self):
        z = np.clip(self.data, -60.0, 60.0)
        s = 1.0 / (1.0 + np.exp(-z))
        out = Tensor(s, self.requires_grad, dtype=self.data.dtype,
                     _children=(self,), _op="sigmoid")

        def _backward():
            self._accumulate(out.grad * s * (1.0 - s))

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, dtype=self.data.dtype,
                     _children=(self,), _op="tanh")

        def _backward():
            self._accumulate(out.grad * (1.0 - t * t))

        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(np.clip(self.data, -60.0, 60.0))
        out = Tensor(e, self.requires_grad, dtype=self.data.dtype,
                     _children=(self,), _op="exp")

        def _backward():
            self._accumulate(out.grad * e)

        out._backward = _backward
        return out

    def log(self):
        x = np.clip(self.data, 1e-12, None)
        out = Tensor(np.log(x), self.requires_grad, dtype=self.data.dtype,
                     _children=(self,), _op="log")

        def _backward():
            self._accumulate(out.grad / x)

        out._backward = _backward
        return out

    def reciprocal(self):
        x = np.clip(self.data, 1e-12, None)
        out = Tensor(1.0 / x, self.requires_grad, dtype=self.data.dtype,
                     _children=(self,), _op="reciprocal")

        def _backward():
            self._accumulate(out.grad * (-1.0 / (x * x)))

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / np.sum(e, axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, dtype=self.data.dtype,
                     _children=(self,), _op="softmax")

        def _backward():
            g = out.grad
            self._accumulate(s * (g - np.sum(g * s, axis=axis, keepdims=True)))

        out._backward = _backward
        return out

    # ---------- 形状 ----------
    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad,
                     dtype=self.data.dtype, _children=(self,), _op="sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                g = np.expand_dims(g, axes)
            self._accumulate(np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([self.shape[a] for a in axes]))
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), self.requires_grad,
                     dtype=self.data.dtype, _children=(self,), _op="mean")

        def _backward():
            g = out.grad / n
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                g = np.expand_dims(g, axes)
            self._accumulate(np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), self.requires_grad,
                     dtype=self.data.dtype, _children=(self,), _op="reshape")

        def _backward():
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if not axes:
            axes = tuple(range(self.ndim))[::-1]
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(axes), self.requires_grad,
                     dtype=self.data.dtype, _children=(self,), _op="transpose")
        inverse = tuple(int(i) for i in np.argsort(axes))

        def _backward():
            self._accumulate(out.grad.transpose(inverse))

        out._backward = _backward
        return out