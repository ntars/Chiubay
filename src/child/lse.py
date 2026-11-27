"""LSE：分层技能集成层（Layered Skill Ensemble）。

结构（自研，不借用任何现成 MoE / Transformer 实现）：

    x [B, D]
      ├─ ExpertBank  → H [B, N, M]      每个专家输出一个技能向量
      ├─ Gate        → w [B, N]         稀疏门控，只保留 Top-K
      └─ LinkageBridge → H' [B, N, M]   被共同激活的专家之间交换信息
    h = Σ_i w_i · H'_i  → [B, M] → Norm → Decoder → [B, out]

设计约束：
1. Top-K 选择是离散的，mask 视为常数；梯度只流经被选中项的权重，与硬路由一致。
2. 联动桥的 Q/K 由门控权重缩放（`use_gate_scaled_qk`），并对未激活专家做 key 屏蔽
   （`mask_inactive`）。禁止退化成 attn(H, H, H) —— 那是普通自注意力，不是联动。
3. 所有开关都可关闭，用于消融实验；关闭后必须仍能训练。
"""
from __future__ import annotations

import numpy as np

from ..runtime.nn import LayerNorm, Linear, Module
from ..runtime.tensor import Tensor
from .metrics import normalized_entropy


class ExpertBank(Module):
    """N 个专家共享一次矩阵乘，按 [B, N, M] 输出。

    参数布局：weight 为 [N*M, D]，第 i 个专家占第 i*M:(i+1)*M 行。
    这样一次 GEMM 即可算出全部专家，避免 N 次小矩阵乘的开销。
    """

    def __init__(self, dim: int, n_experts: int, expert_dim: int,
                 activation: str = "relu", seed: int | None = None):
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(dim)
        self.n_experts = int(n_experts)
        self.expert_dim = int(expert_dim)
        self.activation = activation
        self.weight = Tensor(
            rng.uniform(-bound, bound, (n_experts * expert_dim, dim)),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(n_experts * expert_dim), requires_grad=True)

    def forward(self, x):
        b = x.shape[0]
        flat = x @ self.weight.transpose() + self.bias
        h = flat.reshape(b, self.n_experts, self.expert_dim)
        if self.activation == "relu":
            return h.relu()
        if self.activation == "tanh":
            return h.tanh()
        return h


class Gate(Module):
    """Top-K 稀疏门控。

    返回 (probs [B, N], mask [B, N])。mask 为 0/1 常数，不参与梯度。
    """

    def __init__(self, dim: int, n_experts: int, top_k: int,
                 temperature: float = 1.0, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.n_experts = int(n_experts)
        self.top_k = int(min(top_k, n_experts))
        self.temperature = float(temperature)
        self.weight = Tensor(
            rng.normal(0.0, 0.02, (n_experts, dim)), requires_grad=True
        )
        self.bias = Tensor(np.zeros(n_experts), requires_grad=True)

    def forward(self, x):
        logits = x @ self.weight.transpose() + self.bias
        if self.temperature != 1.0:
            logits = logits * (1.0 / self.temperature)

        k = self.top_k
        if k >= self.n_experts:
            mask = np.ones(logits.data.shape, dtype=logits.data.dtype)
        else:
            idx = np.argsort(-logits.data, axis=1, kind="stable")[:, :k]
            mask = np.zeros(logits.data.shape, dtype=logits.data.dtype)
            np.put_along_axis(mask, idx, 1.0, axis=1)

        # 未选中项加 -1e9，softmax 后严格为 0；该项为常数，不产生梯度
        gated = logits + Tensor((mask - 1.0) * 1e9, dtype=logits.data.dtype)
        return gated.softmax(axis=-1), mask


class LinkageBridge(Module):
    """联动桥：让同时被激活的专家互相读取对方状态。

    scores = (w·Q)(w·K)ᵀ / sqrt(dh)，未激活专家的 key 被屏蔽；
    H' = H + O·Attention·V。门控权重同时缩放 Q 与 K，因此
    “被选中的强度”直接决定该专家在联动中的话语权。
    """

    def __init__(self, expert_dim: int, bridge_dim: int | None = None,
                 use_gate_scaled_qk: bool = True, mask_inactive: bool = True,
                 seed: int | None = None):
        rng = np.random.default_rng(seed)
        dh = int(bridge_dim or expert_dim)
        self.use_gate_scaled_qk = bool(use_gate_scaled_qk)
        self.mask_inactive = bool(mask_inactive)
        self.dh = dh
        self.q = Linear(expert_dim, dh, seed=int(rng.integers(1 << 30)))
        self.k = Linear(expert_dim, dh, seed=int(rng.integers(1 << 30)))
        self.v = Linear(expert_dim, dh, seed=int(rng.integers(1 << 30)))
        self.o = Linear(dh, expert_dim, seed=int(rng.integers(1 << 30)))
        self.scale = 1.0 / np.sqrt(dh)

    def forward(self, h, probs, mask):
        b, n, _ = h.shape
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        if self.use_gate_scaled_qk:
            w = probs.reshape(b, n, 1)
            q = q * w
            k = k * w

        scores = (q @ k.transpose(0, 2, 1)) * self.scale
        if self.mask_inactive and mask is not None:
            scores = scores + Tensor((mask[:, None, :] - 1.0) * 1e9,
                                     dtype=scores.data.dtype)
        attn = scores.softmax(axis=-1)
        return h + self.o(attn @ v)


class LSELayer(Module):
    """完整 LSE 层。"""

    def __init__(self, dim: int, out_dim: int, n_experts: int = 16,
                 expert_dim: int = 64, top_k: int = 2, bridge_dim: int | None = None,
                 use_bridge: bool = True, use_gate_scaled_qk: bool = True,
                 mask_inactive: bool = True, activation: str = "relu",
                 norm: bool = True, temperature: float = 1.0,
                 seed: int | None = None):
        rng = np.random.default_rng(seed)
        seeds = [int(s) for s in rng.integers(0, 1 << 30, size=4)]
        self.dim = int(dim)
        self.n_experts = int(n_experts)
        self.expert_dim = int(expert_dim)
        self.top_k = int(min(top_k, n_experts))
        self.use_bridge = bool(use_bridge)
        self.experts = ExpertBank(dim, n_experts, expert_dim,
                                  activation=activation, seed=seeds[0])
        self.gate = Gate(dim, n_experts, self.top_k, temperature=temperature,
                         seed=seeds[1])
        self.bridge = (LinkageBridge(expert_dim, bridge_dim,
                                     use_gate_scaled_qk=use_gate_scaled_qk,
                                     mask_inactive=mask_inactive, seed=seeds[2])
                       if use_bridge else None)
        self.norm = LayerNorm(expert_dim) if norm else None
        self.decoder = Linear(expert_dim, out_dim, seed=seeds[3])

    def forward(self, x, return_stats: bool = False):
        probs, mask = self.gate(x)
        h = self.experts(x)                       # [B, N, M]
        if self.bridge is not None:
            h = self.bridge(h, probs, mask)
        b, n, m = h.shape
        agg = (probs.reshape(b, n, 1) * h).sum(axis=1)   # [B, M]
        if self.norm is not None:
            agg = self.norm(agg)
        out = self.decoder(agg)
        if not return_stats:
            return out
        return out, {
            "gate_probs": probs.data,
            "mask": mask,
            "entropy": normalized_entropy(probs.data),
            "entropy_k": normalized_entropy(probs.data, top_k=self.top_k),
            "active": mask.sum(axis=1),
            "expert_usage": probs.data.mean(axis=0),
        }

    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))


def build_default_lse(seed: int = 0) -> LSELayer:
    """手册约定的基线配置：约 9 万参数，远低于 1M 上限。"""
    return LSELayer(dim=64, out_dim=64, n_experts=16, expert_dim=64,
                    top_k=2, bridge_dim=64, seed=seed)