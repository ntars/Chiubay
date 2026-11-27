# 玖白（Chiubay）开发手册 v1.0

> 中文名「玖白」是一个完整的名字；英文名 Chiubay 为威妥玛式音译变体，读 "chew-bay"。
> 本手册以 [边界说明书](docs/00-边界说明书.md) 为最高约束，与当前代码（M0/M1 已实证）同步。
> v0.x 的 PyTorch/gymnasium/MPS 内容已全部作废。

---

## 0. 阅读顺序

| 顺序 | 文档 | 作用 |
|---|---|---|
| 1 | `docs/00-边界说明书.md` | 目标、硬约束、不可行项、G0–G3 |
| 2 | `docs/01-养育大纲.md` | 按发育阶段 S0–S5 的方案 |
| 3 | 本手册 | 怎么写代码、怎么验收 |
| 4 | `docs/02-M0验收记录.md`、`docs/03-M1验收记录.md` | 已验证的结论与踩坑 |

---

## 1. 项目定义与硬约束

| 项 | 约定 |
|---|---|
| 目标 | 人级 AGI 为**方向**，不是当前交付承诺；报告只写“验证了什么” |
| 交付 | L 层：本机机制验证（<10M，上限 1B/极限 3B）；S 层：单机规模阶段 |
| 形态 | 单人全栈自研，每周约 20h，周期 10 年+ |
| 自研边界 | 架构、学习规则、训练循环、评估协议自研；Python/NumPy 仅作数学运算器 |
| 禁用 | torch、TensorFlow、JAX、transformers、sklearn、gymnasium、预训练权重 |
| 基础设施例外 | BLAS 与驱动允许，需 ADR（见 `docs/adr/ADR-001-blas-exception.md`） |
| 本机 | Apple M5 Max / 40 核 GPU / 128 GiB 统一内存 / 无 CUDA、无集群 |

**量级事实**：本机约 10¹⁴ FLOPS；人级基座需 10²⁴–10²⁶ FLOPs，差 4–6 个数量级。这是物理边界，不是优化问题。

---

## 2. 快速开始

```bash
cd "/Users/ntars/Documents/Github/Chiubay"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # 目前只有 NumPy
.venv/bin/python -m unittest discover -s tests -v  # 30 项
.venv/bin/python experiments/m0_xor.py             # XOR 100%
.venv/bin/python experiments/m1_lse.py             # 分化 + 消融，报告入 runs/m1_lse/
```

**环境就绪标准**：`import numpy` 成功；`src/runtime/` 与 `src/child/` 中黑名单 import 为 0。

---

## 3. 开发环境（本机实测基线）

| 项 | 值 |
|---|---|
| 系统 | macOS 27.2.1（26B131），arm64 |
| 芯片 | Apple M5 Max，40 核 GPU，Metal 4 |
| 内存 / 磁盘 | 128 GiB 统一内存 / ~2.6 TiB 可用 |
| Python | venv 内 3.13.12；系统另有 3.15.2、3.9.6 |
| 已装 | NumPy 2.5.5、pip 26.3 |
| GPU 后端 | 无 CUDA；张量栈为单线程 NumPy，多线程留到 S 层 |

> 目录改名后 `.venv/bin/*` 的 shebang 与 `pyvenv.cfg` 会失效。用 `.venv/bin/python -m pip` 可绕过；彻底修复见 §13。

---

## 4. 仓库结构与约定

```
Chiubay/
├── docs/                    边界说明书、养育大纲、里程碑验收、ADR
├── src/runtime/             Tensor、算子、Module、优化器、序列化、损失
├── src/child/               LSE、分化指标、GridWorld、Trainer
├── tests/                   单元 / 梯度 / 吞吐 / 合规 / 分化测试
├── experiments/             可运行实验入口
├── runs/                    实验产物（gitignore）
├── DEVELOPMENT.md           本手册
└── README.md                项目主页
```

约定：
- 模块内不 import 任何 ML 库；CI 由 `tests/test_compliance.py` 扫描。
- 每个实验入口接受 `seed`，结果写 `runs/<name>/report.json`。
- 文档数字必须来自实测，不写估计值。

---

## 5. 核心模块开发指南

### 5.1 Tensor（`src/runtime/tensor.py`）

手写反向传播，NumPy 只负责内存块与底层 BLAS。

接口：`+ - * / @ **`、`relu sigmoid tanh exp log reciprocal softmax`、
`sum mean reshape transpose`、`backward()`、`detach()`、`numpy()`。

规则：
- 广播梯度必须经 `_unbroadcast` 缩回原形状。
- `backward()` 遍历拓扑时跳过 `grad is None` 的常量分支。
- `transpose(axes)` 的反向用**逆置换** `np.argsort(axes)`；原实现只对自逆置换正确，已修。

### 5.2 Module / Optimizer（`src/runtime/nn.py`、`optim.py`）

- `Module.named_parameters()` 遍历 `__dict__`，递归处理 Module 与 list/tuple。
- `Module.__call__(*args, **kwargs)` 透传，LSE 的 `return_stats` 依赖它。
- `SGD`、`Adam` 的动量数组在构造时按参数 dtype 分配；改 dtype 后需重建优化器。
- `LayerNorm` 只用 `mean` / `pow`，不调用外部实现。

### 5.3 LSE 层（`src/child/lse.py`）

```
x [B,D] → ExpertBank → H [B,N,M]
        → Gate       → w [B,N]（Top-K 稀疏，mask 为常数）
        → LinkageBridge → H' [B,N,M]（可选）
        → Σ w·H' → LayerNorm → Decoder → [B,out]
```

| 开关 | 默认 | 说明 |
|---|---|---|
| `top_k` | 2 | **必须 ≥2**。k=1 时门控梯度精确为 0（见 §13） |
| `use_bridge` | False | 单任务无增益（见 §13）；组合任务再验 |
| `use_gate_scaled_qk` | True | 桥的 Q/K 由门控权重缩放；禁止退化成 `attn(H,H,H)` |
| `mask_inactive` | True | 屏蔽未激活专家的 key |
| `norm` | True | 聚合后 LayerNorm |

`forward(x, return_stats=True)` 返回 `(out, stats)`；stats 含
`gate_probs`、`mask`、`entropy`（按 log N）、`entropy_k`（按 log k）、`active`、`expert_usage`。

`build_default_lse()`：88,528 参数，batch 64 约 674 steps/s（含桥）。

### 5.4 分化指标（`src/child/metrics.py`）

- `normalized_entropy(p, top_k=k)`：G0 口径必须按 **log(k)** 归一化。
- `soft_mutual_information(p, labels)`：类条件平均路由分布算 bit 数，不依赖 argmax。
- `routing_consistency(p, labels)`：类内样本与类均值分布的余弦相似度。
- `routing_gain(p, labels)`：一致性减去全局基线。**均匀路由时会虚高到 ~1，必须用这个口径**。
- `summarize_route(p, labels, top_k)`：一次算全部门槛。

**禁止**用 `argmax` 直接算一致性：k≥2 时两个专家概率接近，top1 会随机抖动。

### 5.5 GridWorld（`src/child/gridworld.py`，M2）

- 8×8 网格，1 目标、0–3 障碍、1 可推物体；16 维状态向量。
- 5 个离散动作：上、下、左、右、交互。
- 规则奖励：到达目标 +1；每步 -0.01；撞墙 -0.05；推箱 +0.1；箱到目标 +0.5。
- 接口：`reset() -> np.ndarray(16,)`；
  `step(a) -> (obs, reward, terminated, truncated, info)`；
  `render() -> str`。
- 与 gymnasium 接口相似但**自己实现**，不 import。

### 5.6 Trainer（`src/child/trainer.py`，M2）

- REINFORCE：一个 episode 一条轨迹，episode 末按折扣回报更新。
- `log_probs` 必须是 Tensor（梯度入口）；回报用 NumPy 计算后作常数。
- 需要 `runtime/losses.py` 的 `log_softmax`；采样动作用 numpy `random.choice`。
- 每 N 个 episode 记录：平均回报、专家熵、MI、一致性、吞吐。

---

## 6. 配置与实验管理

实验配置写在 `experiments/*.py` 顶部常量，不引入配置框架。
每次实验必须记录：seed、步数、batch、学习率、模型开关、参数量、吞吐、指标。
结果写 `runs/<name>/report.json`，`runs/` 已 gitignore。

---

## 7. 开发工作流

1. 改代码前先加测试；梯度相关改动必须过有限差分。
2. 跑 `.venv/bin/python -m unittest discover -s tests -v`。
3. 跑对应 `experiments/*.py`，数字写入 `docs/0N-Mx验收记录.md`。
4. 新决策写 ADR 或更新 `docs/00-边界说明书.md`。
5. 禁止跳阶段：M(n) 未验收，不开 M(n+1) 的正式任务。

---

## 8. 测试规范

| 层 | 内容 |
|---|---|
| 算子 | 有限差分 vs 解析梯度，float64，误差 <5e-4 |
| 模块 | 形状、稀疏性、开关消融、统计口径 |
| 合规 | `src/` 黑名单 import = 0 |
| 吞吐 | 固定 batch 与 seed，报告 steps/s |
| 分化 | 合成簇训练前后 MI/一致性对比 |
| 复现 | 5 seeds（G0 要求，M2 补齐） |

---

## 9. 里程碑与验收

| 里程碑 | 状态 | 内容 |
|---|---|---|
| M0 自研栈 | ✅ | Tensor/Module/Adam/checkpoint；10 项测试；XOR 100% |
| M1 LSE 核心 | ✅ | Expert/Gate/Bridge/指标；20 项测试；分化 MI 0.53→0.97 bit |
| M2 沙盒与训练 | ✅ | GridWorld + REINFORCE + S0 端到端 + 5 seeds；50 项测试 |
| M3 语言与组合 | ⏳ | 序列模块 + 组合任务 + 桥的再验 |
| M4 自主改写 | ⏳ | 影子副本、分裂/合并、审计 |

### G0 门槛（当前对照）

| 条件 | 实测 | 状态 |
|---|---|---|
| 黑名单 import = 0 | 通过 | ✅ |
| 归一化熵 0.4–0.9（按 log k） | 0.583 | ✅ |
| MI > 0.1 bit | 0.968 | ✅ |
| 路由增益 > 0.1 | 27.2% | ✅ |
| 吞吐 ≥200 steps/s（含沙盒） | 6,111（含环境+反向） | ✅ |
| 5 seeds 可复现 | IQM -0.021，std 0.31 | ❌ M2.1 修复 |
| 参数 <1M | 88,528 | ✅ |

G0 尚未通过。唯一阻塞项：5 seeds 方差过大，见 `docs/04-M2验收记录.md` §4.3。

### 阶段表（与养育大纲对应）

| 阶段 | 焦点 | 进入条件 |
|---|---|---|
| S0 感知运动 | 8×8 网格、目标导向、专家分化 | G0 通过 |
| S1 多模态 | 颜色/形状/声音关联 | S0 泛化达标 |
| S2 语言萌芽 | 符号映射、指令执行 | 序列模块通过消融 |
| S3 社交 | 同伴、轮流、合作 | 能力矩阵 ≥2 级 |
| S4 自主目标 | 好奇心、自设目标 | 探索覆盖率达标 |
| S5 自我修改 | 影子副本、架构改写 | 回滚 100%、审计 100% |

S6（人级基座）不做；受物理边界限制，在说明书中明确标注。

---

## 10. 安全与对齐开发规范

- 硬约束写进奖励与 Guardian，不得通过配置关闭。
- 任何架构自修改先在影子副本验证，禁止直接改主模型。
- 修改需记录：改了什么、为什么、回滚快照、验证结果。
- Parents 为纯 API 时，数据脱敏后可出本机；其参数不得进入 Child。

---

## 11. 监控与可解释性

每个 episode 至少记录：回报、步数、专家使用分布、门控熵、MI、一致性。
每 100 episode 输出一次文本渲染的网格，人工确认行为没有奖励黑客。
所有指标写入 JSON，禁止只打终端日志。

---

## 12. 性能预算（实测）

| 配置 | steps/s |
|---|---|
| 默认 LSE，batch 64，含桥 | 674 |
| S0 训练（含环境+反向，batch 1） | 6,111 |
| 8 专家×16 维，无桥 | 3,912 |
| 8 专家×16 维，含桥 | 1,604 |
| 小 MLP（M0） | 18,590 |

G0 的 200 steps/s 是**含环境步进**的口径，必须用 M2 实测替换。

---

## 13. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `k=1` 门控完全不更新 | softmax 为 one-hot，jacobian `s(δ-s)` 恒 0 | 用 `k≥2`；必要时 straight-through |
| 联动桥比无桥更差 | 单任务无需跨专家通信，桥引入噪声 | 默认 `use_bridge=False`，组合任务再验 |
| 一致性虚高到 99% 但 MI 接近 0 | 路由完全均匀，样本与类均值同向 | 改看 `routing_gain`，见 §5.4 |
| 5 seeds 中部分不收敛 | REINFORCE 无 baseline、地图每轮随机 | M2.1：advantage + 固定地图课程 |
| 熵永远不进 0.4–0.8 | 按 log(N) 归一化且 k 小 | 改按 log(k) |
| 换位后 `.venv/bin/pip` 报错 | shebang 指向旧目录 | `.venv/bin/python -m pip` 可用；或批量替换 `.venv/bin/*` 与 `pyvenv.cfg` 里的旧路径 |
| 梯度有限差分不通过 | 常量分支 grad 为 None、transpose 用错逆置换 | 见 §5.1 |

---

## 14. 术语表

| 术语 | 含义 |
|---|---|
| LSE | 分层技能集成：专家 + 门控 + 联动桥 |
| 专家 | 一个把状态映射为技能向量的子网络 |
| 门控 | 选 Top-K 专家并给权重的路由 |
| 联动桥 | 让被共同激活的专家交换信息的注意力 |
| 分化 | 不同专家处理不同状态，用 MI 与一致性度量 |
| G0 | 第一道验收门：合规、分化、吞吐、参数、复现 |
| S0–S5 | 发育阶段；S6 不在范围内 |
| L 层 / S 层 | 本机机制验证 / 单机规模阶段 |

---

## 15. 下一步（M2.1 执行顺序）

1. trainer 加价值基线，用 advantage `G_t − V(s_t)` 替代原始回报。
2. GridWorld 支持 `fixed_layout`，前 100 episode 固定地图做课程学习。
3. 重跑 5 seeds：目标 IQM 回报 > 0、std < 0.2、成功率 > 90%。
4. 补“必须推箱才能到目标”的课程，用到目前闲置的交互动作。
5. 再开 M3：序列模块 + 组合任务 + 联动桥再验。

> 我们不是在造神，我们是在养一个孩子。他可能跌倒，可能犯错，但每一步都是真实的进步。
