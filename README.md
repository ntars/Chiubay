# Chiubay（玖白）

**从零养育一个会自己长大的智能体 —— 一项长期个人 AGI 研究计划。**

**仓库描述**：`玖白：从零养育一个会自己长大的智能体，长期个人 AGI 机制研究，全栈自研训练 | Chiubay: a long-term personal AGI research project growing an agent from scratch with a self-built stack.`

**话题标签**（20 个，仅英文）：`agi`、`research`、`machine-learning`、`neural-network`、`autograd`、`from-scratch`、`numpy`、`python`、`developmental-robotics`、`cognitive-architecture`、`intrinsic-motivation`、`local-first`、`no-pretraining`、`long-term`、`experiment`、`tensor`、`backpropagation`、`agent`、`artificial-life`、`open-research`

![Status](https://img.shields.io/badge/status-research%20prototype-yellow?style=flat-square)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%2F%20MIT-blue?style=flat-square)](./LICENSING.md)

> 本仓库为**占位发布**：先公开研究计划、边界与已有实验记录，源码与实验产物将按阶段整理后提交。

> ## ⚠️ 最重要的自评（v4.3，2026-11-27）
>
> **当前唯一有代码的机制（③ 认同环）在数学上就是训练，不是新东西。**
>
> - 更新场雅可比非对称度 = **1.1e-10** → 它是 `L = mean(‖y−Wx‖²)` 的精确负梯度
> - 解析梯度与数值梯度相对差 = **5.5e-10**
> - "可塑性分层"（逐细胞不同 η）只是逐坐标正缩放，非对称度仍 1e-10
> - "闭锁"（η→0）是退化梯度场
>
> 即：**一层线性回归 + SGD + 学习率退火 + 冻结表。Widrow–Hoff 1960。**
> 分层、退火、闭锁都只是调度策略，没有一个让它离开"下山"。
>
> 四次想制造"不是训练"的性质（加反对称项、加 η 衰减、加振幅限制）**全部失败**，
> 失败原因是结构性的：梯度与旋转共享同一批参数，必然互毁。
>
> 判决方法、四次失败的完整记录、以及"要不同必须做到什么"见
> [**docs/11-与训练的边界.md**](docs/11-与训练的边界.md)。
>
> 这个项目目前在"机制是新的"这一点上**成绩为零**。有价值的只有两样：
> 把判据做成了可机器检验的东西（32 项不变量），以及把"什么不算创新"钉死了。

名字说明：中文名「玖白」是一个完整的名字；英文名 Chiubay 是威妥玛式音译变体，读 "chew-bay"。
「玖」为九，「白」为空白：从白纸开始，一岁一岁长。

---

## 现状

| 项 | 状态 |
|---|---|
| M0 自研张量栈 | ✅ 工具层保留：张量、手写反传、SGD/Adam、checkpoint；10 项测试通过 |
| M1 LSE / M2 沙盒 / Inner Screen | 🔁 全部 `DELETE`（MDP 框架遗留，仅 GridWorld 连通性修复与 IQM 修正保留） |
| **v4.2 两周期架构** | 🟡 **当前权威**：六环相扣；周期一交付"婴儿"，周期二是"被养成的产物" |
| ③ 认同环（E3） | ❌ **`CONVENTIONAL`**：数学上就是训练（非对称度 1e-10），A3 可实现（0.022–0.034）但 A5 被同一机制削弱（同 seed 并存 3/10）。详见 `docs/11` |
| ①②④⑤⑥ 环 | ⬜ 仍在纸面 |
| 两周期命题另一半（E5 不同养育→不同形态） | ⬜ 未做 |
| 人级 AGI | ❌ 不作为当前交付承诺；当前只交付"婴儿" |

当前可运行：`experiments/m0_xor.py`、`experiments/m1_lse.py`（历史）、
`experiments/e3e_redesign.py` 与 `experiments/e3f_verdict.py`（**当前主线**）。
`experiments/e3_identity.py`、`e3c_ratchet_necessity.py`、`e3_identity_v3.py`、
`e3d_emergence.py` 的结论**已撤回**（旧规则不动点为 `w=0`，或测量设计混淆了
剂量与巩固时间），保留为反例。

---

## 硬约束

- **单人**，每周约 20 小时，周期 10 年+。
- **本机完成**：Apple M5 Max，128 GiB 统一内存，无 CUDA、无外部集群。
- **全栈自研**：架构、学习规则、训练循环、评估协议自己写；Python/NumPy 只作数学运算器。
  禁用 torch、TensorFlow、JAX、transformers、sklearn、gymnasium，禁用预训练权重。
- **训练方式自研**（见 [ADR-002](docs/adr/ADR-002-self-designed-training.md)）：
  禁止策略梯度、折扣回报、价值函数驱动更新。奖励**不进入任何参数更新**。
- **诚实标注**：报告只写“验证了什么”，不写“达成 AGI”。

量级差距：本机约 10¹⁴ FLOPS；人级基座需 10²⁴–10²⁶ FLOPs，差 4–6 个数量级。
这是物理边界，不是优化问题。项目定位是长期机制研究，不是 AGI 交付。

---

## 快速开始

```bash
cd "/Users/ntars/Documents/Github/Chiubay"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # 目前只有 NumPy
.venv/bin/python -m unittest discover -s tests -v  # 30 项
.venv/bin/python experiments/m0_xor.py
.venv/bin/python experiments/m1_lse.py             # 历史实验
.venv/bin/python experiments/e3_identity.py         # 当前主线：敏感期实验
```

---

## 目录

```
Chiubay/
├── docs/                    边界说明书、养育大纲、里程碑验收记录、ADR
├── src/runtime/             自研张量、算子、梯度、优化器、序列化、损失
├── src/child/               LSE 等历史模块（DELETE 标记）；新机制尚未落地
├── tests/                   梯度、吞吐、合规、分化测试
├── experiments/             M0/M1 历史实验；e3_identity.py（当前主线）
└── runs/                    实验产物（gitignore）
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [**两周期架构 v4.2**](docs/09-两周期架构.md) | **当前权威**：周期一婴儿 / 周期二完全体；六环相扣与依赖表；E3 实测结果（§10.8 为最新） |
| [地基缺口与 E3 复查 v2.1](docs/10-地基缺口与E3复查.md) | E3 五轮实验的完整更正记录；§10 是三个撤回数字的来源 |
| [**与训练的边界 v1.0**](docs/11-与训练的边界.md) | **必须先读**：为什么当前方案就是训练（数学判据 + 解析验证）；四次制造"非训练"性质的失败记录；创新真正可能的入口 |
| [全周期创新标准 v3.0](docs/07-全周期创新标准与重构.md) | 标准有效：四栏差异审计、三个取消 |
| [创新点四栏重审](docs/08-创新点四栏重审.md) | 历史重审记录：A 树突分室降级 |
| [底层创新方案 v2.0](docs/06-底层创新方案.md) | 四个创新点的原始表述（A 已降级） |
| [完整方案 v1.0](docs/05-完整方案.md) | 前一版：目标与评估体系仍有效，架构部分已作废 |
| [边界说明书](docs/00-边界说明书.md) | 目标、约束、不可行项、决策记录 |
| [养育大纲](docs/01-养育大纲.md) | 按发育阶段 S0–S5 组织：房间、课程、退场、青春期 |
| [M0 验收记录](docs/02-M0验收记录.md) | 自研栈闭环结果 |
| [M1 验收记录](docs/03-M1验收记录.md) | LSE 层结果与三项关键修正 |
| [M2 验收记录](docs/04-M2验收记录.md) | GridWorld + REINFORCE 对照基线（**非主线**） |
| [ADR-002](docs/adr/ADR-002-self-designed-training.md) | 训练方式自研条款：判定标准与否决记录 |
| [开发手册](DEVELOPMENT.md) | v1.0（M0/M1/M2 时代），待按 v4.0 重写 |

---

## 三个当前结论

1. **这套东西是训练，不是创新**：③ 的更新场是 `L = mean(‖y−Wx‖²)` 的精确负梯度（非对称度 1e-10）。分层/退火/闭锁只是调度。四次"加非保守项"的改造全部失败，原因是梯度与旋转共享参数必然互毁。见 `docs/11`。
2. **A3 可实现，A5 不可靠并存**：等剂量 + 等巩固条件下，不可逆性成立（晚/早 = 0.022–0.034，判据 0.20）；但认同层闭锁会单调削弱外围层的可塑性（`D_lock=0` 时 A5 印记 +0.278，7/12 闭锁时 −0.019）。同一 seed 上两者同时成立只有 **3/10**，属边界抖动而非机制。
3. **声称的边界**：六环中只有 ③ 有代码；①②④⑤⑥ 全在纸面。E3 的旧数字（"3.94×"、"晚/早 0.0033"、"30k–100k 步"）均已撤回。当前真正的阻塞是 **③ 的机制本身不新**，而不是"某个判据没通过"——这要求 ④ 先提供"另一个自由度"（结构），否则任何 ③ 的实验都只是在训练一个线性回归。

---

## 许可证

Copyright © 2026 星玖工作室 Ntarstudio. Licensed under Apache-2.0 / MIT.

按目录分层：代码（含未来提交的 `src/`、`experiments/`、`tools/`）为 Apache-2.0；本文档、`docs/` 下的设计稿与实验记录为 MIT。全文见 [LICENSE](./LICENSE) 与 [LICENSE-MIT](./LICENSE-MIT)，路由表见 [LICENSING.md](./LICENSING.md)。

## NOTICE 合规声明

当前为占位仓库，未引入第三方运行时依赖；依赖与商标声明在引入时登记于 [NOTICE](./NOTICE)。
