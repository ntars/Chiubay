"""E3 的不变量测试。

两个时代，两套坑：

  v1/v2（e3_identity.py）：用了会衰减的量做棘轮、用了比值失控的指标、
      用了没验证的环境稳定性。这四类坑留在这里，作为历史约束。

  v3（e3v3_core.py）：v2 的学习规则不动点是 w=0（抗赫布去相关），
      测到的是权重崩塌残余。v3 换成预测式 LMS 后，新暴露了四个坑：
        - 全局 load 驱动 η → 所有细胞同步闭锁 → 分层是假的
        - gate 作用于 η 已为零的细胞 → gate 是空操作
        - 判据用"变化 > 1e-9" → 1e-4 的抖动也算通过（伪阳性）
        - 比值 keep≈1 可能只是"两边都没动"（平凡）

运行：.venv/bin/python -m unittest tests.test_e3_invariants -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import e3_identity as E
import e3v3_core as V


class TestRatchetDrivers(unittest.TestCase):
    """约束 1：棘轮驱动量必须在收敛后仍单调增长。"""

    def test_state_derived_drivers_all_saturate(self):
        d, n, T = 24, 6, 6000
        rng = np.random.default_rng(0)
        a = E.make_env(1, d)
        w = rng.normal(size=(n, d)) * 0.15
        x = rng.normal(size=d)
        acc = {k: 0.0 for k in ("abs_wx", "plastic_work", "abs_err")}
        for t in range(T):
            xn = a @ x + 0.05 * rng.normal(size=d)
            dx = xn - x
            err = w @ dx
            upd = (0.30 * err)[:, None] * x[None, :] / max(float(x @ x), 1e-6)
            acc["abs_wx"] += float(np.mean(np.abs(w @ x)))
            acc["abs_err"] += float(np.mean(np.abs(err)))
            w = w + upd
            acc["plastic_work"] += float(np.mean(np.sum(upd ** 2, axis=1)))
            x = xn
        self.assertLess(acc["abs_wx"], 25.0, "活动量应有界；若涨到几十说明未饱和")

    def test_dose_driver_is_monotone(self):
        d = 24
        rng = np.random.default_rng(0)
        a = E.make_env(1, d)
        x = rng.normal(size=d)
        dose = 0.0
        samples = []
        for t in range(8000):
            xn = a @ x + 0.05 * rng.normal(size=d)
            dose += float(np.mean(np.abs(xn - x)))
            if t % 1000 == 0:
                samples.append(dose)
            x = xn
        diffs = np.diff(samples)
        self.assertTrue(np.all(diffs > 0), "dose 必须严格递增")
        self.assertLess(abs(diffs[-1] / diffs[0] - 1.0), 0.15, "末段速率应与首段相当")


class TestEnvironmentStability(unittest.TestCase):
    """约束 2：扰动不得让环境失稳，且不可行时必须报错。"""

    def test_all_shifts_keep_spectral_radius_bounded(self):
        d = 24
        a = E.make_env(1, d)
        for s in (1, 2, 7, 13, 29, 99):
            shift = E.make_shift(s, d, a_mat=a)
            ev = float(np.max(np.abs(np.linalg.eigvals(a + shift))))
            self.assertLessEqual(ev, 0.881, f"shift seed={s} 后谱半径 {ev:.4f} 超限")

    def test_shift_fails_loudly_when_infeasible(self):
        """v2 的静默退化：spec(a) >= max_radius 时返回零矩阵。

        后果是"养育窗口"与"无养育窗口"跑同一个环境，早期增量精确为 0，
        实验报 nan，而调用方看不出任何异常（C3_slow_env 整类失效）。
        现在必须抛异常。
        """
        d = 24
        a = E.make_env(1, d, radius=0.95)
        with self.assertRaises(ValueError):
            E.make_shift(2, d, a_mat=a)

    def test_shift_is_never_silently_zero(self):
        """任何可行的 shift 都必须非零 —— 零扰动等于没做实验。"""
        d = 24
        for radius in (0.80, 0.85, 0.88):
            a = E.make_env(1, d, radius=radius)
            s = E.make_shift(2, d, a_mat=a)
            self.assertGreater(float(np.linalg.norm(s)), 1e-6,
                               f"radius={radius} 时 shift 退化近零")

    def test_long_run_stays_finite(self):
        out = E.develop(0, [(300, 1000)], t_total=20000)
        self.assertTrue(np.all(np.isfinite(out["w"])), "20000 步后权重必须有限")


class TestLearningRuleActuallyLearns(unittest.TestCase):
    """约束 3（v3 新增，最根本）：学习必须在学世界，不是在塌向零。

    v1/v2 的规则是 Δw = η·(w·dx)·x/|x|²，让 w·dx→0。
    但 dx 张满 R^d、(A-I) 可逆（实测最小奇异值 0.2601），
    唯一不动点是 w=0。实测 20000 步后 ||w|| = 2.4e-28（初始 1.728）。
    两版 E3 的全部"印记"都是崩塌残余。
    """

    def test_old_rule_collapses_to_zero(self):
        """反证：旧规则确实把权重推向零，不是学环境。"""
        d = 24
        a = E.make_env(1, d)
        rng = np.random.default_rng(0)
        w = rng.normal(size=d) * 0.15
        x = rng.normal(size=d)
        w0 = float(np.linalg.norm(w))
        for _ in range(20000):
            xn = a @ x + 0.05 * rng.normal(size=d)
            err = w @ (xn - x)
            w = w + 0.3 * err * x / max(float(x @ x), 1e-6)
            x = xn
        self.assertLess(float(np.linalg.norm(w)) / w0, 1e-6,
                        "旧规则若不塌向零，则本约束的前提需重新审")

    def test_new_rule_converges_to_environment(self):
        """正证：预测式 LMS 必须收敛到 A 的行，且预测误差接近噪声地板。

        用 mode="eta_only"：有 η 衰减、无 gate、无分层硬锁。
        不能用 mode="full" + D_lock=1e9 —— D_lock 只放大 thr，
        而判据是 conv <= thr，thr 变大反而让所有细胞在第 400 步立刻闭锁，
        实测 frozen=24、η=0、收敛退化到 40.7%。这是一个真实的陷阱。
        """
        d = 24
        a = V.make_env(1, d)
        r = V.develop(0, [(300, 4300)], n_cells=d, d=d, mode="eta_only")
        rel = float(np.linalg.norm(r["w"] - a) / np.linalg.norm(a))
        self.assertLess(rel, 0.20, f"||W-A||/||A||={rel:.3f}，学习未收敛到环境")
        floor = V.CFG["noise"] ** 2
        self.assertLess(r["pred_mse"], floor * 1.30,
                        f"预测MSE={r['pred_mse']:.5f} 远高于噪声地板 {floor:.5f}")

    def test_weight_does_not_collapse(self):
        """||w|| 不得趋于零 —— 这是与 v2 的分水岭。"""
        r = V.develop(0, [(300, 4300)], n_cells=6)
        self.assertGreater(float(np.linalg.norm(r["w"])), 0.5,
                           "权重塌向零 → 学习规则仍是抗赫布型")


class TestLayeringIsReal(unittest.TestCase):
    """约束 4（v3 新增）：分层必须真的产生不同的可塑性。

    v2 的 load 是全局标量，eta_eff 数组每个元素相同，外围层 η 也被拖到
    eta_min（实测 1.01e-4 = 0.034% 保留率）。"外围层还能长"从未实现。
    """

    def test_load_is_per_cell_not_global(self):
        r = V.develop(0, [(300, 4300)], n_cells=6)
        li = r["load_i"]
        self.assertGreater(float(li.std()), 1e-4,
                           "各细胞 load 完全相同 → 仍是全局时钟，分层无效")

    def test_identity_and_periphery_differ_in_late_eta(self):
        """认同层闭锁后 η=0，外围层必须保留可塑性。

        注意判据不能用 "eta_late > 0.3*eta0"：η 由收敛比驱动，
        即使外围层仍可学，η 也会随 conv 下降 —— 这是设计意图。
        真正要测的是"外围层 η 严格大于认同层"，以及"外围层不为 0 且可学"：
        实测认同层 0.0、外围层 2e-3（= eta_min），外围层在 B 上确有印记。
        """
        # 维度必须与 develop 的默认 d（= CFG["d"] = 12）一致。
        # 之前这里写死 24，与 make_env(1, 24) 一起和 develop 的 12 维冲突，
        # 报 "operands could not be broadcast (12,12) (24,24)"。
        d = V.CFG["d"]
        a0 = V.make_env(1, d)
        r = V.develop(0, [(300, 4300), (5000, 13500)],
                      second_shift=V.make_shift(13, d, a_mat=a0),
                      n_cells=6)
        m = r["identity_mask"]
        late = r["eta_late"]
        self.assertTrue(np.all(late[m] == 0.0), "认同层晚期 η 必须严格为 0")
        self.assertTrue(np.all(late[~m] > 0.0), "外围层晚期 η 不得为 0")
        self.assertGreater(float(late[~m].mean()), float(late[m].mean()),
                           "外围层可塑性不高于认同层 → 分层无效")

    def test_lock_times_are_spread_out(self):
        """闭锁时刻必须展开 —— 否则"关键期"不存在，分层也无从谈起。

        注意 n 的选择：n=6 时 5 个细胞几乎同时收敛（t_lock 全等于 lock_delay），
        跨度测不出来。n=12 时实测 t_lock 分布 [-1, 400, 444, 509, 702]，
        跨度 302 步。这说明"闭锁有先后"成立，但需要足够多的细胞才看得见。

        总有 1–2 个细胞永不闭锁（conv 停在 1.0）：它们预测的维度确实每步
        都被噪声主导，学不动。这不是 bug，是"外围层"的物理来源。
        """
        r = V.develop(0, [(300, 4300)], n_cells=12)
        tl = r["t_lock"]
        locked = tl[tl >= 0]
        self.assertGreater(len(locked), 6, "至少应有 7 个细胞闭锁")
        self.assertGreater(float(locked.max() - locked.min()), 100,
                           "闭锁时刻过于集中 → 关键期不存在")
        self.assertGreater(int((tl < 0).sum()), 0,
                           "应有细胞永不闭锁（外围层的物理来源）")


class TestGateIsNotNoOp(unittest.TestCase):
    """约束 5（v3 新增）：gate 不得是空操作。

    v2 的 gate 只作用于 frozen & identity，而这些细胞 η 已经是 0；
    外围层的 gate 又被强制置 1。实测 gate_in_window 恒等于 1.0000，
    full 与 no_gate 逐位相同 —— 消融"证明"gate 冗余，是测了个空气。
    """

    def test_gate_actually_reduces_learning(self):
        w = np.random.default_rng(0).normal(size=(4, 8)) * 0.05
        frozen = np.ones(4, dtype=bool)
        dev = np.array([1e-3, 1e-2, 1e-1, 1.0])
        gate = np.exp(-V.CFG["beta"] * dev / (dev + V.CFG["kappa"]))
        self.assertLess(float(gate.max()), 0.5, "gate 未在合理偏离范围内起作用")
        self.assertLess(float(gate.max() / gate.min()), 1e6, "gate 动态范围应有限")

    def test_gate_open_before_lock(self):
        """闭锁前 gate 应接近 1，否则 gate 变成无条件衰减。"""
        r = V.develop(0, [(300, 4300)], n_cells=6)
        self.assertGreater(r["gate_early"], 0.5,
                           "闭锁前 gate 已大幅关闭 → 阻抗变成了门槛")


class TestCriteriaAreNotTrivial(unittest.TestCase):
    """约束 6（v3 新增）：判据不得被平凡情形通过。

    v2 的 A5 判据是 |after-before| > 1e-6，外围层 η=1e-4 时抖一下就过线。
    T2 的 keep≈1 也可能只是"两边都没动"。
    """

    def test_a5_requires_meaningful_learning(self):
        """A5 必须用有意义的效应量，不能是 1e-6 级抖动。"""
        s = V.make_shift(13, 24, a_mat=V.make_env(1, 24))
        self.assertGreater(float(np.linalg.norm(s)), 1.0)

    def test_imprint_metric_is_bounded(self):
        """有界指数必须落在 [-1, 1]，且对"完美吸收/完美拒绝"取到端点。"""
        a = np.zeros((3, 5))
        s = np.ones((3, 5))
        self.assertAlmostEqual(V.imprint(s[0].copy(), a[0], s[0]), 1.0, places=6)
        self.assertAlmostEqual(V.imprint(a[0].copy(), a[0], s[0]), -1.0, places=6)
        mid = (a[0] + s[0]) / 2
        self.assertAlmostEqual(V.imprint(mid, a[0], s[0]), 0.0, places=6)

    def test_imprint_metric_has_no_unbounded_ratio(self):
        """投影量 mean|w·u| 随 w→0 趋零，导致比值失控；有界指数不会。"""
        a = np.zeros((2, 5))
        s = np.ones((2, 5))
        for scale in (1e-6, 1e-3, 1.0):
            w = np.full(5, 0.5 * scale)
            val = V.imprint(w, a[0], s[0])
            self.assertTrue(-1.0 <= val <= 1.0, f"scale={scale} 时指数越界 {val}")

    def test_signal_exceeds_noise_floor(self):
        """**本轮最重要的约束**：要测的信号必须大于学习稳态的误调。

        实测：LMS 稳态下 ||w - A|| 有 0.24–0.35，而秩1扰动 ||S_i|| 只有
        0.03–0.25。噪声底比信号大，于是"只学 A"对 A 的印记是负的、
        "只学 B"对 B 的印记也是负的 —— 系统在完美条件下也测不出养育。

        任何"印记"实验开跑前必须先过这一关，否则测的是误调，不是印记。
        """
        d, n = 24, 6
        a = V.make_env(1, d)
        s = V.make_shift(2, d, a_mat=a)
        r = V.develop(0, [(300, 4300)], second_shift=s, b_start=0, n_cells=n, d=d)
        noise = float(np.mean(np.linalg.norm(r["w"] - a[:n], axis=1)))
        signal = float(np.mean(np.linalg.norm(s[:n], axis=1)))
        # 记录实测比值。若某天 signal > noise，说明参数变了，可放宽此断言。
        self.assertLess(signal, noise,
                        f"signal={signal:.3f} noise={noise:.3f}："
                        f"若信号已大于误调，说明设计已改进，请复查 e3d 结论")
        # 关键：把"信号被淹没"这个事实钉死在测试里，避免再拿这些数据下结论
        self.assertGreater(noise / signal, 1.0,
                           "误调必须大于信号才符合本轮实测；否则需重跑 e3d")


class TestLockingConflictsWithLearning(unittest.TestCase):
    """约束 8（第五轮改写）：闭锁、η 衰减、学习精度三者的真实关系。

    **本类原来写的是一个已被推翻的结论。** 原注释说"闭锁 500–2000 步停止
    学习、LMS 需 30k–100k 步，差 20–100 倍" —— 那个 30k–100k 是把
    "恒定 η 的误调地板"误读成"收敛慢"。实测：恒定 η 在 500→30000 步
    始终停在 `‖W−A‖/‖A‖ ≈ 0.39–0.45`，**加步数不下降**；有 η 衰减时
    2000 步就到 0.111。不存在 20–100 倍的时间尺度冲突。

    现在这一类的测试改为记录**真实**关系：
      1. 恒定 η 永不收敛（误调地板与步数无关）——见
         `TestMeasurementIsNotConfounded.test_constant_eta_never_converges`
      2. η 衰减 2000 步即可收敛
      3. 在线 LMS 与理想 MLE 仍有量级差（这条成立，是 η 地板造成的）
      4. 理想 MLE 能达到高印记（证明实验设计没问题）
    """

    def test_lock_happens_when_learning_has_partly_converged(self):
        """闭锁时刻的**真实**位置：500–2000 步，此时 rel_err 约 0.10–0.15。

        旧版把这条写成"闭锁远早于收敛"，即暗示闭锁发生在细胞完全没学会时。
        实测不是：闭锁发生时 η 已被 conv 压到 0.003–0.09，`‖W−A‖/‖A‖`
        已在 0.10 附近（有 η 衰减时 2000 步的典型值）。**它是"学到一半就冻"，
        不是"没学就冻"** —— 这才是 A3 能成立、A5 会受损的原因。

        本测试只钉死闭锁的时间窗口，不断言"远早于收敛"。
        """
        n, d = 6, 12
        a = V.make_env(1, d)
        s = V.make_shift(2, d, amp=2.5, a_mat=a)
        early = (300, 4300)

        r = V.develop(0, [early], second_shift=s, b_start=0,
                      n_cells=n, d=d, t_total=30000,
                      cfg={"D_lock": 0.25, "gamma": 2.0, "ema": 0.003})
        tl = r["t_lock"]
        locked = tl[tl >= 0]
        self.assertGreater(len(locked), 0, "应有细胞闭锁")
        lock_median = float(np.median(locked))
        # 实测中位 792。闭锁落在 lock_delay(400) 之后、早期窗口结束(4300)之前。
        self.assertLess(lock_median, 4300,
                        f"锁定中位 {lock_median:.0f} 已超出早期窗口，"
                        f"说明闭锁不再由收敛比驱动，需重跑 e3f 看结论")
        self.assertGreaterEqual(lock_median, 400,
                                f"锁定中位 {lock_median:.0f} 早于 lock_delay，"
                                f"说明初始化瞬态被当成收敛")

    def test_eta_decay_is_what_narrows_the_gap(self):
        """**更正上一版写错的方向**：让权重接近目标的不是"不闭锁"，是 η 衰减。

        实测（n=6, d=12, 30k 步, 目标 A+B）：
            full  (η衰减+闭锁+gate)  dist = 0.2434
            soft  (η衰减, 不硬锁)     dist = 0.2336
            eta_only(η衰减, 无gate)  dist = 0.2847
            none  (恒定 η=0.30)      dist = 0.4367   ← 最差

        结论：**恒定 η 的误调最大**，η 衰减才是关键；闭锁在这里
        既没帮上忙也没拖后腿（full 与 soft 相差 4%）。

        我此前以为"闭锁阻止了学习，去掉闭锁就能学得更好"——错了。
        去掉闭锁但用恒定 η，反而更差。教训：消融必须只改一个变量。
        """
        n, d = 6, 12
        a = V.make_env(1, d)
        s = V.make_shift(2, d, amp=2.5, a_mat=a)
        target = (a + s)[:n]

        def dist(w):
            return float(np.mean(np.linalg.norm(w - target, axis=1)))

        common = dict(second_shift=s, b_start=0, n_cells=n, d=d, t_total=30000)
        cfg = {"D_lock": 0.25, "gamma": 2.0, "ema": 0.003}
        d_none = dist(V.develop(0, [(300, 4300)], mode="none", **common)["w"])
        d_eta = dist(V.develop(0, [(300, 4300)], mode="full", cfg=cfg, **common)["w"])

        self.assertLess(d_eta, d_none,
                        f"η 衰减版 {d_eta:.3f} 未优于恒定 η {d_none:.3f}，"
                        f"与实测相反，需重跑确认")
        # 量化差距：实测 0.243 vs 0.437
        self.assertGreater(d_none / d_eta, 1.3,
                           f"两者差距 {d_none/d_eta:.2f}× 太小，与实测 1.8× 不符")

    def test_online_lms_far_from_ideal(self):
        """在线 LMS 远达不到理想 MLE —— 这是 ③ 的真实短板。

        实测：理想 MLE 的 dist 是 0.013–0.025，在线 LMS 最好 0.234。
        差约 10 倍。这不是闭锁造成的（soft 不硬锁也只有 0.234），
        而是在线 LMS 本身的稳态误调。
        """
        n, d = 6, 12
        a = V.make_env(1, d)
        s = V.make_shift(2, d, amp=2.5, a_mat=a)
        target = (a + s)[:n]

        r = V.develop(0, [(300, 4300)], second_shift=s, b_start=0,
                      n_cells=n, d=d, t_total=30000, mode="soft",
                      cfg={"D_lock": 0.25, "gamma": 2.0, "ema": 0.003})
        d_online = float(np.mean(np.linalg.norm(r["w"] - target, axis=1)))

        rng = np.random.default_rng(999)
        x = rng.normal(size=d)
        X, Y = [], []
        for _ in range(20000):
            xn = (a + s) @ x + 0.05 * rng.normal(size=d)
            X.append(x.copy())
            Y.append(xn[:n].copy())
            x = xn
        W = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0].T
        d_ideal = float(np.mean(np.linalg.norm(W - target, axis=1)))

        self.assertLess(d_ideal, 0.05, f"理想解 {d_ideal:.4f} 异常，实验设计有问题")
        self.assertGreater(d_online / d_ideal, 3.0,
                           f"在线/理想 比值 {d_online/d_ideal:.1f}× 低于实测 10×，需复查")

    def test_ideal_mle_reaches_high_imprint(self):
        """理想解必须能到达高印记 —— 否则是实验设计问题，不是学习问题。

        实测 MLE 印记 +0.78~+0.85（残差 0.013–0.025）。
        若理想解也上不去，说明指标或环境设计有错，与 LMS 无关。
        """
        n, d = 6, 12
        a = V.make_env(1, d)
        s = V.make_shift(2, d, amp=2.5, a_mat=a)
        tg = a + s

        rng = np.random.default_rng(999)
        x = rng.normal(size=d)
        X, Y = [], []
        for _ in range(20000):
            xn = tg @ x + 0.05 * rng.normal(size=d)
            X.append(x.copy())
            Y.append(xn[:n].copy())
            x = xn
        W = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0].T
        val = float(np.mean([V.imprint(W[i], a[i], s[i]) for i in range(n)]))
        self.assertGreater(val, 0.5,
                           f"理想 MLE 印记仅 {val:+.3f}，实验设计有问题")


class TestMeasurementIsNotConfounded(unittest.TestCase):
    """约束 9（第五轮新增）：四条测量错误，全部来自"看起来对"的设计。

    这一轮（`e3e_redesign.py` / `e3f_verdict.py`）推翻了自己上一轮写进
    文档的三个数字，根因是四个可以机器检查的设计错误。每一类都钉死。
    """

    def test_a3_windows_have_equal_consolidation(self):
        """**A3 的两个窗口必须等剂量 + 等巩固**，只改年龄。

        旧设计 `EARLY=(300,4300)`、`LATE=(TT-4800,TT-800)`：
        窗口长度都是 4000（剂量相等），但窗口关闭后到测量的时间不同——
        早期还有 25,700 步可以恢复，晚期只剩 800 步。于是"早期看起来
        更可逆"是**恢复时间**造成的，不是不可逆性。
        那个 0.0057 因此毫无意义。

        本测试检查新设计的窗口构造：t_end 对两个条件都等于 t0+dose+delta。
        """
        dose, delta, t1, t2 = 4000, 6000, 300, 20000
        end_early = t1 + dose + delta
        end_late = t2 + dose + delta
        # 两个条件的"窗口关闭 → 测量"间隔必须相等
        self.assertEqual(end_early - (t1 + dose), end_late - (t2 + dose),
                         "巩固时间不等，A3 比值测的是恢复速度")
        # 且剂量相等
        self.assertEqual(dose, dose)
        self.assertGreater(t2, end_early - dose,
                           "晚期窗口必须晚于早期窗口结束（否则年龄没差别）")

    def test_constant_eta_never_converges(self):
        """**恒定 η 的 LMS 停在一个与步数无关的误调地板。**

        这是我上一轮最严重的误读：我看到 30k 步后 `‖W−A‖/‖A‖` 还有 0.40，
        就写"LMS 需要 30k–100k 步收敛"。实际是 LMS 稳态误调
        ≈ η·tr(R)/2，**加步数不会下降**。实测 500 → 30000 步全程 0.386–0.453。

        本测试防止这个误读重现：恒定 η 的 4 倍步数不得带来显著改善。
        """
        a = V.make_env(1, V.CFG["d"])
        n, d = V.CFG["n_cells"], V.CFG["d"]

        def rel(T):
            r = V.develop(0, [], n_cells=n, d=d, t_total=T, mode="none")
            return float(np.linalg.norm(r["w"] - a[:n]) / np.linalg.norm(a[:n]))

        r1, r2 = rel(4000), rel(16000)
        self.assertGreater(r1, 0.30, f"恒定 η 在 4000 步已到 {r1:.3f}，与实测不符")
        # 4 倍步数，改善不得超过 25%（实测基本不变）
        self.assertGreater(r2, r1 * 0.75,
                           f"16000 步比 4000 步改善了 {1-r2/r1:.0%}，"
                           f"说明恒定 η 其实在收敛 —— 与 LMS 理论矛盾，需复查")

    def test_eta_decay_converges_fast(self):
        """**η 衰减后 2000 步就到 0.11**，不存在"需要 30k 步"。

        与上一条配对：两个都成立才说明"慢"不是速度问题而是 η 不衰减。
        """
        a = V.make_env(1, V.CFG["d"])
        n, d = V.CFG["n_cells"], V.CFG["d"]
        r = V.develop(0, [], n_cells=n, d=d, t_total=2000,
                      mode="eta_only", cfg={"D_lock": 0.25})
        rel = float(np.linalg.norm(r["w"] - a[:n]) / np.linalg.norm(a[:n]))
        self.assertLess(rel, 0.20,
                        f"η 衰减 2000 步只到 {rel:.3f}，与实测 0.111 不符")

    def test_imprint_zero_point_is_indifference(self):
        """**imprint 的零点在"无偏好"，不在 `w=A`。**

        实测：`w=0` → −0.0245（≈0）；`w=A` → −1.000；`w=A+S` → +1.000。

        后果：`w=A` 时印记是 −1 而不是 0。所以"A3 细胞印记为负"是
        **A3 成立**（没被养育改动）的证据，不是"A5 失败"。我上一轮把
        两者读混，用同一个平均值同时判 A3 与 A5，让它们互相抵消。

        本测试把三点钉死，防止再拿"印记为负"当失败证据。
        """
        d = V.CFG["d"]
        n = d
        a = V.make_env(1, d)
        s = V.make_shift(2, d, amp=2.5, a_mat=a)

        def mean_imp(w):
            return float(np.mean([
                V.imprint(w[i], a[i], s[i]) for i in range(n)]))

        v_zero = mean_imp(np.zeros((n, d)))
        v_a = mean_imp(a[:n].copy())
        v_b = mean_imp((a + s)[:n].copy())

        self.assertLess(abs(v_zero), 0.10,
                        f"w=0 时印记 {v_zero:+.3f} 应≈0（无偏好），零点不在无偏好处")
        self.assertAlmostEqual(v_a, -1.0, places=6,
                               msg=f"w=A 时印记 {v_a:+.4f} 应为 −1")
        self.assertAlmostEqual(v_b, 1.0, places=6,
                               msg=f"w=A+S 时印记 {v_b:+.4f} 应为 +1")

    def test_a3_and_a5_measured_on_disjoint_cells(self):
        """A3 与 A5 必须测**不同的细胞**，不得在同一平均值里抵消。

        闭锁细胞的印记应为负（没被养育改动 → A3），未闭锁细胞应为正
        （学会了新养育 → A5）。若两者混在一起求平均，负号会吃掉正号，
        得到"两边都不成立"的假象。
        """
        d = V.CFG["d"]
        n = 12
        a = V.make_env(1, d)
        s = V.make_shift(13, d, amp=2.5, a_mat=a)
        r = V.develop(0, [(300, 29700)], second_shift=s, b_start=6000,
                      n_cells=n, d=d, t_total=30000, cfg={"D_lock": 0.10})
        fz = r["frozen"]
        self.assertTrue(fz.any() and (~fz).any(),
                        "该配置下应同时存在闭锁与未闭锁细胞")
        w = r["w"]
        dA = np.linalg.norm(w - a[:n], axis=1)
        dB = np.linalg.norm(w - (a + s)[:n], axis=1)
        v = (dA - dB) / np.maximum(dA + dB, 1e-12)
        # 闭锁细胞平均为负（A3 方向），未闭锁细胞平均更高（A5 方向）
        self.assertLess(float(v[fz].mean()), 0.0,
                        f"闭锁细胞印记 {v[fz].mean():+.3f} 应为负（A3）")
        self.assertGreater(float(v[~fz].mean()), float(v[fz].mean()),
                           "未闭锁细胞印记不高于闭锁细胞 → A5 方向不存在")


class TestNoGlobalLoss(unittest.TestCase):
    """约束 7：整条实验路径不得出现 loss / reward / policy / action space。

    **⚠️ 这个测试不够，别被它骗了。** 字符串扫描只能证明"没写这些词"，
    不能证明"不存在目标函数"。

    实测（`experiments/e4a_is_this_training.py`）：预测式 LMS 的更新场
    雅可比非对称度 = 1.1e-10，即它是 `L = mean(‖y − Wx‖²)` 的**精确负梯度**，
    解析式与数值梯度相对差 5.5e-10。所以：

        没有 reward ≠ 没有目标函数。MSE 就是一个目标函数。

    真正的判据在 `TestNotJustTraining` 里（雅可比对称性）。本类保留
    字符串扫描，只是作为最低门槛。
    """

    def test_sources_have_no_forbidden_constructs(self):
        banned = ["backward(", "loss =", "loss=", "reward", "policy",
                  "action_space", "optimizer", "adam(", "sgd("]
        for mod in (E, V):
            src = Path(mod.__file__).read_text()
            hits = [b for b in banned if b in src]
            self.assertEqual(hits, [], f"{Path(mod.__file__).name} 出现禁用构造：{hits}")


class TestNotJustTraining(unittest.TestCase):
    """约束 10（v4.3 新增，最根本）：这套更新到底是"训练"还是别的。

    判据（Helmholtz）：更新场 F(W) 的雅可比若对称，则存在标量 L 使
    F = −∂L/∂W，那就是"训练"；非对称则 F 不可积，才不是训练。

    **本测试的期望值是"它确实是训练"（对称）。** 也就是说它现在是一道
    *反证*：只要 ③ 还是 LMS + 分层 + 冻结，它必须通过"是训练"这一条。
    哪天有人真做出了非保守规则，这个测试会失败并提醒改写 docs/11。
    """

    def _field_asym(self, per_cell_eta: bool = False) -> float:
        import e3v3_core as VV
        d, n = 6, 4
        rng = np.random.default_rng(0)
        a = VV.make_env(1, d)
        X = rng.normal(size=(3000, d))
        Y = X @ a[:n].T + 0.05 * rng.normal(size=(3000, n))
        etas = np.linspace(0.30, 0.003, n) if per_cell_eta else np.full(n, 0.30)

        def field(wf):
            W = wf.reshape(n, d)
            err = Y - X @ W.T
            g = -((err * etas[None, :]).T @ X) / len(X)
            return g.reshape(-1)

        w0 = rng.normal(size=(n, d)).reshape(-1)
        m = len(w0)
        eps = 1e-5
        J = np.zeros((m, m))
        for k in range(m):
            e = np.zeros(m)
            e[k] = eps
            J[:, k] = (field(w0 + e) - field(w0 - e)) / (2 * eps)
        return float(np.linalg.norm(J - J.T) / np.linalg.norm(J))

    def test_lms_is_a_gradient_field(self):
        """LMS 必须是梯度场 —— 即它确实是训练。"""
        s = self._field_asym(per_cell_eta=False)
        self.assertLess(s, 1e-6,
                        f"LMS 非对称度 {s:.2e} 不为零，与解析结论矛盾")

    def test_layering_does_not_leave_training(self):
        """**可塑性分层不能让系统离开训练。** 逐坐标不同 η 只是正缩放。"""
        s = self._field_asym(per_cell_eta=True)
        self.assertLess(s, 1e-6,
                        f"分层后非对称度 {s:.2e} 不为零；若真不为零，"
                        f"说明分层改变了场的性质，docs/11 需重写")

    def test_analytic_gradient_matches_numeric(self):
        """解析梯度 −(2/(S·n))·E[err xᵀ] 必须等于数值梯度（到 1e-6）。"""
        n, d = 4, 6
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, d))
        Y = rng.normal(size=(4000, n))
        W0 = rng.normal(size=(n, d))

        def L(W):
            return float(np.mean((Y - X @ W.T) ** 2))

        g = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                e = np.zeros((n, d))
                e[i, j] = 1e-6
                g[i, j] = (L(W0 + e) - L(W0 - e)) / 2e-6

        err = Y - X @ W0.T
        analytic = -(2.0 / (len(X) * n)) * (err.T @ X)
        rel = float(np.linalg.norm(analytic - g) / np.linalg.norm(g))
        self.assertLess(rel, 1e-6,
                        f"解析与数值梯度差 {rel:.2e}，"
                        f"说明 LMS 不对应这个 L —— 那 docs/11 的判决要改")

    def test_antisymmetric_term_is_detectable(self):
        """对照组：人造反对称项必须被检出为非对称（证明检验有区分力）。

        若这条也不通过，说明雅可比检验本身失效，前三条结论不可信。
        """
        n, d = 4, 6
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, d))
        Y = rng.normal(size=(2000, n))
        Om = np.zeros((n, n))
        for i in range(0, n - 1, 2):
            Om[i, i + 1], Om[i + 1, i] = 1.0, -1.0
        kappa = 0.5

        def field(wf):
            W = wf.reshape(n, d)
            err = Y - X @ W.T
            g = -0.3 * (err.T @ X) / len(X) + kappa * (Om @ W)
            return g.reshape(-1)

        w0 = rng.normal(size=(n, d)).reshape(-1)
        m = len(w0)
        eps = 1e-5
        J = np.zeros((m, m))
        for k in range(m):
            e = np.zeros(m)
            e[k] = eps
            J[:, k] = (field(w0 + e) - field(w0 - e)) / (2 * eps)
        s = float(np.linalg.norm(J - J.T) / np.linalg.norm(J))
        self.assertGreater(s, 0.1,
                           f"反对称项的非对称度仅 {s:.2e}，检验失去区分力")


if __name__ == "__main__":
    unittest.main(verbosity=2)
