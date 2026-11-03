# DeepSeek 核心消融（五组）

针对 CG-JSD 三个命题中还没有单独验证过的三个组件：**限幅**、**c_t 门控 vs 常数 β**、**1/(1−β) 归一化**。
训练部分除下表列出的键外，五组与 [`../deepseek_token_gate/04_jsd_q05.yaml`](../deepseek_token_gate/04_jsd_q05.yaml)（A04，方案 A JSD q05）逐项相同；
评测改为回答上限 32768、每题 8 条，其余评测设置不变。`tests/test_vopd_core_ablation.py` 会检查这两点。不扫多 seed，全部 seed=42。

| 编号 | 配置 | dense_objective | β | C_F | 归一化 | 对照（已有） | 回答的问题 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | `01_jsd_q05_noclip.yaml` | token_jsd | 1(q_max≥0.5)·(1−c_t) | ∞ | 是 | A04 | 限幅有没有用（分叉点 a，命题 3） |
| 02 | `02_mix_q05_noclip.yaml` | token_mix | 同上 | ∞ | — | A01 | 与 01、A04、A01 组成 2×2，把“分布插值 vs 损失插值”和限幅拆开（分叉点 b） |
| 03 | `03_jsd_beta05.yaml` | token_jsd | 0.5，所有位置 | ∞ | 是 | A04 | 门控 vs 常数 β（GKD 式 JSD，命题 1） |
| 04 | `04_fkl.yaml` | token_jsd | 1，所有位置（精确 FKL） | ∞ | — | A04、A07 | 全局 FKL 基线 |
| 05 | `05_jsd_q05_unnorm.yaml` | token_jsd | 1(q_max≥0.5)·(1−c_t) | 0.16944029927253723（A04 标定值） | 否 | A04 | 1/(1−β) 归一化是否必要（命题 2） |

## 数学约定

记 p=student、q=teacher（完整词表 softmax），a_t=argmax q，c_t=p(a_t)。

- 方案 A 不变：`β = sg[1(q_max ≥ τ_q)·(1−c_t)]`，`m = (1−β)q + βp`，
  `D = KL(q‖m)/β + KL(p‖m)/(1−β)`；β=0 精确为 RKL，β=1 精确为 FKL，内部系数限制在 [0.001, 0.999]。
  KL 混合为 `β·FKL + (1−β)·RKL`。细节见 [方案 A 说明](../deepseek_token_gate/README.md)。
- `token_gate_clip: .inf`：C_F=∞，forward 分量不裁剪，也不做首个 microbatch 的 p95 标定。
  日志里 `token_gate/clip_fraction` 应恒为 0，`clip_threshold` 记为 inf。
- `token_gate_beta: 0.5` / `1.0`：所有位置使用同一个常数 β，不看 τ_q 和 c_t（`gate_fraction` 记为 1）。
  配置里保留的 `token_gate_tau_q` 此时不起作用。常数 β 必须显式给出 `token_gate_clip`，因为 p95 标定依赖门控。
  β=1 走精确 FKL 分支。
- `token_gate_normalize: false`：先按 A04 裁剪，再乘 `sg(1−β_t)`：

  ```text
  L_t = sg(1−β_t) · [ min(KL(q‖m)/β_t, C_F) + KL(p‖m)/(1−β_t) ]
  ```

  这等于从响应函数 `ψ_β(r) = log(β + (1−β)r)/(1−β)` 中去掉前面的 1/(1−β)。
  门控外 β=0，仍是精确 RKL；student 已同意的位置（c_t→1）几乎不变；
  对 teacher 首选的拉力从 `log(1+q_max−c_t)` 变成 `c_t·log(1+q_max−c_t)`，随 c_t 消失，
  限幅区的 κ(q_max) 也乘上 c_t。C_F 固定为 A04 标定值，使裁剪区与 A04 一致。
  - 没有采用 GKD 原式 `(1−β)·KL(q‖m) + β·KL(p‖m) = β(1−β)·D`：它在门控位置还多乘一个 β。
    A04 日志里约 81% 的位置过门控，门控位置的平均 β 约 0.13，用原式会把多数位置的损失缩小一个数量级，
    测到的是“信号变弱”，而不是拉力机制。
- 01–04 不裁剪：常数 β 的 GKD / FKL 基线本来就没有限幅，01、02 正是要去掉限幅。

## 共同训练与评测协议

与 A04 完全相同：vOPD；student=`DeepSeek-R1-Distill-Qwen-1.5B`，teacher=`JustRL-DeepSeek-1.5B`；
DAPO14k、seed=42；8 GPU，每卡 batch=1，梯度累积=8，全局 batch=64，211 次更新；
LoRA r64/alpha128/dropout0.05，LR=1e-5 cosine；训练生成上限 2048，temperature=1；
完整词表损失、128-token loss chunk、逐序列平均；核对初始 LoRA SHA256
`4460b8025f2a60d82df0234014a92e08101ed61efb15bd34216c46de2dab3f13`。

训练结束后自动评测 final：MATH500、Minerva、AMC23、AIME24、AIME25，每题 8 条，
回答上限 32768（总长度上限 33792），temperature=0.6、top_p=0.9、seed=42，8 个评测引擎，之后做 complete-box 重评分。
这与此前 A04 / A07 的 32k 补评协议相同（`outputs/analysis/deepseek_32k_avg16_20261030/`），只是那次每题 16 条。

## 已有对照结果（32k）

回答上限 32768，按最后一个完整答案框判分（`last_complete`），单 seed、每题 16 条。
avg@n 是单条回答正确率的估计，期望不随 n 变化，所以可以直接和新组的 avg@8 比，只是 avg@16 方差更小。

| ID | 实验 | MATH500 | Minerva | AMC23 | AIME24 | AIME25 | 宏 avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A04 | JSD q05，带限幅 | 83.35 | 31.25 | 75.78 | 42.92 | 30.00 | 52.66 |
| B2048_psi0 | A04 的梯度等价复现（4 卡） | 84.39 | 31.41 | 77.34 | 42.92 | 30.63 | 53.34 |
| A07 | 纯 RKL | 82.95 | 32.01 | 74.38 | 42.71 | 26.04 | 51.62 |
| A01 | KL 混合 q05，带限幅 | 未测 | 未测 | 未测 | 未测 | 未测 | — |

A04 与 RKL 的差为 +1.04pp，题目 bootstrap 区间 [−0.22, +2.34]，不含训练 seed 方差。
32k 下只有 A04 / B2048_psi0 这一对复现（差 0.68pp），运行间噪声还没有可靠估计。
A01 的 final checkpoint 还没做 32k 评测，02 组暂时缺这一格对照。
