# rethinking 新增功能

将原生 `rethinking-opd` 和本地 `rethinking-opd-jsd` 合并到一个 module。
上游基线提交：`ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。
配置只有 `configs/rethinking.yaml`；以下 actor 参数位于 `training.actor_rollout_ref.actor`。

| 新增功能 | 简述与配置 |
| --- | --- |
| 密集 RKL | `dense_distill_mode: rkl`，在学生生成的 prefix 上直接对全词表 RKL 求梯度，不再用 top-k policy-gradient reward。 |
| 渐进 JSD | `dense_distill_mode: linear_jsd`，第一个 optimizer update 为 FKL，最后一个为 RKL，调度依赖实际总更新预算。 |
| 教师熵 JSD | `dense_distill_mode: entropy_jsd`，每个 token 用教师未 temper 分布的熵决定 α，保留精确端点和内部数值裁剪。 |
| Entropy power | `dense_entropy_power >= 1`，α=t^((1+H_bits)^power)，放大熵对 FKL 阶段的影响；1 还原基础 entropy JSD。 |
| 后半程温度调度 | `dense_beta_max >= 1`，前半程 β=1，后半程线性到 beta_max；对两侧 logits 同时 temper，损失除以 β²；熵仍由原教师分布计算。 |

保留原生 OPD：`dense_distill_mode: off`。采样 token 或 top-k 蒸馏由
`training.actor_rollout_ref.rollout.log_prob_top_k` 控制；保留 `only_stu`、`only_tch`、`intersection`、`union`、`union-intersection`，以及 student/teacher/无概率权重、教师温度、格式 reward mask、PPO clipping、FSDP 与优化器 offload、学习率调度、断点续训等 OPD 参数。

选择密集目标会自动关闭 `reward_model.enable`、rollout 的 top-k reward/logprob 收集，并使用 `reward_model.model.path` 作为密集 teacher。
JSD 要求每个 rollout batch 只有一次 actor update；入口会检查该约束，避免调度与实际 optimizer 更新错位。