# vOPD 新增功能

来源：OPD_full_orig/modules/vOPD 的本地修改；上游基线提交 `d4146d5`。
所有开关位于 `configs/vopd.yaml`，下表省略 `training.` 前缀。

| 新增功能 | 简述与配置 |
| --- | --- |
| 全词表归一化广义 JSD | `dense_objective: progressive_jsd`；teacher 固定、student 在混合分布中保留梯度。配合 `distill_mode: gkd_rkl`、`use_baseline: false`。 |
| FKL → RKL 渐进调度 | `alpha_start`、`alpha_end`；按 optimizer update 调度，首末步精确命中端点。α=0 为 FKL，α=1 为 RKL；相同起止值可做固定 α。 |
| 独立密集 RKL 对照 | `dense_objective: reverse_kl`，直接对完整分布求梯度；`alpha_schedule: linear`。 |
| 教师熵决定 token 的 α | `alpha_schedule: teacher_entropy`，α=t^(1+H_teacher/log(2))；内部区间裁剪到 [0.001,0.999]，端点保持精确。 |
| 聚合 entropy 调度 | `alpha_schedule: teacher_entropy_mean`；先计算各有效 response token 的 α，再跨 GPU 和整个梯度累积批次按 token 数聚合。一次 optimizer update 的所有 token 共用同一 α。需要一次 generation batch 对应一次 update。 |

DeepSeek 的五组消融配置及启动方式见 [配置说明](../../configs/deepseek_jsd_ablation/README.md)。

保留原生功能：采样 OPD、全词表/学生 top-k OPD、全词表/top-k control-variate baseline、reward clipping、gradient variance 统计、LoRA/全参数训练、vLLM rollout、学习率调度、checkpoint 续训。
对应参数包括 `distill_mode`、`use_baseline`、`kl_top_k`、`opd_top_k`、`reward_clip_lambda`、`log_grad_variance`、`use_peft` 及其余原生训练字段。
`kl_top_k` 控制 baseline，`opd_top_k` 控制损失支持集，`top_k` 控制生成采样，三者用途不同。
