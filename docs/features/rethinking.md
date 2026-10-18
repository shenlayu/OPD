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
| Packed 输入及边界修复 | 生成后的每个 response token 对齐其前一位置 logits；不让相邻 packed 序列串扰，保留首个 response 的监督。密集目标要求 remove-padding、动态 microbatch、SP=1。 |
| 分块稳定损失 | `dense_token_chunk_size`（旧实现固定 64，迁移后可配置），FP32 运算、重计算，teacher 无梯度；保持原生动态 microbatch 加权约定。 |
| 教师内存生命周期 | 密集目标仅在更新期间将冻结 teacher 移到 GPU，更新后卸载到 CPU；避免 teacher 占用 rollout 显存。 |
| 有效 α 指标 | 跨 microbatch、跨 DP rank 汇总 token 加权平均 α 及裁剪比例，同时记录 β、熵指数、loss 和 grad norm 到主日志。密集路径不把占位 PG reward 当作实测准确率。 |
| 可控学习率及预算 | `actor.optim.lr`、`actor.optim.lr_scheduler_type: constant/cosine`、warmup/min_lr_ratio 和 `training.trainer.total_training_steps`；单次实验即可覆盖原先分散 campaign 的设置。 |
| Base 学生模板适配 | `run.teacher_chat_template: true`：临时复用 teacher chat template 和双方 EOS，词表不变；训练和最终导出使用同一协议。非 thinking Qwen 使用 `training.data.apply_chat_template_kwargs: {enable_thinking: false}`。 |
| 独立最终评测 | `evaluation.*`；使用作者 parquet prompt 和 `grade_answer_verl`，默认 AIME24/AIME25/AMC23 的 avg@16、T=0.7、top_p=0.95、31744 response tokens。保留特殊 stop token 检查、独立随机种子、checkpoint 导出后的 tokenizer/generation_config 恢复。 |
| 运行隔离 | 本次运行的私有 Ray 集群，退出释放所属进程，避免全局 `ray stop`；不连接已有集群。 |
| 两文件评测产物 | `scores.json` 汇总所有 benchmark；`results.jsonl` 合并完整回答与判分。保留原 seed=20261011+task_index×100000+example_id×32+sample_id，可通过 `evaluation.seed_*` 调整；中断保留已保存回答，未完成项不给最终准确率。见 [格式](../results_format.md)。 |

保留原生 OPD：`dense_distill_mode: off`。采样 token 或 top-k 蒸馏由
`training.actor_rollout_ref.rollout.log_prob_top_k` 控制；保留 `only_stu`、`only_tch`、`intersection`、`union`、`union-intersection`，以及 student/teacher/无概率权重、教师温度、格式 reward mask、PPO clipping、FSDP 与优化器 offload、学习率调度、断点续训等 OPD 参数。

选择密集目标会自动关闭 `reward_model.enable`、rollout 的 top-k reward/logprob 收集，并使用 `reward_model.model.path` 作为密集 teacher。
JSD 要求每个 rollout batch 只有一次 actor update；入口会检查该约束，避免调度与实际 optimizer 更新错位。

本次整理新增：统一目录编号、配置与代码快照、最终 HF 导出与评测串联、主日志输出、参数一致性校验。
保留这些可复用 feature 的实现，不保留 sweep/campaign、token gap 分析、临时 smoke 脚本及逐 token 绘图。
