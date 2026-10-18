# vOPD 新增功能

来源：OPD_full_orig/modules/vOPD 的本地修改；上游基线提交 `d4146d5`。
所有开关位于 `configs/vopd.yaml`，下表省略 `training.` 前缀。

| 新增功能 | 简述与配置 |
| --- | --- |
| 全词表归一化广义 JSD | `dense_objective: progressive_jsd`；teacher 固定、student 在混合分布中保留梯度。配合 `distill_mode: gkd_rkl`、`use_baseline: false`。 |
| FKL → RKL 渐进调度 | `alpha_start`、`alpha_end`；按 optimizer update 调度，首末步精确命中端点。α=0 为 FKL，α=1 为 RKL；相同起止值可做固定 α。 |
| 独立密集 RKL 对照 | `dense_objective: reverse_kl`，直接对完整分布求梯度；`alpha_schedule: linear`。 |
| 教师熵决定 token 的 α | `alpha_schedule: teacher_entropy`，α=t^(1+H_teacher/log(2))；内部区间裁剪到 [0.001,0.999]，端点保持精确。 |
| 分块及重计算 | `loss_token_chunk_size` 控制 token 分块，FP32 损失计算与非重入 checkpoint 降低中间张量开销；仍需完整 student/teacher logits。 |
| 初始化可重复性 | 在创建 PEFT 前设置 `seed`；跨 rank 比较 LoRA SHA256；`expected_initial_adapter_sha256` 可约束成对实验初始化，结果进主日志。 |
| 梯度累积修复 | 密集目标恢复 Trainer 正确的累积缩放，避免 TRL/GRPO 特殊缩放影响 JSD；由 `gradient_accumulation_steps` 控制。 |
| 输入和 token 一致性修复 | 截断 prompt 到 `max_prompt_length`，密集前向使用与左 padding 匹配的 position_ids；启动时验证师生词表/token ID 完全一致。 |
| 序列化兼容 | 教师模型加载后将 dtype 还原为可序列化值，避免 TRL/Transformers 记录配置失败。 |
| 最终评测修复与统一 | `evaluation.*` 控制多次采样和 prompt/长度；保留数学规则评分，检查样本完整性并统计截断。默认 `seed_mode: engine`、`stop_mode: model_eos` 对齐旧 vOPD；也支持独立 request seed 和注册特殊 stop token。 |
| 两文件评测产物 | 全部 benchmark 合并到 `scores.json`、`results.jsonl`，保留完整回答、原始判分、avg/pass、majority vote、格式率与截断信息；部分完成不给最终准确率。见 [格式](../results_format.md)。 |

保留原生功能：采样 OPD、全词表/学生 top-k OPD、全词表/top-k control-variate baseline、reward clipping、gradient variance 统计、LoRA/全参数训练、vLLM rollout、学习率调度、checkpoint 续训。
对应参数包括 `distill_mode`、`use_baseline`、`kl_top_k`、`opd_top_k`、`reward_clip_lambda`、`log_grad_variance`、`use_peft` 及其余原生训练字段。
`kl_top_k` 控制 baseline，`opd_top_k` 控制损失支持集，`top_k` 控制生成采样，三者用途不同。

本次整理新增：单 YAML/单入口、原子实验编号、配置和实际执行代码快照、统一主日志、最终自动评测，以及把原先遗漏传递的 `max_grad_norm` 接入 Trainer。
训练 prompt 的 `enable_thinking`、蒸馏温度 `kd_temperature`、评测各项采样参数可显式控制；禁用 top-k 时自动适配 vLLM 的 -1 与 Transformers 的 0。
当前默认配置对齐历史线性 JSD 组的 4 GPU / batch 64 / LoRA r64；评测采用 T=0.6、top_p=0.9、4096 response tokens。旧初始化未配对的实验在历史 README 中注明。
