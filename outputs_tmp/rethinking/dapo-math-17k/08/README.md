# DeepSeek / JustRL · Entropy power λ=2 · 279步

Module：rethinking；训练数据：dapo-math-17k。
Student：`DeepSeek-R1-Distill-Qwen-1.5B`；Teacher：`JustRL-DeepSeek-1.5B`。

Feature：全词表直接反传；packed response 对齐；teacher entropy JSD，power=2.0，beta_max=1.0。
本结果的 checkpoint step：279；原训练总预算：279；训练卡数：4。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| AIME24（n=16） | 47.50 | 76.67 | 480 / 480 |
| AIME25（n=16） | 34.58 | 56.67 | 480 / 480 |
| AMC23（n=16） | 87.19 | 97.50 | 640 / 640 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：作者 parquet prompt / grade_answer_verl，chat template、enable_thinking=False；T=0.7、top_p=0.95、31744 response tokens、每题16次。seed=20261011+task_index×100000+example_id×32+sample_id。

本组从原始 student 权重开始；表中步数为完整训练预算。

全参数训练；64 prompts × 4 responses；LR=1e-06 constant；训练长度=1024+7168。

相对 `configs/rethinking.yaml` 的配置差异如下。资源改成 lite 的相对路径，GPU 编号改成从 0 开始；保留原训练预算，运行目录和主日志由新入口管理。重建参数保存在 `scores.json` 的 `experiment` 中；旧文件未明确记录的字段沿用 lite 默认值。

```yaml
training:
  actor_rollout_ref:
    actor:
      dense_distill_mode: entropy_jsd
      dense_teacher_path: model/JustRL-DeepSeek-1.5B
      dense_entropy_power: 2.0
    rollout:
      calculate_log_probs: false
      log_prob_top_k: 0
  reward_model:
    enable: false
  trainer:
    total_training_steps: 279
```

原始结果：`OPD_full_orig/outputs/rethinking_r1_justrl_entropy_power2_20261015/entropy_power2_279/eval`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
