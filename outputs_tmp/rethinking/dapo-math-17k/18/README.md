# Qwen 1.7B · 线性 JSD + cosine 2e-6 · 40步

Module：rethinking；训练数据：dapo-math-17k。
Student：`Qwen3-1.7B-Base`；Teacher：`Qwen3-1.7B`。

Feature：全词表直接反传；packed response 对齐；线性 FKL→RKL JSD；Base 学生复用 teacher chat template / 合并 EOS；cosine LR，起始2e-6，零 warmup。
本结果的 checkpoint step：40；原训练总预算：40；训练卡数：2。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| AIME24（n=16） | 5.83 | 30.00 | 480 / 480 |
| AIME25（n=16） | 未完成 | 未完成 | 473 / 480 |
| AMC23（n=16） | 未完成 | 未完成 | 619 / 640 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：作者 parquet prompt / grade_answer_verl，chat template、enable_thinking=False；T=0.7、top_p=0.95、31744 response tokens、每题16次。seed=20261011+task_index×100000+example_id×32+sample_id。

本组从原始 student 权重开始；表中步数为完整训练预算。

训练与导出已完成，评测中断。保留全部已有回答；未完成 benchmark 的最终 avg/pass 为 null，不沿用旧 completion-order 临时百分比。

全参数训练；64 prompts × 4 responses；LR=2e-06 cosine；训练长度=1024+7168。

相对 `configs/rethinking.yaml` 的配置差异如下。资源改成 lite 的相对路径，GPU 编号改成从 0 开始；保留原训练预算，运行目录和主日志由新入口管理。重建参数保存在 `scores.json` 的 `experiment` 中；旧文件未明确记录的字段沿用 lite 默认值。

```yaml
run:
  teacher_chat_template: true
  gpus:
  - 0
  - 1
training:
  actor_rollout_ref:
    actor:
      optim:
        lr: 2.0e-06
        lr_warmup_steps: 0
        lr_scheduler_type: cosine
      dense_distill_mode: linear_jsd
      dense_teacher_path: model/Qwen3-1.7B
    rollout:
      calculate_log_probs: false
      log_prob_top_k: 0
    model:
      path: model/Qwen3-1.7B-Base
  data:
    apply_chat_template_kwargs:
      enable_thinking: false
  reward_model:
    enable: false
    model:
      path: model/Qwen3-1.7B
  trainer:
    total_training_steps: 40
    n_gpus_per_node: 2
```

原始结果：`OPD_full_orig/outputs/rethinking_qwen17b_jsd_cosine_20261015/jsd_cosine_40/eval`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
