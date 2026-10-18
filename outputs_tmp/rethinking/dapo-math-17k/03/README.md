# DeepSeek / JustRL · 作者 RKL top-16 · step20/279

Module：rethinking；训练数据：dapo-math-17k。
Student：`DeepSeek-R1-Distill-Qwen-1.5B`；Teacher：`JustRL-DeepSeek-1.5B`。

Feature：作者 top-16 advantage / policy-gradient。
本结果的 checkpoint step：20；原训练总预算：279；训练卡数：4。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| AIME24（n=16） | 28.96 | 70.00 | 480 / 480 |
| AIME25（n=16） | 26.88 | 50.00 | 480 / 480 |
| AMC23（n=16） | 74.06 | 95.00 | 640 / 640 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：作者 parquet prompt / grade_answer_verl，chat template、enable_thinking=False；T=0.7、top_p=0.95、31744 response tokens、每题16次。seed=20261011+task_index×100000+example_id×32+sample_id。

这是 279 步训练中的第 20 步 checkpoint，不是独立 20 步实验。配置保留总预算 279；JSD 进度按 (step−1)/(279−1) 计算。

这是作者 top-16 advantage 路径；与 dense JSD 的差别同时涉及目标和梯度估计，不能解释为仅更换散度。

全参数训练；64 prompts × 4 responses；LR=1e-06 constant；训练长度=1024+7168。

相对 `configs/rethinking.yaml` 的配置差异如下。资源改成 lite 的相对路径，GPU 编号改成从 0 开始；保留原训练预算，运行目录和主日志由新入口管理。重建参数保存在 `scores.json` 的 `experiment` 中；旧文件未明确记录的字段沿用 lite 默认值。

```yaml
training:
  trainer:
    total_training_steps: 279
```

原始结果：`OPD_full_orig/outputs/external_opd_20261011/rethinking_opd/eval/rkl_step20_diagnostic`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
