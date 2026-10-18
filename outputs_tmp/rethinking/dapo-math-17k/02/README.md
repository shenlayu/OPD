# DeepSeek / JustRL · Teacher 参考

Module：rethinking；训练数据：dapo-math-17k。
Student：`DeepSeek-R1-Distill-Qwen-1.5B`；Teacher：`JustRL-DeepSeek-1.5B`。

Feature：教师参考模型。
本结果的 checkpoint step：0；原训练总预算：0；训练卡数：不适用。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| AIME24（n=16） | 54.38 | 80.00 | 480 / 480 |
| AIME25（n=16） | 36.88 | 60.00 | 480 / 480 |
| AMC23（n=16） | 89.84 | 97.50 | 640 / 640 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：作者 parquet prompt / grade_answer_verl，chat template、enable_thinking=False；T=0.7、top_p=0.95、31744 response tokens、每题16次。seed=20261011+task_index×100000+example_id×32+sample_id。

这是已有模型的参考评测，没有执行本地训练，不套用训练配置。

原始结果：`OPD_full_orig/outputs/external_opd_20261011/rethinking_opd/eval/teacher_reference`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
