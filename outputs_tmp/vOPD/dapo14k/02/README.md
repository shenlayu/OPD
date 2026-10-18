# Qwen 1.7B · Teacher 参考

Module：vOPD；训练数据：dapo14k。
Student：`Qwen3-1.7B-Base`；Teacher：`Qwen3-1.7B`。

Feature：教师参考模型。
本结果的 checkpoint step：0；原训练总预算：0；训练卡数：不适用。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| amc23（n=32） | 38.05 | 90.00 | 1280 / 1280 |
| aime24（n=32） | 10.00 | 40.00 | 960 / 960 |
| aime25（n=32） | 10.31 | 40.00 | 960 / 960 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：原 vOPD plain math prompt / grader，T=0.6、top_p=0.9、4096 response tokens；engine seed=42。原评测分片会改变采样批次，新入口不承诺逐 token 重放历史随机轨迹。

这是已有模型的参考评测，没有执行本地训练，不套用训练配置。

原始结果：`OPD_full_orig/outputs/progressive_jsd/eval/teacher`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
