# Qwen 4B · 线性 JSD

Module：vOPD；训练数据：dapo14k。
Student：`Qwen3-4B-Base`；Teacher：`Qwen3-4B`。

Feature：全词表归一化 JSD；线性 FKL→RKL；LoRA。
本结果的 checkpoint step：211；原训练总预算：211；训练卡数：4。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| math500（n=8） | 73.90 | 90.40 | 4000 / 4000 |
| minerva（n=8） | 26.06 | 45.96 | 2176 / 2176 |
| amc23（n=32） | 45.39 | 77.50 | 1280 / 1280 |
| aime24（n=32） | 15.52 | 46.67 | 960 / 960 |
| aime25（n=32） | 13.54 | 30.00 | 960 / 960 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：原 vOPD plain math prompt / grader，T=0.6、top_p=0.9、4096 response tokens；engine seed=42。原评测分片会改变采样批次，新入口不承诺逐 token 重放历史随机轨迹。

该系列的成对实验已核对相同初始 adapter；保存原 expected_initial_adapter_sha256（如有），用于显式校验。

LoRA r=64、alpha=128；全局 batch=64；LR=1e-05 cosine；训练长度=1024+2048；seed=42。

相对 `configs/vopd.yaml` 的配置差异如下。资源改成 lite 的相对路径，GPU 编号改成从 0 开始；保留原训练预算，运行目录和主日志由新入口管理。重建参数保存在 `scores.json` 的 `experiment` 中；旧文件未明确记录的字段沿用 lite 默认值。

```yaml
training:
  model_name_or_path: model/Qwen3-4B-Base
  teacher_model_name_or_path: model/Qwen3-4B
  expected_initial_adapter_sha256: e993dc0017c0ab0de66ea38dde3f4fe36b29e9b3c13441a1743d13bfc2f6081c
```

原始结果：`OPD_full_orig/outputs/progressive_jsd_4b/eval/progressive_jsd`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
