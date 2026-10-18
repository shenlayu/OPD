# Qwen 1.7B · 线性 JSD

Module：vOPD；训练数据：dapo14k。
Student：`Qwen3-1.7B-Base`；Teacher：`Qwen3-1.7B`。

Feature：全词表归一化 JSD；线性 FKL→RKL；LoRA。
本结果的 checkpoint step：211；原训练总预算：211；训练卡数：4。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| math500（n=8） | 67.50 | 85.00 | 4000 / 4000 |
| minerva（n=8） | 23.02 | 45.22 | 2176 / 2176 |
| amc23（n=32） | 37.11 | 70.00 | 1280 / 1280 |
| aime24（n=32） | 8.65 | 20.00 | 960 / 960 |
| aime25（n=32） | 5.52 | 23.33 | 960 / 960 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：原 vOPD plain math prompt / grader，T=0.6、top_p=0.9、4096 response tokens；engine seed=42。原评测分片会改变采样批次，新入口不承诺逐 token 重放历史随机轨迹。

早期 LoRA A 初始化未配对；seed=42 不能据此证明与其他方法具有相同初始化。新入口已在 PEFT 创建前设 seed，不能精确重放旧初始化。

LoRA r=64、alpha=128；全局 batch=64；LR=1e-05 cosine；训练长度=1024+2048；seed=42。

相对 `configs/vopd.yaml` 的配置差异如下。资源改成 lite 的相对路径，GPU 编号改成从 0 开始；保留原训练预算，运行目录和主日志由新入口管理。重建参数保存在 `scores.json` 的 `experiment` 中；旧文件未明确记录的字段沿用 lite 默认值。

```yaml
{}
```

原始结果：`OPD_full_orig/outputs/progressive_jsd/eval/progressive_jsd`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
