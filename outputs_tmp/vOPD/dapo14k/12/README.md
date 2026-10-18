# OLMo 7B · 全词表 RKL

Module：vOPD；训练数据：dapo14k。
Student：`Olmo-3-1025-7B`；Teacher：`Olmo-3-7B-Think`。

Feature：全词表直接反传 RKL；LoRA。
本结果的 checkpoint step：211；原训练总预算：211；训练卡数：8。

| Benchmark | Avg（%） | Pass@n（%） | 已保存 / 应有回答 |
| --- | --- | --- | --- |
| math500（n=8） | 56.38 | 73.40 | 4000 / 4000 |
| minerva（n=8） | 21.65 | 37.13 | 2176 / 2176 |
| amc23（n=32） | 29.69 | 40.00 | 1280 / 1280 |
| aime24（n=32） | 3.33 | 16.67 | 960 / 960 |
| aime25（n=32） | 6.77 | 16.67 | 960 / 960 |

[分数与协议](scores.json) · [完整回答与原判分](results.jsonl)

评测：原 vOPD plain math prompt / grader，T=0.6、top_p=0.9、4096 response tokens；engine seed=42。原评测分片会改变采样批次，新入口不承诺逐 token 重放历史随机轨迹。

该系列的成对实验已核对相同初始 adapter；保存原 expected_initial_adapter_sha256（如有），用于显式校验。

LoRA r=64、alpha=128；全局 batch=64；LR=1e-05 cosine；训练长度=1024+2048；seed=42。

相对 `configs/vopd.yaml` 的配置差异如下。资源改成 lite 的相对路径，GPU 编号改成从 0 开始；保留原训练预算，运行目录和主日志由新入口管理。重建参数保存在 `scores.json` 的 `experiment` 中；旧文件未明确记录的字段沿用 lite 默认值。

```yaml
run:
  gpus:
  - 0
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
training:
  model_name_or_path: model/Olmo-3-1025-7B
  teacher_model_name_or_path: model/Olmo-3-7B-Think
  gradient_accumulation_steps: 8
  vllm_gpu_memory_utilization: 0.17
  vllm_tensor_parallel_size: 4
  dense_objective: reverse_kl
  expected_initial_adapter_sha256: cf9ea61e99a406d7c243e07c24f80976baa0c6805ceeeb2a92a1eac782a51171
```

原始结果：`OPD_full_orig/outputs/progressive_jsd_7b/eval/reverse_kl`。

仅迁移结果和说明；未迁移历史 checkpoint、代码快照、日志或评测分片。保留原始回答和正确性标签，未重新生成或重新判分。
