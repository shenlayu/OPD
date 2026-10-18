# OPD 历史实验结果

每组只含 `scores.json`、`results.jsonl`、`README.md`；路径为 `outputs/<module>/<训练数据>/<编号>/`。组名链接进入 feature、参数和来源说明。未迁移历史代码、checkpoint、模型、数据、日志和评测分片。

表格为 **Avg / Pass@n（%）**：Avg=正确回答数/全部回答数；Pass@n=至少答对一次的题数/题目数。`—` 表示未评测。未完成的 benchmark 仅展示已保存数量，分数 JSON 中最终准确率为 `null`。详见 [结果格式](results_format.md)。

| Module | 训练设置 | 评测设置  |
| --- | --- | ---  |
| vOPD | DAPO14k；LoRA r64/alpha128；211步；全局 batch64；LR 1e-5 cosine | 原 plain math prompt / grader；T=0.6、top_p=0.9、4096 tokens；MATH500/Minerva n=8，AMC/AIME n=32 |
| rethinking | DAPO-math-17k；全参数；64 prompts×4 responses；LR 1e-6 constant，cosine组起始2e-6 | 作者 parquet prompt / grader；chat template、enable_thinking=False；T=0.7、top_p=0.95、31744 tokens；三项均 n=16 |

vOPD 训练回答上限2048；rethinking为7168。不同 module 的 prompt、采样次数、训练范围和长度预算不同，应在各自协议内比较。

## vOPD · Qwen 1.7B

Student：`Qwen3-1.7B-Base` → Teacher：`Qwen3-1.7B`。

| 结果组 | checkpoint / 原预算 | math500 | minerva | amc23 | aime24 | aime25 | 完成人 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [01 · 训练前 student](../outputs/vOPD/dapo14k/01/README.md) | 未训练 | 60.95 / 83.80 | 18.01 / 41.18 | 33.44 / 87.50 | 4.48 / 23.33 | 3.02 / 23.33 |  |
| [02 · Teacher 参考](../outputs/vOPD/dapo14k/02/README.md) | 未训练 | — | — | 38.05 / 90.00 | 10.00 / 40.00 | 10.31 / 40.00 |  |
| [03 · 全词表 RKL](../outputs/vOPD/dapo14k/03/README.md) | 211 / 211 | 50.70 / 83.20 | 17.56 / 43.01 | 29.45 / 75.00 | 6.46 / 23.33 | 4.69 / 23.33 |  |
| [04 · 线性 JSD](../outputs/vOPD/dapo14k/04/README.md) | 211 / 211 | 67.50 / 85.00 | 23.02 / 45.22 | 37.11 / 70.00 | 8.65 / 20.00 | 5.52 / 23.33 |  |
| [05 · Entropy JSD λ=1](../outputs/vOPD/dapo14k/05/README.md) | 211 / 211 | 67.23 / 86.40 | 24.63 / 43.01 | 38.13 / 75.00 | 9.38 / 26.67 | 5.00 / 16.67 |  |
| [06 · 纯 FKL](../outputs/vOPD/dapo14k/06/README.md) | 211 / 211 | 66.03 / 86.20 | 22.38 / 46.32 | 35.94 / 77.50 | 7.29 / 26.67 | 5.10 / 23.33 |  |
| [07 · 固定 α=0.5 JSD](../outputs/vOPD/dapo14k/07/README.md) | 211 / 211 | 66.75 / 85.60 | 23.99 / 43.75 | 35.94 / 70.00 | 7.08 / 30.00 | 5.63 / 23.33 |  |

早期线性 JSD、RKL、entropy JSD 的 LoRA A 初始化未配对；seed 同为42也不是完全控制初始化。后续纯 FKL 与固定α=0.5组已核对相同初始化。Teacher仅保存AMC/AIME三项。

## vOPD · Qwen 4B

Student：`Qwen3-4B-Base` → Teacher：`Qwen3-4B`。

| 结果组 | checkpoint / 原预算 | math500 | minerva | amc23 | aime24 | aime25 | 完成人 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [08 · 训练前 student](../outputs/vOPD/dapo14k/08/README.md) | 未训练 | 72.83 / 90.00 | 32.17 / 52.94 | 47.89 / 92.50 | 10.42 / 40.00 | 9.38 / 40.00 |  |
| [09 · 全词表 RKL](../outputs/vOPD/dapo14k/09/README.md) | 211 / 211 | 58.68 / 89.80 | 21.00 / 48.16 | 45.08 / 82.50 | 14.58 / 36.67 | 12.29 / 36.67 |  |
| [10 · 线性 JSD](../outputs/vOPD/dapo14k/10/README.md) | 211 / 211 | 73.90 / 90.40 | 26.06 / 45.96 | 45.39 / 77.50 | 15.52 / 46.67 | 13.54 / 30.00 |  |
| [11 · Entropy JSD λ=1](../outputs/vOPD/dapo14k/11/README.md) | 211 / 211 | 75.10 / 90.20 | 26.01 / 47.43 | 47.34 / 80.00 | 15.00 / 43.33 | 15.10 / 33.33 |  |

三个训练组已核对相同初始 adapter。全部 vOPD RKL 都是全词表直接反传，与采样 control-variate baseline 不同。

## vOPD · OLMo 7B

Student：`Olmo-3-1025-7B` → Teacher：`Olmo-3-7B-Think`。

| 结果组 | checkpoint / 原预算 | math500 | minerva | amc23 | aime24 | aime25 | 完成人 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [12 · 全词表 RKL](../outputs/vOPD/dapo14k/12/README.md) | 211 / 211 | 56.38 / 73.40 | 21.65 / 37.13 | 29.69 / 40.00 | 3.33 / 16.67 | 6.77 / 16.67 |  |
| [13 · 线性 JSD](../outputs/vOPD/dapo14k/13/README.md) | 211 / 211 | 58.00 / 74.60 | 24.36 / 40.81 | 31.17 / 45.00 | 3.85 / 13.33 | 8.13 / 16.67 |  |

两组均完成211步并核对相同初始化；没有此协议下训练前 student 的完整 benchmark 结果。

## rethinking · DeepSeek / JustRL

Student：`DeepSeek-R1-Distill-Qwen-1.5B` → Teacher：`JustRL-DeepSeek-1.5B`。

| 结果组 | checkpoint / 原预算 | AIME24 | AIME25 | AMC23 | 完成人 |
| --- | --- | --- | --- | --- | --- |
| [01 · 训练前 student](../outputs/rethinking/dapo-math-17k/01/README.md) | 未训练 | 32.29 / 66.67 | 21.88 / 46.67 | 72.66 / 95.00 |  |
| [02 · Teacher 参考](../outputs/rethinking/dapo-math-17k/02/README.md) | 未训练 | 54.38 / 80.00 | 36.88 / 60.00 | 89.84 / 97.50 |  |
| [03 · 作者 RKL top-16](../outputs/rethinking/dapo-math-17k/03/README.md) | 20 / 279 | 28.96 / 70.00 | 26.88 / 50.00 | 74.06 / 95.00 |  |
| [04 · 全词表 RKL](../outputs/rethinking/dapo-math-17k/04/README.md) | 20 / 20 | 32.29 / 70.00 | 25.21 / 50.00 | 75.47 / 95.00 |  |
| [05 · 线性 JSD](../outputs/rethinking/dapo-math-17k/05/README.md) | 20 / 20 | 31.25 / 63.33 | 25.00 / 50.00 | 75.47 / 95.00 |  |
| [06 · 作者 RKL top-16](../outputs/rethinking/dapo-math-17k/06/README.md) | 279 / 279 | 50.21 / 80.00 | 36.04 / 60.00 | 86.88 / 95.00 |  |
| [07 · 线性 JSD](../outputs/rethinking/dapo-math-17k/07/README.md) | 279 / 279 | 48.33 / 80.00 | 35.00 / 56.67 | 87.19 / 97.50 |  |
| [08 · Entropy power λ=2](../outputs/rethinking/dapo-math-17k/08/README.md) | 279 / 279 | 47.50 / 76.67 | 34.58 / 56.67 | 87.19 / 97.50 |  |

原生 RKL 使用作者 top-16 advantage；JSD/entropy 使用全词表直接反传。两者差值不能只归因于更换散度。独立20步 dense RKL与JSD对照使用对齐的密集路径。

## rethinking · Qwen 1.7B

Student：`Qwen3-1.7B-Base` → Teacher：`Qwen3-1.7B`。

| 结果组 | checkpoint / 原预算 | AIME24 | AIME25 | AMC23 | 完成人 |
| --- | --- | --- | --- | --- | --- |
| [09 · 全词表 RKL](../outputs/rethinking/dapo-math-17k/09/README.md) | 20 / 40 | 3.96 / 13.33 | 3.75 / 16.67 | 32.34 / 80.00 |  |
| [10 · 线性 JSD](../outputs/rethinking/dapo-math-17k/10/README.md) | 20 / 20 | 3.33 / 26.67 | 2.50 / 23.33 | 27.81 / 67.50 |  |
| [11 · Entropy JSD λ=1](../outputs/rethinking/dapo-math-17k/11/README.md) | 20 / 40 | 2.71 / 16.67 | 1.88 / 10.00 | 26.25 / 70.00 |  |
| [12 · 全词表 RKL](../outputs/rethinking/dapo-math-17k/12/README.md) | 40 / 40 | 7.50 / 30.00 | 5.42 / 26.67 | 31.56 / 75.00 |  |
| [13 · 线性 JSD](../outputs/rethinking/dapo-math-17k/13/README.md) | 40 / 40 | 5.00 / 16.67 | 3.75 / 23.33 | 32.97 / 75.00 |  |
| [14 · Entropy JSD λ=1](../outputs/rethinking/dapo-math-17k/14/README.md) | 40 / 40 | 6.88 / 26.67 | 4.58 / 30.00 | 33.91 / 75.00 |  |
| [15 · Entropy power λ=1.25](../outputs/rethinking/dapo-math-17k/15/README.md) | 40 / 40 | 6.88 / 23.33 | 3.54 / 16.67 | 35.63 / 77.50 |  |
| [16 · Entropy tempered βmax=1.25](../outputs/rethinking/dapo-math-17k/16/README.md) | 40 / 40 | 5.00 / 23.33 | 2.92 / 10.00 | 34.06 / 77.50 |  |
| [17 · Entropy power λ=2](../outputs/rethinking/dapo-math-17k/17/README.md) | 40 / 40 | 6.04 / 26.67 | 4.38 / 20.00 | 36.72 / 85.00 |  |
| [18 · 线性 JSD + cosine 2e-6](../outputs/rethinking/dapo-math-17k/18/README.md) | 40 / 40 | 5.83 / 30.00 | 未完成 473/480 | 未完成 619/640 |  |

本表全部使用全词表直接反传，并为Base学生复用teacher chat template与EOS。独立20步线性JSD在20步内完成0→1；entropy第20步属于40步预算，进度为19/39，不能改成独立20步调度。RKL第20步同样来自40步训练。

Entropy基础映射为 α=u^(1+Hbits)，power为 α=u^((1+Hbits)^λ)；tempered组在后半程将β从1提高到1.25。定义见 [Entropy JSD](notes/entropy_jsd.md)，配置开关见 [features](features/rethinking.md)。

cosine组训练与导出完成，评测保存1,572/1,600条：AIME24 480/480、AIME25 473/480、AMC23 619/640。后两项没有最终分数。没有此rethinking协议下Qwen训练前student的完整评测，不能用vOPD的student分数替代。