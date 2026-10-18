# 结果格式

每组实验的全部 benchmark 共用 `scores.json` 和 `results.jsonl`，放在 `outputs/<module>/<data>/<编号>/`。
历史组额外有一份简短 `README.md`，说明 feature、参数和来源；没有历史代码、checkpoint、额外 JSON 或日志。
新训练还保留本次 `configs/`、`checkpoint/`、`code/` 与 `main.log`，关闭最终评测时不创建这两个结果文件。

`scores.json` 的 `schema_version` 为 1。`module` / `dataset` 标明实验归属，`evaluation` 保存采样协议，`benchmarks` 按评测集名索引。
新实验的完整参数见 `config_file` 指向的实际配置；历史实验的 `experiment` 记录模型配对、方法、feature、checkpoint step、原训练预算和重建的 `config_overrides`。历史参数依据是旧 `cfg_meta.json` / `trainer_state.json` 或 `campaign.json` 的实际覆盖值，GPU 编号及资源路径已改为可移植表示。

| 分数字段 | 含义 |
| --- | --- |
| `samples_per_question` | 每题预期采样数 n |
| `expected_questions` / `expected_samples` | 本次协议应评测的题目数 / 回答数 |
| `completed_samples` / `graded_samples` | 已保存回答数 / 有布尔判分的回答数 |
| `correct_samples` | 已保存回答中的正确数 |
| `complete_questions` | 已收齐 n 条回答且均有判分的题数 |
| `passed_questions` | 上述完整题目中至少答对一次的题数 |
| `avg_at_n_pct` | 整个 benchmark 完整后：正确回答数 / 全部回答数 × 100 |
| `pass_at_n_pct` | 整个 benchmark 完整后：至少答对一次的题数 / 全部题数 × 100 |
| `grading_errors` | 缺少有效判分的回答数；对应行 `correct: null` |
| `truncated_samples` / `truncation_pct` | 已保存回答的长度截断数 / 百分比；旧结果缺少 finish_reason 时为 null |
| `avg_generated_tokens` | 已保存回答的平均生成长度；缺少 token_count 时为 null |
| `majority_vote_at_n_pct` / `format_rate` | vOPD 的原始多数投票准确率 / boxed 格式率；按原规则对答案字符串投票，平票按 sample_id 顺序 |

所有百分比都是 0–100。`complete` 表示该 benchmark 的题目与样本均齐全且判分无误。
仅当全部配置的 benchmark 完整且本次评测未报错时，组级 `status` 才为 `complete`，否则为 `partial`。
未完成 benchmark 的 `avg_at_n_pct` / `pass_at_n_pct` / `majority_vote_at_n_pct` 是 `null`；已完成的其他 benchmark 仍保留最终分数。
部分结果的截断率和平均长度仅描述已有回答，不能视为完整 benchmark 统计。

`results.jsonl` 每行是一条回答，唯一键为 `(benchmark, example_id, sample_id)`，没有重复保存另一份 graded 或 raw 文件。

| 回答字段 | 含义 |
| --- | --- |
| `benchmark` | 评测集名；沿用 vOPD 小写与 rethinking 作者大写名字 |
| `example_id` / `sample_id` | 本组内从 0 开始的题目序号 / 采样序号 |
| `source_example_id` | 迁移 vOPD 时保留的旧 problem_id；可能不是连续整数 |
| `prompt` / `answer` | 原问题正文 / 原参考答案；prompt 不含模板和追加的 math 指令，模板方式在 evaluation 中说明 |
| `response` | 未裁剪的完整模型回答，保留换行和 Unicode |
| `correct` / `grading_error` | 原判分布尔值或 null / 错误说明 |
| `predicted_answer` | vOPD 提取出的 boxed 答案；无 boxed 时保留旧占位字符串。旧 rethinking 未保存该字段，置 null |
| `formatted` | vOPD 是否具有 boxed 答案 |
| `seed` | 独立 request seed；vOPD engine RNG 无逐样本 seed，置 null，在协议中记录 engine seed |
| `token_count` / `finish_reason` / `stop_reason` | 已知的生成长度与停止信息；旧结果没有记录时置 null，不重新推断 |

行顺序可以是题目顺序或完成顺序，读取时以唯一键为准。`results_sha256` 是 JSONL 文件字节的 SHA256，便于检查分数文件与回答文件是否成对。
历史结果另存来源相对路径和源文件 SHA256，不需要原仓库才能查看或分析。

新评测默认 vOPD 使用原 engine RNG、多样本生成与模型 EOS；rethinking 使用原独立 seed 公式、作者 prompt 和注册特殊 stop token。
这对齐了方法与评分协议，但旧多进程分片、采样批次以及早期 LoRA 初始化会影响随机轨迹；重建配置不等于逐 token 重放旧实验。
初始化不配对、仅评测部分 benchmark、取中间 checkpoint 和评测中断等差异，均在对应历史组 README 说明。
