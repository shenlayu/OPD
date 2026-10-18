# OPD_lite

## 环境与资产

假定已有 Linux x86_64、Python 3.12 和可用的 NVIDIA 驱动。两个框架的 Transformers / PEFT 版本不同，分别安装到独立环境：

```bash
python3.12 -m venv .venv-vopd
.venv-vopd/bin/python -m pip install -r modules/vOPD/requirements.txt

python3.12 -m venv .venv-rethinking
.venv-rethinking/bin/python -m pip install -r modules/rethinking/requirements.txt
```

requirements 只列必要依赖，间接依赖由 pip 安装；其中已包含与 PyTorch 2.8 / CUDA 12.8 匹配的官方 Flash Attention wheel，无需额外安装命令。

数据和模型下载分别见两个文件夹的对应 markdown.

## 启动

先编辑对应 YAML，设置模型、数据、GPU、训练预算及评测参数，再执行：

```bash
.venv-vopd/bin/python scripts/run_vopd.py
.venv-rethinking/bin/python scripts/run_rethinking.py
```

两条命令独立运行，各启动一次实验。默认 vOPD 使用历史 Qwen 1.7B 的 4 GPU 线性 JSD 设置，默认 rethinking 使用 DeepSeek / JustRL 的 4 GPU 原生 top-16 OPD。
两者默认训练 1 epoch 并自动评估最终 checkpoint；可用 `evaluation.enabled: false` 关闭最终评测。

```bash
# 只校验配置、查看最终参数；不占 GPU、不下载资源、不创建实验目录。
.venv-vopd/bin/python scripts/run_vopd.py --dry-run
.venv-rethinking/bin/python scripts/run_rethinking.py --dry-run

# CLI 可覆盖 YAML 已定义的字段，实际生效值会保存。
.venv-vopd/bin/python scripts/run_vopd.py training.max_steps=40
.venv-rethinking/bin/python scripts/run_rethinking.py \
  training.actor_rollout_ref.actor.dense_distill_mode=entropy_jsd \
  training.actor_rollout_ref.actor.dense_entropy_power=1.25 \
  training.trainer.total_training_steps=40
```

新增功能及对应开关见 [vOPD](docs/features/vOPD.md)、[rethinking](docs/features/rethinking.md)。
rethinking 选择密集目标时，入口自动关闭原生 reward/PG 路径，使用同一份教师路径。
配置保留原生 OPD/FSDP/vLLM 参数名；不再提供 RL、SFT、Megatron、SGLang、异步 agent loop 和实验 campaign 入口。

## 实验目录

```text
outputs/<vOPD|rethinking>/<run.dataset>/<01|02|...>/
├── configs/
│   ├── requested.yaml        # 输入文件原文
│   └── vopd.yaml             # 或 rethinking.yaml：覆盖后实际参数、资源绝对路径
├── checkpoint/
│   ├── checkpoint-*/        # vOPD 可恢复训练状态
│   ├── global_step_*/       # rethinking FSDP 模型/优化器/RNG/数据迭代器状态
│   └── final/               # vOPD 最终 LoRA/模型，或 rethinking 导出的 HF 模型
├── code/                    # 本次 module、共享运行/评测代码及依赖清单的快照
├── main.log                 # 训练、导出、评测的唯一主日志
├── scores.json              # 本组全部 benchmark 的分数、计数和评测协议
└── results.jsonl            # 每条回答一行：题目、答案、完整输出、判分及采样信息
```

编号按 module 和数据集独立递增，使用原子 mkdir 防止同时启动覆盖。
子进程从 `code/` 快照运行，修改工作目录代码不会影响已启动实验。
checkpoint 内保留加载/续训必需的模型 JSON 和训练状态；评测只有一个分数 JSON 和一个回答 JSONL，不写分片、单独 graded 缓存、campaign、process、audit、W&B、SwanLab 等产物。
Ray 的内部运行日志位于该实验的临时目录，退出时清理；只清理本次进程的子进程，不停止共享 Ray 集群。
评测同时在 `main.log` 打印 avg@N、pass@N、截断数量和评分错误数量。中断或评分出错时保留已保存的回答，`scores.json` 标记 `partial`；未完成 benchmark 的最终分数为 `null`。格式见 [结果格式](docs/results_format.md)。

## 历史结果与方法文档

[历史结果总表](docs/experiment_results_20261016.md) 收录 31 组 vOPD / rethinking 结果，每组只有 `scores.json`、`results.jsonl`、`README.md`。
历史组的 README 说明 feature、模型配对、训练总预算、评估 checkpoint 和相对于当前单一 YAML 的配置差异；未复制历史代码、模型、数据、checkpoint 或日志。
历史分数由原保存回答和判分复核，未重新生成或重新判分。新训练仍按上面的目录保存本次 configs、checkpoint、code 和主日志。

迁移的方法笔记：[Entropy JSD](docs/notes/entropy_jsd.md)、[Progressive JSD 草稿](docs/notes/progressive_jsd_draft.md)、[无训练进度的 entropy 推导](docs/notes/entropy_jsd_without_time.md)。草稿中的候选方案有明确标注，不代表已实现功能。

## 续训与验证

vOPD 设置 `training.resume_from_checkpoint` 为已有 `checkpoint-N`；
rethinking 设置 `training.trainer.resume_mode: resume_path` 和 `training.trainer.resume_from_path` 为已有 `global_step_N`。
每次启动仍建立新编号，保留原实验。渐进 JSD 的总步数是调度预算，恢复同一实验时应保持原预算。