1. 学习率 不同学习率 + cosine annealing / linear annealing
2. 改散度 
    - $\alpha$ FKL + $(1-\alpha)$ RKL
    - $\alpha$ 散度
3. 不清楚时间步信息是否有价值
    - 尝试去掉时间步信息，直接将 token entropy 代入得到一个新的散度
4. 在已有的结果基础上评测一些其它 benchmark (比如 math500, minerva，探究为什么不同 benchmark 的结果趋势不同)
5. 在已有的两个 feature （Entropy power 和 dense_beta_max）基础上做一些调参或扩展
6. 设计其它 schedule 方式（例如同时依赖 student 和 teacher）
7. 找一些其它 codebase, 尝试在其上效果与 RKL 比较
8. 其它模型组合（比如 Qwen3 1.7B 蒸 4B 之类的）

其它，任何方向都可以尝试


做实验的时候可以优先用 rethinking 那个 codebase, Qwen3 1.7B Base 蒸 Qwen3 1.7B Instruct 组合，**可以只训练 40 steps 对照结果**
做完实验之后可以填一下 docs/experiment_results.md, 在完成人那里标记自己的名字。如果对实验结果分析有什么发现可以在 docs/findings 写一下文档（任何微小的分析都可以）

## vOPD DeepSeek：student 基线与五组 JSD 消融

以下命令都在上传后的 **OPD 仓库根目录**执行；配置、数据、模型和新环境均位于本目录内。
配置目录：`configs/deepseek_jsd_ablation/`。没有外部实验目录或旧机器环境依赖。

### 1. 在目标机器创建环境（首次运行）

目标机器使用 Linux x86_64、Python 3.12，以及可运行 PyTorch 2.8 / CUDA 12.8 的 NVIDIA GPU 环境。
不要上传或复用旧机器的虚拟环境，在目标机器重新创建：

```bash
bash scripts/setup_vopd.sh
```

它依据 `requirements-vopd.lock.txt` 安装到本仓库的 `.venv-vopd/`。
配置中的 `run.python: null` 表示沿用启动命令所使用的 Python。

### 2. 准备本仓库内的数据与模型

数据已放在本仓库 `data/` 下；若未上传这些文件，运行以下命令会按固定版本下载并转换。
AMC23 只保留 2023 年的 40 题，AIME25 只使用完整的 30 题，不叠加 part1/part2。

```bash
.venv-vopd/bin/python scripts/prepare_vopd_assets.py
```

两个模型应位于 `model/DeepSeek-R1-Distill-Qwen-1.5B/` 和 `model/JustRL-DeepSeek-1.5B/`。
当前目录中已有完整权重；如果目标机器没有上传模型，按固定版本下载：

```bash
.venv-vopd/bin/hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --local-dir model/DeepSeek-R1-Distill-Qwen-1.5B

.venv-vopd/bin/hf download hbx/JustRL-DeepSeek-1.5B \
  --revision 0637e4096c789c67f9eecbe8355e0bdeddede1c2 \
  --local-dir model/JustRL-DeepSeek-1.5B
```

仅评测 student 时，不需要 teacher 权重和 DAPO 训练数据；准备数据可加 `--evaluation-only`。

### 3. 直接评测原始 DeepSeek student（不训练）

```bash
.venv-vopd/bin/python scripts/evaluate_vopd_student.py \
  --config configs/deepseek_jsd_ablation/00_student.yaml
```

直接加载原始 student，不创建或加载 LoRA；评测协议与下面五组相同。

### 4. 分别运行五组训练实验

每条命令都包含 **训练 → 保存 final → 自动评测 final → complete-box 重评分**。
请等待上一条命令结束后再执行下一条；也可以使用下方顺序脚本。

```bash

# 评测 student
.venv-vopd/bin/python scripts/evaluate_vopd_student.py \
  --config configs/deepseek_jsd_ablation/00_student.yaml

# 实验 1：去掉散度的时间调度，固定 u=0.5；token entropy 区分保留
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_jsd_ablation/01_entropy_static.yaml

# 实验 2：纯 FKL，alpha=0
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_jsd_ablation/02_fkl.yaml

# 实验 3：固定 alpha=0.5 的归一化 JSD
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_jsd_ablation/03_jsd_fixed_05.yaml

# 实验 4：逐 token 计算 alpha 后取均值，整次 optimizer update 共用
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_jsd_ablation/04_entropy_mean.yaml

# 实验 5：纯线性 FKL→RKL schedule，不使用 entropy
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_jsd_ablation/05_jsd_linear.yaml
```

依次运行五组训练（任一组失败即停止；student 基线单独执行上面的命令）：

```bash
bash configs/deepseek_jsd_ablation/run_all.sh
```

### 5. 评测、保存与结果位置

- 所有配置默认使用 GPU 0–7；训练 211 步，全局 batch=64，seed=42。
- 训练成功后自动评测 **final checkpoint**。50、100、150、200、211 步 checkpoint 都会保存；
  **中间 checkpoint 不会自动逐个评测**，后续画 curve 时另外评测。
- student 与训练实验均评测 MATH500、Minerva（每题 8 条）、AMC23、AIME24、AIME25（每题 32 条）；
  temperature=0.6、top_p=0.9、回答上限 4096 tokens，采用相同的停止和判分规则。
- 结果在 `outputs/vOPD/` 下按配置中的实验名分目录，每次启动创建新编号；
  student 的实验名为 `deepseek_student`，五组训练为 `dapo14k_deepseek_01_entropy_static` 等。
- 每次运行保留 `main.log`、`configs/`、`code/`、`scores.json`、`results.jsonl`，以及
  `diagnostics/complete_box_scores.json`；训练组另有 `checkpoint/`。
- 聚合组跨所有 GPU 和整个梯度累积批次统计有效 response token，排除 prompt/padding；
  它有额外 teacher 前向成本。五组学习率等其余训练设置保持一致。
- 任一单组命令加 `--dry-run` 可检查配置及文件路径，不占用 GPU、不启动训练/评测。
  环境和数据准备后，可用 `bash configs/deepseek_jsd_ablation/run_all.sh --dry-run` 检查五组配置。
