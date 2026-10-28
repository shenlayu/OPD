# DeepSeek JSD 消融

完整的环境安装、数据/模型准备、student 直接评测和五组启动命令见仓库根目录 [TODO.md](../../TODO.md)。
所有 YAML 中的模型、数据和输出路径均相对于 OPD 根目录；`run.python: null` 使用启动命令的 Python。
在目标机器通过 `bash scripts/setup_vopd.sh` 重建本仓库的 `.venv-vopd`，不需要任何外部实验目录。

| 配置 | 定义 |
| --- | --- |
| `00_student.yaml` | 直接评测原始 DeepSeek student，使用 `scripts/evaluate_vopd_student.py`；不训练、不加载 adapter。 |
| `01_entropy_static.yaml` | 固定 u=0.5，保留 α_i=clip(0.5^(1+H_i),0.001,0.999)；cosine 学习率保留。 |
| `02_fkl.yaml` | 所有 step/token 的 α=0，完整词表 FKL。 |
| `03_jsd_fixed_05.yaml` | 所有 step/token 的 α=0.5，归一化 generalized JSD。 |
| `04_entropy_mean.yaml` | 先逐 token 计算并裁剪 α，再对一次 optimizer update 的所有有效 response token 求均值。 |
| `05_jsd_linear.yaml` | α=u，第 1 步精确 FKL，第 211 步精确 RKL。 |

H_i 为 bit 单位的 teacher 全词表 entropy；第 k 次更新的 u=(k-1)/210。
聚合组在 TRL 划分 generation batch 前完成 teacher 前向和跨卡汇总；整个梯度累积期间复用同一标量。
它要求 `steps_per_generation == gradient_accumulation_steps` 且 `num_iterations == 1`，本组配置满足这两个条件。
不能先平均 entropy 再代入公式。该组增加 teacher 前向耗时。

五组训练均为 8 GPU、全局 batch 64、211 步、LoRA r64/alpha128、seed 42、LR=1e-5 cosine，
`stop_weight=0`，保留初始 adapter SHA256 核验。每组训练完成后自动评测 final 并进行 complete-box 重评分。
中间 checkpoint 保留但不自动逐个评测。

仅串行运行五组训练实验（student 基线单独执行 TODO.md 中的命令）：

```bash
bash configs/deepseek_jsd_ablation/run_all.sh
```
