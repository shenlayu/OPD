## 方法思路：置信门控 JSD（CG-JSD）

记 p=student、q=teacher，a_t = argmax q 是 teacher 的首选 token，q_max = q(a_t)，c_t = p(a_t) 是 student 给它的概率。

**一句话**：OPD 的师生分歧集中在少数位置：teacher 有明确首选，student 却几乎不给它概率（c_t 很小）。
RKL 对这个首选的拉力正比于 c_t，所以恰好在最该修的地方修不动。我们让每个位置向 FKL 偏移 β_t = 1 − c_t，
student 已同意时自动退回 RKL；在极端漏看处再给拉力封顶。

**问题。** RKL 按 student 自己的分布分配修正：对 a_t 的拉力约为 c_t·log(q_max/c_t)，c_t → 0 时消失；
从错误选项移走的质量也按 p 重新分配，会流向 teacher 同样否定的候选（“只排除、不搬运”）。
固定 β 的 JSD（GKD）拉力同样会消失；全局 FKL 拉力不消失，但在 teacher 不确定的位置也会把 student 拉糊。

**方法**：一个量 c_t，三步，对应三个命题。

1. **门控（命题 1）**：`β_t = sg[1(q_max ≥ τ_q)·(1 − c_t)]`，默认 τ_q=0.5。
   拉力不消失要求 1 − β_t = O(c_t)，再要求 student 同意时（c_t → 1）精确退回 RKL，就得到 1 − c_t。
   teacher 没有明确首选（q_max < τ_q）时 a_t 没有意义，取 β=0，即精确 RKL。
2. **归一化 skew-JSD（命题 2）**：`m = (1−β)q + βp`，`D = KL(q‖m)/β + KL(p‖m)/(1−β)`。
   β=0 时精确为 RKL，β=1 时精确为 FKL；小分歧时尺度与 KL 一致，换 β 不用重调学习率。
   其中 1/(1−β) = 1/c_t 正好抵消 RKL 的 c_t 权重，对 a_t 的拉力变成 log(1 + q_max − c_t)：不消失，又被对数压缩。
3. **teacher 侧限幅（命题 3）**：c_t → 0 时只有 KL(q‖m)/β 会发散，把它裁到 C_F，student 侧不裁。
   C_F 取首个 microbatch 门控位置上 β·FKL 的 p95，之后冻结。裁剪区的拉力降到有界的 κ(q_max) ≈ 0.07–0.19，
   相当于朝 a_t 的弱 CE。

实现见 `modules/vOPD/src/token_gate.py`，主配置是 `configs/deepseek_token_gate/04_jsd_q05.yaml`（下文称 A04）。

**现状。** DeepSeek-R1-Distill-Qwen-1.5B ← JustRL-DeepSeek-1.5B，单 seed：32k avg@16 五项宏平均 A04 52.66，纯 RKL 51.62，
差 +1.04，区间 [−0.22, +2.34]（只对题目 bootstrap）；8k avg@4 上是 47.17 vs 43.32。
已经做过（都在 8k avg@4 下）τ_q 扫描、JSD vs KL 混合、ψ 的形状，以及 stop 通道、方案 B、偏好差门控、事后门控、CE 替换等附加部件（都没有带来提升）。
**三个组件还没有单独证明有效：限幅、门控 vs 常数 β、归一化**。下面五组实验就是补这三个消融。

## vOPD DeepSeek：五组核心消融（限幅 / 常数 β / 归一化）

以下命令都在 **OPD 仓库根目录**执行；配置、数据、模型和新环境均位于本目录内。
配置目录：`configs/deepseek_core_ablation/`。每组改了什么、数学定义和对照组见
[configs/deepseek_core_ablation/README.md](configs/deepseek_core_ablation/README.md)。
没有外部实验目录或旧机器环境依赖。只跑 seed=42，不扫多 seed。

| 编号 | 配置 | 改动（相对 JSD q05 / A04） | 对照（已有） |
| --- | --- | --- | --- |
| 01 | `01_jsd_q05_noclip.yaml` | 去掉限幅，C_F=∞ | A04 |
| 02 | `02_mix_q05_noclip.yaml` | KL 混合 q05，去掉限幅 | A01 |
| 03 | `03_jsd_beta05.yaml` | 不门控，所有位置固定 β=0.5 的归一化 JSD，不限幅 | A04 |
| 04 | `04_fkl.yaml` | 不门控，所有位置精确 FKL，不限幅 | A04、A07 |
| 05 | `05_jsd_q05_unnorm.yaml` | 去掉 1/(1−β) 归一化，限幅同 A04 | A04 |

### 1. 在目标机器创建环境（首次运行）

目标机器使用 Linux x86_64、Python 3.12，以及可运行 PyTorch 2.8 / CUDA 12.8 的 NVIDIA GPU 环境。
不要上传或复用旧机器的虚拟环境，在目标机器重新创建：

```bash
bash scripts/setup_vopd.sh
```

它依据 `requirements-vopd.lock.txt` 安装到本仓库的 `.venv-vopd/`。
配置中的 `run.python: null` 表示沿用启动命令所使用的 Python。

### 2. 准备本仓库内的数据与模型

若 `data/` 下没有数据文件，运行以下命令会按固定版本下载并转换。
AMC23 只保留 2023 年的 40 题，AIME25 只使用完整的 30 题，不叠加 part1/part2。

```bash
.venv-vopd/bin/python scripts/prepare_vopd_assets.py
```

两个模型应位于 `model/DeepSeek-R1-Distill-Qwen-1.5B/` 和 `model/JustRL-DeepSeek-1.5B/`。
如果目标机器上没有，按固定版本下载：

```bash
.venv-vopd/bin/hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --local-dir model/DeepSeek-R1-Distill-Qwen-1.5B

.venv-vopd/bin/hf download hbx/JustRL-DeepSeek-1.5B \
  --revision 0637e4096c789c67f9eecbe8355e0bdeddede1c2 \
  --local-dir model/JustRL-DeepSeek-1.5B
```

### 3. 检查代码与配置（不占 GPU）

```bash
.venv-vopd/bin/python -m unittest tests.test_vopd_core_ablation
bash configs/deepseek_core_ablation/run_all.sh --dry-run
```

单元测试检查三种新开关的损失和梯度，并确认五个配置除被消融的键外与 A04 完全相同。
`--dry-run` 只检查配置及文件路径，不启动训练和评测。

### 4. 分别运行五组训练实验

每条命令都包含 **训练 → 保存 final → 自动评测 final → complete-box 重评分**。
请等待上一条命令结束后再执行下一条；也可以使用下方顺序脚本。

```bash
# 实验 1：JSD q05，去掉 teacher 侧限幅（C_F=∞）
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_core_ablation/01_jsd_q05_noclip.yaml

# 实验 2：KL 混合 q05，去掉限幅；与实验 1、A04、A01 组成 2×2
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_core_ablation/02_mix_q05_noclip.yaml

# 实验 3：不门控，固定 β=0.5 的归一化 JSD
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_core_ablation/03_jsd_beta05.yaml

# 实验 4：不门控，全局 FKL
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_core_ablation/04_fkl.yaml

# 实验 5：JSD q05 去掉 1/(1−β) 归一化，C_F 固定为 A04 标定值
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_core_ablation/05_jsd_q05_unnorm.yaml
```

依次运行五组训练（任一组失败即停止）：

```bash
bash configs/deepseek_core_ablation/run_all.sh
```

### 5. 评测、保存与结果位置

- 所有配置使用 GPU 0–7，需要八张空闲卡；启动时任一卡已占用超过 1 GiB 显存会直接退出，不会与别的任务重叠。
  按此前的训练和 32k 评测耗时估算，八卡上每组约 1.5–2 小时，五组约 8–10 小时。
- 不要改卡数和梯度累积。只有四张卡时，可以在命令末尾加
  `'run.gpus=[0,1,2,3]' training.gradient_accumulation_steps=16 evaluation.max_parallel_workers=4`，
  保持全局 batch=64、评测仍为 8 个题目分片（分两轮），但要在结果里注明，因为训练随机流会变。
- 训练 211 步，全局 batch=64，seed=42，LR 等其余设置与 A04 一致。
  启动时会核对初始 LoRA 的 SHA256；如果报 `Paired LoRA initialization mismatch`，请记录报错并告知，不要删掉该字段后继续跑。
- 训练成功后自动评测 **final checkpoint**。50、100、150、200、211 步 checkpoint 都会保存，
  **中间 checkpoint 不会自动评测**。
- 评测 MATH500、Minerva、AMC23、AIME24、AIME25 五个 benchmark，每题 8 条；
  temperature=0.6、top_p=0.9、回答上限 32768 tokens（8192×4，总长度上限 33792）。
- 结果在 `outputs/vOPD/dapo14k_deepseek_core_<配置名>/<日期>/<编号>/` 下，每次启动创建新编号。
  每次运行保留 `main.log`、`configs/`、`code/`、`scores.json`、`results.jsonl`、
  `diagnostics/complete_box_scores.json` 和 `checkpoint/`。
- 成绩统一读 `diagnostics/complete_box_scores.json` 中的 **`last_complete`**；宏平均为五项 avg@8 等权平均。
- 跑完后在 `main.log` 的训练日志里顺手确认开关生效：01–04 的 `token_gate/clip_fraction` 恒为 0；
  03、04 的 `token_gate/gate_fraction` 为 1，`token_gate/beta` 分别为 0.5、1.0；
  05 的 `token_gate/clip_fraction` 应与 A04 同量级（均值约 0.6%）。

### 6. 填写结果

跑完一组填一行（每格填 avg@8），在完成人一栏写自己的名字。前几行是已有对照，不需要重跑。
已有结果同样是回答上限 32768、`last_complete` 判分，但每题 16 条（avg@16）。
avg@n 的期望不随 n 变化，可以直接和 avg@8 比，只是方差更小。
32k 下只有 A04 与其 4 卡梯度等价复现 B2048_psi0 这一对（差 0.68pp），运行间噪声还没有可靠估计，单组小差值不要下结论。
A01 还没有 32k 结果，02 组暂时缺这一格对照。

| ID | 实验 | MATH500 | Minerva | AMC23 | AIME24 | AIME25 | 宏 avg | 完成人 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B2048_psi0 | JSD q05，带限幅（A04 的梯度等价复现，4 卡；已有，avg@16） | 84.39 | 31.41 | 77.34 | 42.92 | 30.63 | 53.34 | — |
| A07 | 纯 RKL（已有，avg@16） | 82.95 | 32.01 | 74.38 | 42.71 | 26.04 | 51.62 | — |
| A01 | KL 混合 q05，带限幅（32k 未测） | | | | | | | |
| 01 | JSD q05，无限幅 | | | | | | | |
| 02 | KL 混合 q05，无限幅 | | | | | | | |
| 03 | 固定 β=0.5 JSD | | | | | | | |
| 04 | 全局 FKL | | | | | | | |
| 05 | JSD q05，无归一化 | | | | | | | |

### 7. 可选：Minerva 重新判分（不改主流程）

主流程和上表一律用原判分器。原判分器在 Minerva 上会漏判正确答案：272 题中有 58 题的标准答案是 `4.5e33`
这种 Python 科学计数法，模型写成 `4.5 \times 10^{33}` 一律判错；很多小数标准答案是四舍五入后的值，模型多写一位也会判错。

`scripts/rescore_minerva.py` 只对 Minerva 的最后一个完整答案框重新判分：只会把错改对，不会把对改错；
其它 benchmark 原样保留，也不修改 `scores.json` 和 `complete_box_scores.json`。

- 标准答案是单个数时：预测也必须是单个数（可带单位），按标准答案的有效数字四舍五入后相等；整数答案必须完全相等。
- 其它 Python 写法的标准答案（`np.arcsin`、`3./2`）改写成 LaTeX 后交给原判分器。

```bash
.venv-vopd/bin/python scripts/rescore_minerva.py outputs/vOPD/dapo14k_deepseek_core_01_jsd_q05_noclip/<日期>/<编号>
```

可以一次传多个运行目录。结果写到各运行的 `diagnostics/minerva_normalized_scores.json` 和 `minerva_normalized_cases.jsonl`。
在已有 32k 结果上，它让各模型的 Minerva 都提高约 8pp（RKL 32.01 → 40.00，B2048_psi0 31.41 → 39.59），
模型之间的差几乎不变。需要时在表后另附一列，不要和原判分器的数字混在同一列。
