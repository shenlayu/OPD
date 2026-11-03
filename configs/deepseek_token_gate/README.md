# DeepSeek 逐 token 散度扫描（七组）

对应 2026-10-29 提供的《OPD 逐 token 散度选择方案》方案 A。用户确认的六组矩阵，
最后追加一组不带 stop 辅助项的 RKL；不执行方案 B、时间调度或 k_odds>1 扫描。

| 顺序 | 配置 | 散度 | τ_q | 单独停止通道 |
| --- | --- | --- | --- | --- |
| 1 | `01_mix_q05.yaml` | β FKL + (1−β) RKL | 0.5 | 无 |
| 2 | `02_mix_q07.yaml` | 同上 | 0.7 | 无 |
| 3 | `03_mix_q05_stop1.yaml` | 同上，作用于内容条件分布 | 0.5 | k_odds=1 |
| 4 | `04_jsd_q05.yaml` | 归一化 generalized JSD | 0.5 | 无 |
| 5 | `05_jsd_q07.yaml` | 同上 | 0.7 | 无 |
| 6 | `06_jsd_q05_stop1.yaml` | 同上，作用于内容条件分布 | 0.5 | k_odds=1 |
| 7 | `07_rkl.yaml` | 原有完整词表 RKL | 不使用 | 无，stop_weight=0 |

## 共同训练与评测协议

- vOPD；student=`DeepSeek-R1-Distill-Qwen-1.5B`，teacher=`JustRL-DeepSeek-1.5B`。
- DAPO14k，同一输入顺序与 seed=42；8 GPU，211 次更新；每卡 batch=1，梯度累积=8，全局 batch=64。
- LoRA r64/alpha128/dropout0.05，LR=1e-5 cosine；训练生成上限 2048，temperature=1。
- 保留初始 adapter SHA256 核验、gradient checkpointing、完整词表损失与 128-token loss chunk。
- 每组训练成功后评测 final，再做 complete-box 重评分；保留中间 checkpoint，但不逐个评测。
- 五个 benchmark：MATH500、Minerva、AMC23、AIME24、AIME25，全部完整数据、**每题 4 条**。
- 评测 temperature=0.6、top_p=0.9、seed=42，回答上限 **8192**，总长度上限 9216，8 个独立评测引擎。
- 生成仍只在 EOS `151643` 终止，`</think>` `151649` 只参与损失中的停止通道与监测。

## 具体数学约定

记 p=student、q=teacher，均为当前前缀的完整词表 softmax。

`beta = stop_gradient(1[max(q)>=tau_q] * (1-p(argmax(q))))`。
门控使用完整分布，在停止通道分解前计算，并且不随训练步数变化。
现有 vOPD 已有两侧完整 logits，因此这里使用精确完整词表 FKL/RKL，替代文档中为节省
teacher 通信而建议的 top-k 近似，不新增 top-k 截断因素。

JSD 采用已有代码的归一化约定：`alpha=1-beta`，
`m=alpha*q+(1-alpha)*p`，`JSD=KL(q||m)/(1-alpha)+KL(p||m)/alpha`。
alpha=0 精确为 FKL，alpha=1 精确为 RKL；内部参数裁剪到 [0.001,0.999] 保证 FP32 稳定。
因此未通过 teacher 置信门控的位置精确使用 RKL。

停止集合为 `{EOS, </think>}`，合并成一个二元事件 s。启用停止通道时，
`L=KL_B(q_s||p_s)+stop_gradient(1-p_s)*D(q_c,p_c)`。
内容分布移除整个停止集合后重新归一化；不另加旧版的全词表 stop auxiliary，
也不单独监督两个停止 token 在停止集合内部的相对比例。
门控和内容质量权重都停止梯度，故停止方向的梯度不会混入内容散度大小。

前六组用逐序列平均后再对 batch 平均。第七组沿用原 RKL 实现；在本组每卡 microbatch=1
的固定设置下，其 token mean 同样等于逐序列 mean，然后经累积和跨卡平均。

### Forward 分量裁剪

每个实验在第一次训练 microbatch（跨 8 卡、共 8 条序列）中，用门控开启位置的
`beta*FKL` 95 分位数确定 C_F，之后冻结。排除 padding；没有额外模型前向，也不消耗训练 RNG。
记录位置：`diagnostics/token_gate_calibration.json`。

- KL 混合组裁剪 `beta*FKL`。
- JSD 组用同一参考量标定阈值，并裁剪对应的 teacher forward 分量 `KL(q||m)/beta`；
  reverse 分量不裁剪。停止通道的 Bernoulli 项也不裁剪。
- 这属于带 forward 分量裁剪的 JSD 变体，日志记录实际裁剪比例。第七组不做该裁剪。
- 如需恢复中间 checkpoint，必须把记录的 `clip` 显式传入 `training.token_gate_clip`，
  防止用训练中途的分布重新标定。自动队列不会重试失败任务。

监测包括 beta、门控比例、q_max、student 对 teacher 首选的概率、FKL/RKL、
两部分损失、裁剪比例、尾部质量、停止概率及每 256 token 的停止概率分段。
标准训练日志还保留生成长度和梯度范数。这七组不启用方案 B；其后新增的可选实现见
[方案 B 说明](../deepseek_token_progress/README.md)。按门控划分的参数梯度范数仍未实现。

## 启动和排队

在 OPD 根目录用 vOPD Python 环境执行单组，例如：

```bash
.venv-vopd/bin/python scripts/run_vopd_ablation.py \
  --config configs/deepseek_token_gate/01_mix_q05.yaml
```

加 `--dry-run` 只检查配置。环境安装方式见根目录 TODO.md。

冻结源码和配置并后台挂七组串行队列：

```bash
.venv-vopd/bin/python scripts/queue_vopd_token_gate.py \
  --scheduler /home/coder/code/canva/scripts/GPU_schedule/gpu_schedule.py
```

本机本次复用已验证的 vOPD 环境：
`/home/coder/code/OPD/OPD_vopd_stop_20261026/.venv-vopd/bin/python`。
仓库 YAML 保持相对路径与 `run.python: null`；排队时解析并冻结本机绝对路径。

调度器等待物理 0–7 全部连续空闲 **300 秒**（每 10 秒检查，任何卡重新占用会重置计时）。
每个任务从训练到评测完成始终持有 GPU 预约；失败即停止队列。
第一组正式训练前，先执行一次八卡 1-step JSD+stop 预检，并在五个 benchmark 各取一题、
每题 4 条、最多 64 tokens 验证评测链路。预检通过后才开始正式第一组；预检不计入七组结果。

队列目录位于 `outputs/queues/deepseek_token_gate_<北京时间>/`，保留：

- `jobs.json`：七个正式任务，第一项包含预检。
- `code/`、`sha256.json`：不可变运行源码、配置与摘要；后续修改工作区不会影响已挂队列。
- `scheduler.json`：PID、完整命令和启动时间。
- `scheduler.log`、`logs/*/results.jsonl`：调度日志、任务退出码。
- `preflight/`：预检结果，和正式结果分开。

正式结果位于 `outputs/vOPD/dapo14k_deepseek_token_<配置名>/<日期>/<编号>/`。
可以用 `tail -f <队列目录>/scheduler.log` 查看进度；
`kill -TERM <scheduler.json 中的 PID>` 取消等待并停止本队列启动的任务。
不要直接重跑旧 jobs.json：调度器从第一项开始，不会跳过已完成实验。

## 验证边界

CPU 测试覆盖门控 detach、KL/JSD 值和梯度、关闭门控时的精确 RKL、停止梯度、
forward clip、含 NaN padding 的 sequence mean、分块梯度、跨两进程标定及七组配置一致性。
GPU 预检只证明端到端训练和评测链路能运行，不代表正式训练质量。

2026-10-29 本机验证：调度器 26 项测试、vOPD 33 项相关测试、评测 14 项测试通过；
UUID 倒序映射经真实设备查询验证。八卡 1-step JSD+stop 预检完成，五项短评测共 20 条记录完整，
complete-box 重评分成功；正式第一组已完成首个更新且 loss/梯度有限。
原队列：`outputs/queues/deepseek_token_gate_20261029-220150/`，调度 PID `3523696` 已按用户要求停止。

2026-10-29 22:35（北京时间）按用户新指令重新排队：
`outputs/queues/deepseek_token_gate_20261029-220150/after_teacher_ood_20261029-223519/`，
等待/调度 PID `3614064`。等待 teacher_ood_check 当前渲染与 Gemini/RM 打分流程成功结束，
校验 8 个渲染分片和 judge+8 个 RM 分片的成功退出记录，确认原进程退出后，
才启动 GPU 调度器的连续空闲 300 秒计时。前置任务失败或中断则阻止启动。
`dependency.json` 保存前置进程身份和检查条件，`state.json` 保存状态，`dispatch.log` 保存运行日志。
第一组停在第一个 checkpoint 保存前，因此从 step 0 新建运行，保留旧日志；其余六组不变。
复用冻结源码和已通过的八卡预检，不重做预检。`kill -TERM 3614064` 可以取消本次等待；
完成交接后同一 PID 会替换成调度器，仍可用该命令停止本队列。

### 当前队列与后续补评（2026-10-30）

上述依赖等待已被替代：渲染和 RM 均成功完成，额外的 Gemini 阶段因鉴权失败未完成。
用户随后要求直接启动，七组现由 PID `3777647` 顺序执行，当前调度目录为：
`outputs/queues/deepseek_token_gate_20261029-220150/direct_start_20261029-235105/`。
本次按用户指令取消五分钟等待，仍保留显卡空闲检查与调度锁。

用户追加的三个任务已经挂在上述完整七组之后，配置位于
`configs/deepseek_token_gate_followup/`：

1. `08_student_math_minerva.yaml`：原始 student，无训练、无 adapter；补评 MATH500 500 题、
   Minerva 272 题，每题 4 条、回答上限 8192，共 3088 条记录，再做 complete-box 重评分。
2. `09_mix_q03.yaml`：KL 混合，新增置信阈值 τ_q=0.3，不拆停止通道。
3. `10_jsd_q03.yaml`：JSD，新增置信阈值 τ_q=0.3，不拆停止通道。

新增两组训练除阈值和输出名称外，与正在运行的相应 τ_q=0.5 配置一致；
均为 211 次更新，最后评测五项 benchmark、每题 4 条。阈值 0.3 是追加时建议的默认值。
baseline 补评采用当前八个独立推理引擎；旧 AMC23/AIME24/AIME25 结果来自四个引擎，
采样、长度、评分协议相同，但不宣称更换分片后随机流完全一致。

后续队列：`outputs/queues/deepseek_token_followup_20261030-011125/`，等待 PID `3955543`。
`dependency.json` 绑定当前调度 PID 的启动身份及原 jobs.json 摘要，要求七个任务的退出码
全部为零且原调度进程退出，才依次运行这三个任务。等待期间不占 GPU，不会在原队列
两个任务之间提前启动；原队列失败则停止等待，不会重跑已经完成的任务。
`state.json`、`dispatch.log` 保存等待状态；`jobs.json`、`code/`、`sha256.json` 保存冻结执行内容。
训练与评分源码复用原七组快照；只修复 student 启动器对调度器 GPU UUID 顺序的保留。
本次通过 17 项依赖等待测试、6 项 student/可迁移性测试，以及三个任务和调度器 dry-run；
尚未执行新增 GPU 任务。

### JSD 阈值 0.4 / 0.6（2026-10-30）

根据已完成的 `04_jsd_q05` 继续扫描，只改 teacher 置信门控阈值：
`configs/deepseek_token_gate_followup/11_jsd_q04.yaml` 和
`configs/deepseek_token_gate_followup/12_jsd_q06.yaml`。
两组均不拆停止通道，stop_weight=0；从原始 student 开始训练，seed=42、211 步、八卡，
最终五项 benchmark 各 rollout=4、8192 回答长度、complete-box 重评分。
forward clip 仍按各自首个 microbatch 标定，初始化 adapter 摘要检查保留。

队列 `outputs/queues/deepseek_jsd_q04_q06_20261030-092621/` 按 0.4 → 0.6 顺序执行。
源码与 `04_jsd_q05` 实际运行快照逐文件一致；新配置除阈值、输出名称及解释器路径外一致。
两组配置和队列 dry-run 均通过，冻结内容见 `code/`、`sha256.json`，
启动信息见 `dispatch.json`，运行进度见 `dispatch.log` 与 `logs/*/results.jsonl`。
沿用用户直接启动指令，idle-seconds=0，保留两次空闲检查与调度锁；失败即停止队列。
