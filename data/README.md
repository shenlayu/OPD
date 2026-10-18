# 数据下载占位

数据未复制，以下命令从 OPD_lite 根目录执行；安装对应环境的 `datasets` 和 `huggingface_hub`。
为两个框架保留各自题目格式和评分协议，大小写目录有意区分。

## 默认 vOPD

- [dapo14k](dapo14k.md)
- [math500](math500.md)
- [minerva](minerva.md)
- [amc23](amc23.md)
- [aime24](aime24.md)
- [aime25](aime25.md)

## 默认 rethinking

- [DAPO-Math-17k](dapo-math-17k.md)
- [AIME24](rethinking-AIME24.md)
- [AIME25](rethinking-AIME25.md)
- [AMC23](rethinking-AMC23.md)

## 其他已用数据与作者附带的可选评测集

- [DAPO 原始数据与 boxed 派生集](dapo-math-17k-boxed.md)
- [OpenThoughts3 OPD prompts](OpenThoughts3_opd.md)
- [Skywork 数学数据](Skywork-OR1-RL-Data.md)
- [MATH-500](rethinking-MATH-500.md)
- [Minerva](rethinking-Minerva.md)
- [Olympiad-Bench](rethinking-Olympiad-Bench.md)
- [BRUMO25](rethinking-BRUMO25.md)
- [CMIMC25](rethinking-CMIMC25.md)
- [HMMT25](rethinking-HMMT25.md)

如果使用已有下载，在 YAML 中直接填写本地路径即可。vOPD 接受本地 Hub snapshot、
`save_to_disk` 目录、parquet、JSON/JSONL；rethinking 接受作者格式的 parquet 文件列表。
下载命令依据原目录中的资源清单和固定提交整理，本次迁移未重新下载资产。
