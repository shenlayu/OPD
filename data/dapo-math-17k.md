# dapo-math-17k

资源占位，使用 rethinking 作者发布的已处理 parquet，保留原 prompt 和 ground_truth。
固定来源：[Rethinking-OPD / dapo-math-17k.parquet](https://github.com/Thinking-Space/Rethinking-OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/datasets/dapo-math-17k.parquet)。

在 OPD_lite 根目录执行：

```bash
mkdir -p data/rethinking
curl -fL 'https://raw.githubusercontent.com/Thinking-Space/Rethinking-OPD/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/datasets/dapo-math-17k.parquet' -o data/rethinking/dapo-math-17k.parquet
```

配置 `training.data.train_files: [data/rethinking/dapo-math-17k.parquet]`。
