# rethinking / AIME25

作者提供的评测 parquet 占位；保留原始 prompt（不重复追加数学指令）和 reward_model.ground_truth。
默认链路只评估 AIME24、AIME25、AMC23；其余为原目录附带的可选 benchmark。

```bash
mkdir -p data/AIME25
curl -fL 'https://raw.githubusercontent.com/Thinking-Space/Rethinking-OPD/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/scripts/val/data/AIME25/test.parquet' -o data/AIME25/test.parquet
```

下载后可加入 `evaluation.datasets`，设置 `path: data/AIME25/test.parquet`、`split: test`、`samples: 16`、`limit: null`。
