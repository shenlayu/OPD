# Skywork-OR1-RL-Data

旧目录中使用过的数学训练来源，仅保留下载占位；默认 OPD 链路不依赖它。

```bash
hf download Skywork/Skywork-OR1-RL-Data --repo-type dataset \
  --revision 1cdedc52e0e2db85fdf252f9be682e63a5a38c33 \
  --include 'data/math-*.parquet' --local-dir data/Skywork-OR1-RL-Data
```

切换 rethinking 数据前，应将其转换为作者的 `prompt`（chat 消息列表）、
`reward_model.ground_truth`、`data_source` 格式。历史数据筛选和混合实验脚本不迁移。
