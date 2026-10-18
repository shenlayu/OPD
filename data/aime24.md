# aime24

资源占位，未迁移数据。来源：[HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)。

在 OPD_lite 根目录执行：

```bash
hf download HuggingFaceH4/aime_2024 --repo-type dataset \
  --revision 2fe88a2f1091d5048c0f36abc874fb997b3dd99a --local-dir data/aime24/raw
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
files = sorted(str(p) for p in Path('data/aime24/raw').rglob('*.parquet'))
kind = 'parquet'
assert files, 'No benchmark files downloaded'
rows = load_dataset(kind, data_files=files, split='train')
rows.to_parquet('data/aime24/test.parquet')
PY
```

原始来源 split：`train`；本地统一命名 `test.parquet`，不改变题目和答案。
