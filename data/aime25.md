# aime25

资源占位，未迁移数据。来源：[yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)。

在 OPD_lite 根目录执行：

```bash
hf download yentinglin/aime_2025 --repo-type dataset \
  --revision 6f71d77b0b89b9dabe07ab466c51df33f514df7f --local-dir data/aime25/raw
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
files = sorted(str(p) for p in Path('data/aime25/raw').rglob('*.parquet'))
kind = 'parquet'
assert files, 'No benchmark files downloaded'
rows = load_dataset(kind, data_files=files, split='train')
rows.to_parquet('data/aime25/test.parquet')
PY
```

原始来源 split：`train`；本地统一命名 `test.parquet`，不改变题目和答案。
