# math500

资源占位，未迁移数据。来源：[HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)。

在 OPD_lite 根目录执行：

```bash
hf download HuggingFaceH4/MATH-500 --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be --local-dir data/math500/raw
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
files = [str(Path('data/math500/raw/test.jsonl'))]
kind = 'json'
assert files, 'No benchmark files downloaded'
rows = load_dataset(kind, data_files=files, split='train')
rows.to_parquet('data/math500/test.parquet')
PY
```

原始来源 split：`test`；本地统一命名 `test.parquet`，不改变题目和答案。
