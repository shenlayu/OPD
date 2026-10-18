# minerva

资源占位，未迁移数据。来源：[math-ai/minervamath](https://huggingface.co/datasets/math-ai/minervamath)。

在 OPD_lite 根目录执行：

```bash
hf download math-ai/minervamath --repo-type dataset \
  --revision ee46ddc498933b1977577953250ca5c66be64f96 --local-dir data/minerva/raw
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
files = [str(Path('data/minerva/raw/test.jsonl'))]
kind = 'json'
assert files, 'No benchmark files downloaded'
rows = load_dataset(kind, data_files=files, split='train')
rows.to_parquet('data/minerva/test.parquet')
PY
```

原始来源 split：`test`；本地统一命名 `test.parquet`，不改变题目和答案。
