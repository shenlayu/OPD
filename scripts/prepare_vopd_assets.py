"""Prepare the exact vOPD datasets used by the DeepSeek main experiment."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Exact revisions/files avoid mixing AMC years or duplicate AIME25 subsets.
SOURCES = [
    ("dapo14k", "guanning-ai/dapo14k", "42edfa38f7f2ac452eb12e20049b2b3e2efc6651", "train",
     ["data/train-00000-of-00001.parquet"], 13542),
    ("math500", "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test",
     ["test.jsonl"], 500),
    ("minerva", "math-ai/minervamath", "ee46ddc498933b1977577953250ca5c66be64f96", "test",
     ["test.jsonl"], 272),
    ("amc23", "AI-MO/aimo-validation-amc", "69d78a4a2c840e82d69af6bc742bda09005f6316", "train",
     ["data/train-00000-of-00001.parquet"], 40),
    ("aime24", "HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", "train",
     ["data/train-00000-of-00001.parquet"], 30),
    ("aime25", "yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f", "train",
     ["data/train-00000-of-00001-243207c6c994e1bd.parquet"], 30),
]


def prepare(source, *, check=False):
    import pyarrow.parquet as parquet

    name, repo, revision, split, files, expected = source
    target = ROOT / "data" / name
    destination = target / ("train.parquet" if name == "dapo14k" else "test.parquet")
    metadata = target / "source.json"
    if destination.exists():
        if not metadata.exists():
            raise ValueError(f"Missing provenance: {metadata.relative_to(ROOT)}")
        saved = json.loads(metadata.read_text())
        if (saved.get('repo'), saved.get('revision'), saved.get('files')) != (repo, revision, files):
            raise ValueError(f"Existing data differs from the main experiment: {name}")
        actual = parquet.ParquetFile(destination).metadata.num_rows
        if actual != expected:
            raise ValueError(f"{name}: expected {expected} rows, got {actual}")
        print(f"OK {destination.relative_to(ROOT)}: {actual} rows")
        return
    if check:
        raise FileNotFoundError(f"Missing {destination.relative_to(ROOT)}; run this script without --check")
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    raw = Path(snapshot_download(repo, repo_type="dataset", revision=revision,
                                 allow_patterns=files, local_dir=target / "raw"))
    kind = "parquet" if files[0].endswith(".parquet") else "json"
    rows = load_dataset(kind, data_files=[str(raw / f) for f in files], split="train")
    filtering = None
    if name == "amc23":
        filtering = "Keep URL containing /2023_AMC_ (exclude the 2022 questions)"
        rows = rows.filter(lambda row: "/2023_AMC_" in row["url"])
    if len(rows) != expected:
        raise ValueError(f"{name}: expected {expected} rows, got {len(rows)}")
    rows.to_parquet(destination)
    metadata.write_text(json.dumps(dict(repo=repo, revision=revision, source_split=split,
        files=files, rows=len(rows), columns=rows.column_names, filtering=filtering), indent=2) + "\n")
    print(f"Prepared {destination.relative_to(ROOT)}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify local datasets without downloading")
    parser.add_argument("--evaluation-only", action="store_true", help="Skip the training dataset")
    args = parser.parse_args()
    for source in SOURCES:
        if args.evaluation_only and source[0] == "dapo14k":
            continue
        prepare(source, check=args.check)


if __name__ == "__main__":
    main()
