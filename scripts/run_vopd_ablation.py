"""Run one ablation while reserving its GPUs through training and evaluation."""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._runtime import launch, load_config, validate_assets


@contextmanager
def reserve_gpus(config):
    """Share the same reservation between student evaluation and training."""
    with ExitStack() as stack:
        inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits",
        ], text=True)
        devices = {int(parts[0]): (parts[1], int(parts[2]))
                   for line in inventory.splitlines()
                   if (parts := [s.strip() for s in line.split(",")])}
        lock_root = Path(os.environ.get("OPD_GPU_LOCK_ROOT", ROOT / ".gpu_locks")).resolve()
        lock_root.mkdir(parents=True, exist_ok=True)
        for gpu in sorted(config.run.gpus):
            uuid, used = devices[gpu]
            lock = stack.enter_context((lock_root / f".opd_gpu_{uuid}.lock").open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if used > 1024:
                raise RuntimeError(f"GPU {gpu} is occupied ({used} MiB); refusing to overlap")
        print(f"Reserved physical GPUs {list(config.run.gpus)} through training and evaluation", flush=True)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config("vOPD", args.config, args.overrides)
    if not config.training.dataset_name:
        raise ValueError("Use scripts/evaluate_vopd_student.py for the student-only config")
    validate_assets(config, "vOPD")
    if args.dry_run:
        return launch("vOPD", "vopd.yaml")
    with reserve_gpus(config):
        return launch("vOPD", "vopd.yaml")


if __name__ == "__main__":
    raise SystemExit(main())
