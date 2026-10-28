"""Evaluate the original DeepSeek student without training or loading an adapter."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from scripts._runtime import allocate_run, load_config, run_command, snapshot_code, validate_assets
from scripts.run_vopd_ablation import reserve_gpus


def evaluate_student(config, requested_config):
    if config.training.use_peft or config.training.resume_from_checkpoint:
        raise ValueError("Student evaluation must use original weights without LoRA or a resumed checkpoint")
    if not config.evaluation.enabled:
        raise ValueError("Student evaluation requires evaluation.enabled=true")
    validate_assets(config, "vOPD", evaluation_only=True)
    with reserve_gpus(config):
        run = allocate_run(config.run.output_root, "vOPD", config.run.dataset)
        (run / "configs").mkdir()
        snapshot_code(ROOT, run / "code", "vOPD")
        shutil.copy2(requested_config, run / "configs/requested.yaml")
        python = config.run.python or sys.executable
        config.run.python = python
        config_path = run / "configs/vopd.yaml"
        OmegaConf.save(config, config_path, resolve=True)
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(map(str, config.run.gpus)),
            PYTHONPATH=str(run / "code"), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS=str(config.run.threads), MKL_NUM_THREADS=str(config.run.threads),
            TOKENIZERS_PARALLELISM="false",
        )
        record = dict(mode="evaluate_student", module="vOPD", status="running",
                      checkpoint=config.training.model_name_or_path,
                      started_at=datetime.now(timezone.utc).isoformat())
        def status(**updates):
            record.update(updates)
            (run / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        status(phase="evaluation")
        print(f"Student evaluation: {run}", flush=True)
        with (run / "main.log").open("w", buffering=1) as log:
            try:
                run_command(
                    [python, "-m", "scripts._evaluate", str(config_path), "vOPD",
                     config.training.model_name_or_path, str(run)],
                    env, run / "code", log,
                )
                if config.evaluation.get("complete_box_scoring", False):
                    status(phase="complete_box_scoring")
                    run_command([python, "-m", "scripts.diagnose_complete_boxes", str(run)],
                                env, run / "code", log)
                status(status="completed", phase="completed", finished_at=datetime.now(timezone.utc).isoformat())
            except BaseException as error:
                status(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}")
                raise
        return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deepseek_jsd_ablation/00_student.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config("vOPD", args.config, args.overrides)
    if config.training.use_peft or config.training.resume_from_checkpoint or not config.evaluation.enabled:
        raise ValueError("Use a student-only evaluation config with use_peft=false and evaluation enabled")
    validate_assets(config, "vOPD", evaluation_only=True)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print("Mode: student evaluation only; no training or adapter")
        return 0
    evaluate_student(config, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
