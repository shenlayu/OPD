#!/usr/bin/env python3
"""Keep complete HF models, removing training shards only after successful evaluation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def complete_evaluation(folder):
    """Require complete coverage, valid grades, and the original answer checksum."""
    path = folder / "scores.json"
    if not path.exists():
        return None
    data = read_json(path)
    benchmarks = data.get("benchmarks", {})
    if data.get("status") != "complete" or not benchmarks:
        return None
    for item in benchmarks.values():
        expected = item.get("expected_samples", 0)
        if (not item.get("complete") or expected < 1
                or item.get("completed_samples") != expected
                or item.get("graded_samples") != expected or item.get("grading_errors") != 0):
            return None
    with (folder / "results.jsonl").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != data.get("results_sha256"):
        raise ValueError(f"Evaluation checksum mismatch: {folder}")
    return data


def weight_files(folder):
    """Reject configuration-only exports and incomplete or unsafe shard indexes."""
    for name in ("config.json", "tokenizer_config.json"):
        if not (folder / name).is_file():
            raise ValueError(f"Missing {name}: {folder}")
    index = next((folder / n for n in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                  if (folder / n).is_file()), None)
    if index:
        names = set(read_json(index).get("weight_map", {}).values())
    else:
        names = {n for n in ("model.safetensors", "pytorch_model.bin") if (folder / n).is_file()}
    if not names or any(Path(n).name != n for n in names):
        raise ValueError(f"No complete HF weights: {folder}")
    files = [folder / n for n in sorted(names)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in files):
        raise ValueError(f"Missing HF weight shard: {folder}")
    return files


def validate_hf(folder):
    """Actually load all weights and the tokenizer, then run a CPU forward pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    files = weight_files(folder)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        folder, local_files_only=True, torch_dtype="auto", device_map="cpu",
        attn_implementation="eager", output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"Incomplete model at {folder}: {key}={info[key]}")
    model.eval()
    with torch.inference_mode():
        logits = model(**tokenizer("1 + 1 =", return_tensors="pt")).logits
        if not torch.isfinite(logits).all():
            raise ValueError(f"Non-finite CPU inference: {folder}")
    return {"validation": "full CPU model/tokenizer load and finite forward pass",
            "weights": {p.name: p.stat().st_size for p in files}}


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def live_pid(pid):
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def eligible_checkpoints(run):
    """A complete final evaluation qualifies the run; partial runs qualify only scored steps."""
    latest = int((run / "checkpoint/latest_checkpointed_iteration.txt").read_text())
    final = run / "checkpoint/final"
    evidence, sources = {}, {}
    root_scores = complete_evaluation(run)
    if root_scores and Path(root_scores["checkpoint"]).resolve() != final.resolve():
        raise ValueError(f"Final evaluation points to another model: {run}")
    if root_scores:
        sources[latest] = final
    for folder in sorted(run.glob("checkpoint_evaluations/*/step_*")):
        if not re.fullmatch(r"step_\d+", folder.name):
            continue
        scores = complete_evaluation(folder)
        if not scores:
            continue
        step = int(folder.name.split("_")[-1])
        model = Path(scores["checkpoint"]).resolve()
        actor_hf = run / f"checkpoint/global_step_{step}/actor/huggingface"
        if model == final.resolve() and step == latest:
            sources[step] = final
        elif model == actor_hf.resolve():
            sources[step] = actor_hf
        elif model == (folder / "model").resolve():
            marker = read_json(model / ".export_complete.json")
            if Path(marker["checkpoint"]).resolve() != actor_hf.parent.parent.resolve():
                raise ValueError(f"Evaluation/export step mismatch: {folder}")
            sources[step] = model
        else:
            raise ValueError(f"Evaluation belongs to another checkpoint: {folder}")
        evidence[step] = str(folder / "scores.json")
    # A queued continuation still needs the exact optimizer/RNG state at its fork.
    protected = set()
    queue_file = run / "study_queue.json"
    if queue_file.exists():
        queue = read_json(queue_file)
        if queue.get("status") == "running" and live_pid(queue["pid"]):
            protected = {int(item["resume_step"]) for item in queue["runs"]
                         if item.get("resume_step") and item.get("status") != "completed"}
    entries = []
    for checkpoint in sorted((run / "checkpoint").glob("global_step_*"),
                             key=lambda p: int(p.name.split("_")[-1])):
        step = int(checkpoint.name.split("_")[-1])
        if step in protected or not (root_scores or step in evidence):
            continue
        if checkpoint.is_symlink() or (checkpoint / "actor").is_symlink():
            raise ValueError(f"Refusing linked training state: {checkpoint}")
        entries.append((checkpoint, evidence.get(step, str(run / "scores.json")), sources.get(step)))
    return entries


def training_files(actor):
    files = []
    for path in actor.iterdir():
        if path.name == "huggingface":
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"Unsafe HF directory: {path}")
            continue
        if (path.is_symlink() or not path.is_file() or not (
                re.fullmatch(r"(?:model|optim|extra_state)_world_size_\d+_rank_\d+\.pt", path.name)
                or path.name == "fsdp_config.json")):
            raise ValueError(f"Unknown actor content; preserving everything: {path}")
        files.append(path)
    return sorted(files)


def export_hf(run, checkpoint, destination, log):
    from omegaconf import OmegaConf
    config = OmegaConf.load(run / "configs/rethinking.yaml")
    code = run / "code"
    python = config.run.python or sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONPATH=os.pathsep.join([str(code), str(code / "modules/rethinking")]),
               PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    subprocess.run([python, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                    "--local_dir", str(checkpoint / "actor"), "--target_dir", str(destination),
                    "--use_cpu_initialization"], cwd=code, env=env, stdout=log, stderr=log, check=True)
    # Match the launcher's template/stopping protocol exactly (including base -> instruct).
    for name in ("tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        source = Path(config.training.actor_rollout_ref.model.path) / name
        if source.exists():
            shutil.copy2(source, destination / name)
    if config.run.teacher_chat_template:
        subprocess.run([python, "-c", "from modules.rethinking.model_view import apply_teacher_metadata; "
                        "import sys; apply_teacher_metadata(*sys.argv[1:])",
                        config.training.actor_rollout_ref.model.path,
                        config.training.reward_model.model.path, str(destination)],
                       cwd=code, env=env, stdout=log, stderr=log, check=True)


def clean_checkpoint(run, checkpoint, evidence, source):
    actor = checkpoint / "actor"
    files = training_files(actor)
    if not files:
        weight_files(actor / "huggingface")
        return 0
    signatures = {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    hf = actor / "huggingface"
    try:
        weight_files(hf)
        has_weights = True
    except ValueError:
        has_weights = False
    with (run / "checkpoint/storage_cleanup.log").open("a", buffering=1) as log:
        if has_weights:
            validation = validate_hf(hf)
        else:
            # Validate in a private staging directory; never overwrite a complete export.
            with tempfile.TemporaryDirectory(prefix=".hf-export-", dir=checkpoint) as temporary:
                staged = Path(temporary) / "model"
                if source is not None and source != hf:
                    weight_files(source)
                    shutil.copytree(source, staged, copy_function=link_or_copy)
                else:
                    export_hf(run, checkpoint, staged, log)
                validation = validate_hf(staged)
                for path in staged.iterdir():
                    if not path.is_file() or path.is_symlink():
                        raise ValueError(f"Unexpected exported file: {path}")
                # Replace the whole directory: stale indexes from a partial export
                # must not override the newly validated weight files.
                previous = Path(temporary) / "previous_hf"
                if hf.exists():
                    hf.rename(previous)
                try:
                    staged.rename(hf)
                except BaseException:
                    if previous.exists():
                        previous.rename(hf)
                    raise
    current = training_files(actor)
    if signatures != {p.name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in current}:
        raise RuntimeError(f"Training state changed during export; preserving shards: {checkpoint}")
    manifest = checkpoint / "storage_cleanup.json"
    record = dict(status="validated", checkpoint_step=int(checkpoint.name.split("_")[-1]),
                  evaluation=evidence, hf_model="actor/huggingface", **validation,
                  export_source=str(source) if source is not None else "FSDP shards",
                  added_weight_bytes=(sum(validation["weights"].values()) if not has_weights and source is None else 0),
                  removed_files={p.name: p.stat().st_size for p in files},
                  resumable_optimizer=False, timestamp=time.time())
    write_json(manifest, record)
    # Export and validation have succeeded. Only the enumerated, unchanged files may go.
    for path in files:
        path.unlink()
    record.update(status="complete", removed_bytes=sum(record["removed_files"].values()))
    write_json(manifest, record)
    print(f"CLEANED {checkpoint}: {record['removed_bytes'] / 1024**3:.2f} GiB training shards", flush=True)
    return record["removed_bytes"]


def cleanup_run(run, apply=False, steps=None):
    run = Path(run).resolve()
    record = read_json(run / "launch.json") if (run / "launch.json").exists() else {}
    if record.get("phase") in {"training", "export"} and record.get("status") == "running" and live_pid(record["pid"]):
        raise RuntimeError(f"Training/export is still active: {run}")
    if not (run / "checkpoint/latest_checkpointed_iteration.txt").is_file():
        print(f"SKIP no saved checkpoints: {run}", flush=True)
        return 0
    entries = eligible_checkpoints(run)
    if steps is not None:
        entries = [e for e in entries if int(e[0].name.split("_")[-1]) in steps]
    if not entries:
        print(f"SKIP no completed evaluation eligible for cleanup: {run}", flush=True)
        return 0
    if not apply:
        total = sum(p.stat().st_size for e in entries for p in training_files(e[0] / "actor"))
        print(json.dumps(dict(run=str(run), steps=[int(e[0].name.split("_")[-1]) for e in entries],
                              training_shard_bytes=total)), flush=True)
        return total
    with (run / "checkpoint/.storage_cleanup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return sum(clean_checkpoint(run, *entry) for entry in entries)


def cleanup_after_evaluation(run, python=None, *, steps=None, log=None):
    """Use the selected ML environment; the launcher itself can stay lightweight."""
    command = [python or sys.executable, str(Path(__file__).resolve()), str(run), "--apply"]
    if steps is not None:
        command += ["--steps", *map(str, steps)]
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true", help="Default is a read-only inventory")
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--wait-pid", nargs="+", type=int, help="Wait for these exact processes before cleanup")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(10)
    if args.wait_pid:
        import psutil
        pending = []
        for pid in args.wait_pid:
            try:
                pending.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        print(f"Waiting for processes {[p.pid for p in pending]}; no active training files will be touched", flush=True)
        while any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in pending):
            time.sleep(30)
    total = sum(cleanup_run(run, args.apply, args.steps) for run in args.runs)
    print(f"{'Removed' if args.apply else 'Eligible'} training shards: {total / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
