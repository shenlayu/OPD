"""Configuration, immutable source snapshots, and the single-experiment lifecycle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from omegaconf import OmegaConf

from scripts.cleanup_checkpoints import cleanup_after_evaluation

ROOT = Path(__file__).resolve().parents[1]


def local_path(value, root=ROOT):
    path = Path(value).expanduser()
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(module, filename, overrides=(), root=ROOT):
    config = OmegaConf.load(filename)
    if set(config) != {"run", "training", "evaluation"}:
        raise ValueError("Config requires exactly run, training, evaluation sections")
    if module == "rethinking":
        # Apply to every existing experiment YAML without editing frozen runs.
        config.training.trainer.setdefault("rollout_data_dir", None)
        config.training.trainer.setdefault("rollout_samples_per_step", 20)
    if module == "vOPD":
        from modules.vOPD.src.token_progress_config import ProgressConfig
        from modules.vOPD.src.token_response import ResponseConfig

        config.training.setdefault("stop_weight", 0.0)
        config.training.setdefault("stop_token_ids", [])
        config.training.setdefault("generation_stop_token_ids", [])
        for key, value in dict(token_gate_tau_q=0.5, token_gate_stop=False,
                               token_gate_stop_ids=[], token_gate_k_odds=1.0,
                               token_gate_clip=None, token_gate_ce_weight=None,
                               token_gate_beta=None, token_gate_normalize=True).items():
            config.training.setdefault(key, value)
        config.training.token_gate_progress = OmegaConf.merge(
            ProgressConfig().to_dict(), config.training.get("token_gate_progress", {})
        )
        config.training.token_gate_response = OmegaConf.merge(
            ResponseConfig().to_dict(), config.training.get("token_gate_response", {})
        )
    config.evaluation.setdefault("max_parallel_workers", None)
    OmegaConf.set_struct(config, True)
    # YAML 1.1 otherwise parses the natural CLI spelling dense_distill_mode=off
    # as a boolean, although this field is an objective name.
    overrides = [
        item.rsplit("=", 1)[0] + "='off'"
        if item == "training.actor_rollout_ref.actor.dense_distill_mode=off"
        else item
        for item in overrides
    ]
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.run.dataset):
        raise ValueError("run.dataset must be a simple directory name")
    gpu_ids = [str(x) for x in config.run.gpus]
    if (
        not gpu_ids
        or len(set(gpu_ids)) != len(gpu_ids)
        or not all(x.isdigit() for x in gpu_ids)
    ):
        raise ValueError("run.gpus must contain distinct nonnegative GPU indices")
    if config.run.python:
        # Resolving a venv's interpreter symlink would silently select its base
        # environment and lose this module's pinned dependencies.
        interpreter = Path(config.run.python).expanduser()
        config.run.python = str(
            (root / interpreter).absolute()
            if not interpreter.is_absolute()
            else interpreter
        )
        if not Path(config.run.python).is_file():
            raise FileNotFoundError(config.run.python)
    config.run.output_root = local_path(config.run.output_root, root)
    for task in config.evaluation.datasets:
        task.path = local_path(task.path, root)
        if task.samples < 1:
            raise ValueError("Evaluation samples must be positive")
    e = config.evaluation
    if e.enabled and not e.datasets:
        raise ValueError("Evaluation requires at least one dataset")
    names = [task.name for task in e.datasets]
    if len(set(names)) != len(names):
        raise ValueError("Evaluation benchmark names must be unique")
    if any(task.limit is not None and task.limit < 1 for task in e.datasets):
        raise ValueError("Evaluation limits must be positive or null")
    if e.seed_mode not in {"engine", "per_sample"}:
        raise ValueError("Evaluation seed_mode must be engine or per_sample")
    if e.stop_mode not in {"model_eos", "registered_special"}:
        raise ValueError("Unknown evaluation stop_mode")
    if (
        e.seed_example_stride < 1
        or e.seed_task_stride < 1
        or (
            e.seed_mode == "per_sample"
            and any(t.samples > e.seed_example_stride for t in e.datasets)
        )
    ):
        raise ValueError("Evaluation seed strides must prevent overlapping samples")
    if (
        e.max_new_tokens < 1
        or e.batch_size < 1
        or e.max_prompt_length + e.max_new_tokens > e.max_model_len
    ):
        raise ValueError("Evaluation prompt + response budget must fit max_model_len")
    if e.tensor_parallel_size < 1 or len(gpu_ids) % e.tensor_parallel_size:
        raise ValueError(
            "Evaluation tensor_parallel_size must divide the number of GPUs"
        )
    evaluation_dp = e.get("data_parallel_size", 1)
    if type(evaluation_dp) is not int or evaluation_dp < 1:
        raise ValueError("Evaluation data_parallel_size must be a positive integer")
    workers = e.max_parallel_workers if e.max_parallel_workers is not None else evaluation_dp
    if type(workers) is not int or not 1 <= workers <= evaluation_dp:
        raise ValueError("max_parallel_workers must be between 1 and data_parallel_size")
    if workers * e.tensor_parallel_size > len(gpu_ids):
        raise ValueError("Evaluation data and tensor parallelism exceed run.gpus")
    allowed_parallel_seed = e.seed_mode == "per_sample" or (module == "vOPD" and e.seed_mode == "engine")
    if evaluation_dp > 1 and (e.tensor_parallel_size != 1 or not allowed_parallel_seed):
        raise ValueError("Parallel evaluation requires TP=1 and per-sample seeds, or vOPD engine question shards")
    t = config.training
    if module == "vOPD":
        progress_config = ProgressConfig.parse(OmegaConf.to_container(t.token_gate_progress, resolve=True))
        response_config = ResponseConfig.parse(OmegaConf.to_container(t.token_gate_response, resolve=True))
        response_config.validate_context(t.dense_objective, t.token_gate_stop, progress_config.enabled,
                                         t.token_gate_ce_weight, list(t.token_gate_stop_ids))
        if t.token_gate_ce_weight is not None:
            if not math.isfinite(t.token_gate_ce_weight) or t.token_gate_ce_weight < 0:
                raise ValueError("token_gate_ce_weight must be finite and nonnegative, or null")
            if t.dense_objective != "token_jsd" or t.token_gate_stop or progress_config.enabled:
                raise ValueError("CE replacement requires token_jsd without split-stop or Scheme B")
        if progress_config.enabled and t.dense_objective not in {"token_mix", "token_jsd"}:
            raise ValueError("Scheme B requires token_mix or token_jsd")
        for key in ["dataset_name", "model_name_or_path", "teacher_model_name_or_path"]:
            if t[key]:
                t[key] = local_path(t[key], root)
        if t.resume_from_checkpoint:
            t.resume_from_checkpoint = local_path(t.resume_from_checkpoint, root)
        if t.dense_objective not in {"upstream", "progressive_jsd", "reverse_kl", "token_mix", "token_jsd"}:
            raise ValueError("Unknown dense_objective")
        if t.dense_objective in {"token_mix", "token_jsd"}:
            if not 0 <= t.token_gate_tau_q <= 1 or type(t.token_gate_stop) is not bool:
                raise ValueError("Invalid token gate threshold or stop switch")
            ids = list(t.token_gate_stop_ids)
            if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("Invalid token_gate_stop_ids")
            if t.token_gate_k_odds != 1 or t.stop_weight or t.alpha_schedule != "linear":
                raise ValueError("Token gate requires k_odds=1, stop_weight=0 and no time schedule")
            if t.token_gate_clip is not None and (math.isnan(t.token_gate_clip) or t.token_gate_clip <= 0):
                raise ValueError("Invalid token_gate_clip")
            from modules.vOPD.src.token_gate import validate_ablation
            validate_ablation(t.dense_objective, t.token_gate_beta, t.token_gate_normalize, t.token_gate_clip,
                              t.token_gate_stop, progress_config.enabled, t.token_gate_ce_weight,
                              response_config.enabled)
            if t.resume_from_checkpoint and t.token_gate_clip is None and not progress_config.enabled:
                raise ValueError("Resume must supply token_gate_clip from diagnostics/token_gate_calibration.json")
            if t.resume_from_checkpoint and progress_config.enabled:
                if not (Path(t.resume_from_checkpoint) / "token_progress_state.json").is_file():
                    raise ValueError("Scheme B resume requires token_progress_state.json in the checkpoint")
        elif t.token_gate_beta is not None or t.token_gate_normalize is not True:
            raise ValueError("token_gate_beta / token_gate_normalize require token_mix or token_jsd")
        if not math.isfinite(t.stop_weight) or t.stop_weight < 0:
            raise ValueError("stop_weight must be finite and nonnegative")
        for key in ("stop_token_ids", "generation_stop_token_ids"):
            ids = list(t[key])
            if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError(f"Invalid {key}")
        if t.stop_weight and (t.dense_objective == "upstream" or not t.stop_token_ids):
            raise ValueError("Stop auxiliary requires dense objective and stop_token_ids")
        if t.generation_stop_token_ids and (
            not t.use_vllm or e.stop_mode != "registered_special"
            or (t.stop_weight and set(t.stop_token_ids) != set(t.generation_stop_token_ids))
        ):
            raise ValueError("Training, evaluation and auxiliary stop protocols must agree")
        if t.distill_mode not in {"tml", "gkd_rkl"}:
            raise ValueError("distill_mode must be tml or gkd_rkl")
        if t.dense_objective != "upstream" and (
            t.distill_mode != "gkd_rkl" or t.use_baseline or t.opd_top_k > 0
        ):
            raise ValueError(
                "Dense objectives require gkd_rkl, use_baseline=false, opd_top_k<=0"
            )
        if not 0 <= t.alpha_start <= t.alpha_end <= 1:
            raise ValueError("Require 0 <= alpha_start <= alpha_end <= 1")
        if t.alpha_schedule not in {"linear", "teacher_entropy", "teacher_entropy_mean"}:
            raise ValueError("Unknown alpha_schedule")
        if t.alpha_schedule == "teacher_entropy_mean" and t.dense_objective != "progressive_jsd":
            raise ValueError("Aggregated entropy requires progressive_jsd")
        if t.dense_objective == "reverse_kl" and t.alpha_schedule != "linear":
            raise ValueError("reverse_kl requires alpha_schedule=linear")
        if t.report_to != "none":
            raise ValueError("OPD_lite logs only to main.log; report_to must be none")
        if t.max_steps == 0 or t.max_steps < -1 or t.num_train_epochs <= 0:
            raise ValueError("Invalid training budget")
        if (
            min(
                t.gradient_accumulation_steps,
                t.per_device_train_batch_size,
                t.loss_token_chunk_size,
            )
            < 1
        ):
            raise ValueError(
                "Batch, accumulation and token chunk sizes must be positive"
            )
        if t.use_vllm and (
            t.vllm_mode != "colocate" or len(gpu_ids) % t.vllm_tensor_parallel_size
        ):
            raise ValueError(
                "Use colocated vLLM; its TP size must divide the GPU count"
            )
        lr = t.learning_rate
    else:
        if type(t.trainer.rollout_samples_per_step) is not int or t.trainer.rollout_samples_per_step < 0:
            raise ValueError("rollout_samples_per_step must be a nonnegative integer")
        a, r, m = (
            t.actor_rollout_ref.actor,
            t.actor_rollout_ref.rollout,
            t.actor_rollout_ref.model,
        )
        if a.strategy not in {"fsdp", "fsdp2"} or r.name != "vllm" or r.mode != "sync":
            raise ValueError("rethinking supports FSDP/FSDP2 with synchronous vLLM")
        if t.algorithm.adv_estimator != "token_reward_direct":
            raise ValueError("Only the OPD token_reward_direct chain is included")
        if t.trainer.nnodes != 1 or t.trainer.n_gpus_per_node != len(gpu_ids):
            raise ValueError("Set trainer.nnodes=1 and n_gpus_per_node=len(run.gpus)")
        if (
            list(t.trainer.logger) != ["console"]
            or t.trainer.test_freq != -1
            or t.trainer.val_before_train
        ):
            raise ValueError(
                "Use console logging and standalone evaluation (test_freq=-1, val_before_train=false)"
            )
        if t.global_profiler.steps or t.reward_model.launch_reward_fn_async:
            raise ValueError(
                "Profiling and asynchronous reward jobs are excluded from the minimal chain"
            )
        m.path = local_path(m.path, root)
        t.reward_model.model.path = local_path(t.reward_model.model.path, root)
        if t.reward_model.model.input_tokenizer:
            t.reward_model.model.input_tokenizer = local_path(
                t.reward_model.model.input_tokenizer, root
            )
        t.data.train_files = [local_path(p, root) for p in t.data.train_files]
        if t.trainer.resume_from_path:
            t.trainer.resume_from_path = local_path(t.trainer.resume_from_path, root)
        if t.trainer.resume_mode not in {"disable", "resume_path"}:
            raise ValueError(
                "Use resume_mode=disable or resume_path with an explicit checkpoint"
            )
        if t.trainer.resume_mode == "resume_path" and not t.trainer.resume_from_path:
            raise ValueError("resume_path requires trainer.resume_from_path")
        if t.trainer.total_epochs < 1 or (
            t.trainer.total_training_steps is not None
            and t.trainer.total_training_steps < 1
        ):
            raise ValueError("Invalid training budget")
        if (
            r.tensor_model_parallel_size < 1
            or len(gpu_ids) % r.tensor_model_parallel_size
        ):
            raise ValueError("Rollout tensor_parallel_size must divide the GPU count")
        if t.data.train_batch_size % a.ppo_mini_batch_size:
            raise ValueError(
                "train_batch_size must be divisible by ppo_mini_batch_size"
            )
        if a.dense_distill_mode not in {"off", "rkl", "linear_jsd", "entropy_jsd", "research"}:
            raise ValueError("Unknown dense_distill_mode")
        if not all(
            math.isfinite(x) and x >= 1
            for x in [a.dense_entropy_power, a.dense_beta_max]
        ):
            raise ValueError(
                "dense_entropy_power and dense_beta_max must be finite and >=1"
            )
        if a.dense_token_chunk_size < 1:
            raise ValueError("dense_token_chunk_size must be positive")
        schedule_updates = a.get("dense_schedule_updates")
        if schedule_updates is not None and (isinstance(schedule_updates, bool)
                or not isinstance(schedule_updates, int) or schedule_updates < 2):
            raise ValueError("dense_schedule_updates must be an integer >=2")
        from modules.rethinking.progressive.config import validate_seq_weight
        seq = validate_seq_weight(a.get("dense_seq_weight", {}))
        seq_enabled = seq["mode"] != "off"
        if seq_enabled:
            if a.dense_distill_mode not in {"linear_jsd", "entropy_jsd"}:
                raise ValueError("Dense trajectory weighting requires linear_jsd or entropy_jsd")
            if r.temperature != 1 or r.top_p != 1 or r.top_k != -1 or r.repetition_penalty != 1:
                raise ValueError("Trajectory weights require untruncated temperature-1 behavior logprobs")
            seq["group_size"] = int(r.n)
            a.dense_seq_weight = seq
        if a.dense_distill_mode != "off":
            if (
                not m.use_remove_padding
                or a.ulysses_sequence_parallel_size != 1
                or not a.use_dynamic_bsz
            ):
                raise ValueError(
                    "Dense OPD requires packed dynamic microbatches and sequence parallel size 1"
                )
            if a.ppo_epochs != 1 or a.ppo_mini_batch_size != t.data.train_batch_size:
                raise ValueError(
                    "Dense OPD requires exactly one optimizer update per rollout batch"
                )
            if a.entropy_coeff or a.use_kl_loss or t.algorithm.use_kl_in_reward:
                raise ValueError(
                    "Dense OPD cannot add entropy or reference-policy KL losses"
                )
            if a.dense_distill_mode != "entropy_jsd" and (
                a.dense_entropy_power != 1 or a.dense_beta_max != 1
            ):
                raise ValueError("Entropy extensions require entropy_jsd")
            # Derived toggles prevent incompatible native reward/PG paths.
            t.reward_model.enable = False
            r.log_prob_top_k = 0
            r.calculate_log_probs = seq_enabled
            a.dense_teacher_path = t.reward_model.model.path
        elif not t.reward_model.enable:
            raise ValueError("Native OPD requires reward_model.enable=true")
        if a.dense_distill_mode == "research":
            from modules.rethinking.research_objective import validate_research
            validate_research(OmegaConf.to_container(a.dense_research, resolve=True))
            if r.temperature != 1 or r.top_p != 1 or r.top_k != -1:
                raise ValueError("Research rollouts require temperature=1, top_p=1, top_k=-1")
            rollout = t.get("research_rollout", {})
            if rollout.get("mode", "online") not in {"online", "offline", "mixture", "schedule"}:
                raise ValueError("Unknown research rollout source")
            if rollout.get("mode", "online") != "online":
                if not rollout.get("cache"):
                    raise ValueError("Offline/mixed rollouts require a frozen token cache")
                t.research_rollout.cache = local_path(rollout.cache, root)
        progressive = t.get("progressive", {})
        if progressive.get("enabled", False):
            from modules.rethinking.progressive import validate_progressive_config

            normalized_progressive = validate_progressive_config(progressive)
            if normalized_progressive["seq_weight"]["mode"] != "off":
                normalized_progressive["seq_weight"]["group_size"] = int(r.n)
                t.progressive = normalized_progressive
            if a.dense_distill_mode != "off" or r.log_prob_top_k != 0:
                raise ValueError("Progressive alpha requires the sampled-token OPD path")
            if not r.calculate_log_probs:
                raise ValueError("Progressive alpha requires rollout behavior logprobs")
            if a.ppo_epochs != 1 or a.ppo_mini_batch_size != t.data.train_batch_size:
                raise ValueError("Alpha schedules require one optimizer update per rollout batch")
            if a.entropy_coeff or a.use_kl_loss or t.algorithm.use_kl_in_reward:
                raise ValueError("Alpha experiment cannot add entropy or reference KL losses")
            if t.algorithm.norm_adv_by_std_in_grpo:
                raise ValueError("Alpha experiment requires unnormalized advantages")
            if r.temperature != 1 or r.top_p != 1 or r.top_k != -1:
                raise ValueError("Alpha experiment requires untruncated, temperature-1 rollout")
        if r.top_k_strategy not in {
            "only_stu",
            "only_tch",
            "intersection",
            "union",
            "union-intersection",
        }:
            raise ValueError("Unknown top_k_strategy")
        if r.reward_weight_mode not in {"student_p", "teacher_p", "none"}:
            raise ValueError("Unknown reward_weight_mode")
        lr = a.optim.lr
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return config


def validate_assets(config, module, *, evaluation_only=False):
    t = config.training
    if evaluation_only:
        if module != "vOPD":
            raise ValueError("Student-only asset validation is supported for vOPD")
        models, data, resume = [t.model_name_or_path], [], None
    elif module == "vOPD":
        models = [
            t.model_name_or_path,
            t.teacher_model_name_or_path or t.model_name_or_path,
        ]
        data = [t.dataset_name]
        resume = t.resume_from_checkpoint
    else:
        models = [t.actor_rollout_ref.model.path, t.reward_model.model.path]
        data = list(t.data.train_files)
        resume = t.trainer.resume_from_path
    for name in models:
        folder = Path(name)
        if (
            not (folder / "config.json").is_file()
            or not (folder / "tokenizer_config.json").is_file()
        ):
            raise FileNotFoundError(
                f"Missing model/tokenizer: {folder}. See model/README.md"
            )
        index = folder / "model.safetensors.index.json"
        if index.exists():
            import json

            files = set(json.loads(index.read_text())["weight_map"].values())
        else:
            files = (
                {"model.safetensors"}
                if (folder / "model.safetensors").exists()
                else {"pytorch_model.bin"}
            )
        if not files or not all(
            (folder / p).is_file() and (folder / p).stat().st_size for p in files
        ):
            raise FileNotFoundError(f"Incomplete model weights: {folder}")
    if config.evaluation.enabled:
        data += [task.path for task in config.evaluation.datasets]
    for name in data + ([resume] if resume else []):
        if not Path(name).exists():
            raise FileNotFoundError(f"Missing input: {name}. See data/README.md")


RUN_TIMEZONE = ZoneInfo("Asia/Shanghai")


def run_date(now=None):
    """Use the user's timezone rather than the host timezone for daily IDs."""
    return (now or datetime.now(timezone.utc)).astimezone(RUN_TIMEZONE).date().isoformat()


def is_dated_run(run, parent):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run.parent.name):
        return False
    try:
        datetime.strptime(run.parent.name, "%Y-%m-%d")
    except ValueError:
        return False
    return (run.parent.parent == parent and run.name.isdigit()
            and int(run.name) > 0 and run.is_dir())


def allocate_run(root, module, dataset):
    parent = Path(root) / module / dataset / run_date()
    parent.mkdir(parents=True, exist_ok=True)
    number = (
        max((int(p.name) for p in parent.iterdir() if p.name.isdigit()), default=0) + 1
    )
    while True:
        path = parent / f"{number:02d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def snapshot_code(root, destination, module):
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", ".ruff_cache", ".pytest_cache"
    )
    shutil.copytree(
        root / "modules" / module, destination / "modules" / module, ignore=ignore
    )
    shutil.copy2(root / "modules/__init__.py", destination / "modules/__init__.py")
    # The math evaluator is shared; no other training module is required.
    shutil.copytree(root / "scripts", destination / "scripts", ignore=ignore)
    other = "run_rethinking.py" if module == "vOPD" else "run_vopd.py"
    (destination / "scripts" / other).unlink(missing_ok=True)


def write_launch_record(run, record, **updates):
    """Keep launch provenance and lifecycle status inside the numbered run."""
    record.update(updates, updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run, prefix=".launch-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, Path(run) / "launch.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_command(command, env, cwd, log):
    """Mirror one child process to main.log and clean up only its descendants."""
    import psutil

    print("Running:", " ".join(map(str, command)), file=log, flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    owned = {}

    def relay():
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    try:
        while process.poll() is None:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    owned[child.pid] = child.create_time()
            except psutil.Error:
                pass
            time.sleep(0.2)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        children = []
        for pid, created in owned.items():
            try:
                child = psutil.Process(pid)
                if child.create_time() == created:
                    child.terminate()
                    children.append(child)
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        reader.join(timeout=5)
        process.stdout.close()


def launch(module, default_config):
    def interrupted(signum, frame):
        # Let run_command's finally block clean up its separately grouped workers.
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(
        description=f"One {module} experiment: train, save, evaluate"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / default_config
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print config; no writes or GPU work",
    )
    parser.add_argument("--reserved-run", type=Path, help="Numbered folder reserved by the study supervisor")
    parser.add_argument(
        "overrides", nargs="*", help="Optional YAML key=value overrides"
    )
    args = parser.parse_args()
    config = load_config(module, args.config, args.overrides)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(
            f"Output: {config.run.output_root}/{module}/{config.run.dataset}/{run_date()}/<next-number>"
        )
        return 0
    validate_assets(config, module)
    if args.reserved_run:
        run = args.reserved_run.resolve()
        expected = Path(config.run.output_root).resolve() / module / config.run.dataset
        if not is_dated_run(run, expected):
            raise ValueError("Reserved run must be an existing date/number folder")
        if any((run / name).exists() for name in ("launch.json", "checkpoint", "code")):
            raise ValueError("Reserved run has already started; refusing to overwrite")
    else:
        run = allocate_run(config.run.output_root, module, config.run.dataset)
    print(f"Experiment: {run}", flush=True)
    (run / "configs").mkdir()
    (run / "checkpoint").mkdir()
    snapshot_code(ROOT, run / "code", module)
    shutil.copy2(args.config, run / "configs/requested.yaml")
    python = config.run.python or sys.executable
    config.run.python = python
    config_path = run / "configs" / default_config
    t = config.training
    if module == "rethinking" and t.actor_rollout_ref.actor.dense_distill_mode == "research":
        t.actor_rollout_ref.actor.dense_research.output_dir = str(run / "diagnostics")
        if "research_rollout" in t:
            t.research_rollout.probe_output = str(run / "diagnostics/probe_rollouts.jsonl")
    if module == "vOPD":
        t.output_dir = str(run / "checkpoint")
        t.run_name = f"{module}-{config.run.dataset}-{run.parent.name}-{run.name}"
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(config.run.gpus)}",
            "--module",
            "modules.vOPD.src.opd_train",
            str(config_path),
        ]
    else:
        t.trainer.default_local_dir = str(run / "checkpoint")
        t.trainer.rollout_data_dir = str(run / "rollouts")
        t.trainer.experiment_name = f"rethinking-{config.run.dataset}-{run.parent.name}-{run.name}"
        t.custom_reward_function.path = str(
            run
            / "code/modules/rethinking/verl/utils/reward_score/ttrl_math/__init__.py"
        )
        command = [python, "-m", "verl.trainer.main_ppo", str(config_path)]
    env = dict(os.environ)
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    if "GPU_SCHEDULE_GPU_IDS" in env:
        scheduled = [int(i) for i in env["GPU_SCHEDULE_GPU_IDS"].split(",")]
        if scheduled != list(config.run.gpus) or not env.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("GPU scheduler allocation must match run.gpus in order")
        visible_devices = env["CUDA_VISIBLE_DEVICES"]
    else:
        visible_devices = ",".join(map(str, config.run.gpus))
    env.update(
        CUDA_VISIBLE_DEVICES=visible_devices,
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS=str(config.run.threads),
        MKL_NUM_THREADS=str(config.run.threads),
        WANDB_MODE="disabled",
        TRL_EXPERIMENTAL_SILENCE="1",
    )
    code = run / "code"
    env["PYTHONPATH"] = os.pathsep.join([str(code), str(code / "modules/rethinking")])
    for key in [
        "RAY_ADDRESS",
        "CUDA_LAUNCH_BLOCKING",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
    ]:
        env.pop(key, None)
    with (
        (run / "main.log").open("w", buffering=1) as log,
        tempfile.TemporaryDirectory(prefix="opd-") as temporary,
    ):
        if module == "rethinking":
            t.ray_kwargs.ray_init._temp_dir = temporary
        OmegaConf.save(config, config_path, resolve=True)
        record = {
            "schema_version": 1,
            "module": module,
            "pid": os.getpid(),
            "python": python,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_date": run.parent.name,
            "run_number": run.name,
            "run_timezone": str(RUN_TIMEZONE),
            "requested_config": "configs/requested.yaml",
            "effective_config": str(config_path.relative_to(run)),
            "overrides": list(args.overrides),
        }
        context = os.environ.get("OPD_LAUNCH_CONTEXT")
        if context:
            record["queue"] = json.loads(context)
        write_launch_record(run, record, status="running", phase="training")
        try:
            run_command(command, env, code, log)
            final = run / "checkpoint/final"
            if module == "rethinking":
                write_launch_record(run, record, phase="export")
                step = int(
                    (run / "checkpoint/latest_checkpointed_iteration.txt").read_text()
                )
                checkpoint = run / f"checkpoint/global_step_{step}/actor"
                run_command(
                    [
                        python,
                        "-m",
                        "verl.model_merger",
                        "merge",
                        "--backend",
                        "fsdp",
                        "--local_dir",
                        str(checkpoint),
                        "--target_dir",
                        str(final),
                        "--use_cpu_initialization",
                    ],
                    env,
                    code,
                    log,
                )
                # Preserve the model's exact template and stopping protocol after export.
                for name in [
                    "tokenizer_config.json",
                    "generation_config.json",
                    "chat_template.jinja",
                ]:
                    source = Path(t.actor_rollout_ref.model.path) / name
                    if source.exists():
                        shutil.copy2(source, final / name)
                if config.run.teacher_chat_template:
                    # Run in the selected environment (the lightweight launcher need not have Transformers).
                    run_command(
                        [
                            python,
                            "-c",
                            "from modules.rethinking.model_view import apply_teacher_metadata; "
                            "import sys; apply_teacher_metadata(*sys.argv[1:])",
                            t.actor_rollout_ref.model.path,
                            t.reward_model.model.path,
                            str(final),
                        ],
                        env,
                        code,
                        log,
                    )
            if config.evaluation.enabled:
                write_launch_record(run, record, phase="evaluation")
                run_command(
                    [
                        python,
                        "-m",
                        "scripts._evaluate",
                        str(config_path),
                        module,
                        str(final),
                        str(run),
                    ],
                    env,
                    code,
                    log,
                )
                if module == "rethinking":
                    write_launch_record(run, record, phase="storage_cleanup")
                    cleanup_after_evaluation(run, python, log=log)
                if module == "vOPD" and config.evaluation.get("complete_box_scoring", False):
                    run_command(
                        [python, "-m", "scripts.diagnose_complete_boxes", str(run)],
                        env, code, log,
                    )
            print("Experiment completed", file=log, flush=True)
            write_launch_record(run, record, status="completed", phase="completed", finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        except BaseException as error:
            print(f"Experiment failed: {error}", file=log, flush=True)
            write_launch_record(run, record, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}", finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            raise
    return 0
