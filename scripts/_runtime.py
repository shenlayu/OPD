"""Configuration, immutable source snapshots, and the single-experiment lifecycle."""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]


def local_path(value, root=ROOT):
    path = Path(value).expanduser()
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(module, filename, overrides=(), root=ROOT):
    config = OmegaConf.load(filename)
    if set(config) != {"run", "training", "evaluation"}:
        raise ValueError("Config requires exactly run, training, evaluation sections")
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
    t = config.training
    if module == "vOPD":
        for key in ["dataset_name", "model_name_or_path", "teacher_model_name_or_path"]:
            if t[key]:
                t[key] = local_path(t[key], root)
        if t.resume_from_checkpoint:
            t.resume_from_checkpoint = local_path(t.resume_from_checkpoint, root)
        if t.dense_objective not in {"upstream", "progressive_jsd", "reverse_kl"}:
            raise ValueError("Unknown dense_objective")
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
        if t.alpha_schedule not in {"linear", "teacher_entropy"}:
            raise ValueError("Unknown alpha_schedule")
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
        if a.dense_distill_mode not in {"off", "rkl", "linear_jsd", "entropy_jsd"}:
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
            r.calculate_log_probs = False
            a.dense_teacher_path = t.reward_model.model.path
        elif not t.reward_model.enable:
            raise ValueError("Native OPD requires reward_model.enable=true")
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


def validate_assets(config, module):
    t = config.training
    if module == "vOPD":
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


def allocate_run(root, module, dataset):
    parent = Path(root) / module / dataset
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
    (destination / "scripts" / other).unlink()


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
    parser.add_argument(
        "overrides", nargs="*", help="Optional YAML key=value overrides"
    )
    args = parser.parse_args()
    config = load_config(module, args.config, args.overrides)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        print(
            f"Output: {config.run.output_root}/{module}/{config.run.dataset}/<next-number>"
        )
        return 0
    validate_assets(config, module)
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
    if module == "vOPD":
        t.output_dir = str(run / "checkpoint")
        t.run_name = f"{module}-{config.run.dataset}-{run.name}"
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
        t.trainer.experiment_name = f"rethinking-{config.run.dataset}-{run.name}"
        t.custom_reward_function.path = str(
            run
            / "code/modules/rethinking/verl/utils/reward_score/ttrl_math/__init__.py"
        )
        command = [python, "-m", "verl.trainer.main_ppo", str(config_path)]
    env = dict(os.environ)
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, config.run.gpus)),
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
        try:
            run_command(command, env, code, log)
            final = run / "checkpoint/final"
            if module == "rethinking":
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
            print("Experiment completed", file=log, flush=True)
        except BaseException as error:
            print(f"Experiment failed: {error}", file=log, flush=True)
            raise
    return 0
