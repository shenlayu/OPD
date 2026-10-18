# Copyright 2026 OPD experiment contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixed-prefix full-vocabulary JSD updates in the native verl actor."""

import torch
from modules.rethinking.objective import dense_jsd_loss, late_beta, linear_alpha
from transformers import AutoModelForCausalLM

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch


def packed_inputs_and_mask(batch):
    """Match the native actor's next-token alignment, including the first response."""
    valid = batch["attention_mask"].bool()
    responses = batch["responses"].shape[-1]
    mask = torch.zeros_like(batch["attention_mask"])
    mask[:, -responses - 1 : -1] = batch["response_mask"]
    return {
        "input_ids": batch["input_ids"][valid].unsqueeze(0),
        "position_ids": batch["position_ids"][valid].unsqueeze(0),
        "attention_mask": None,
        "use_cache": False,
    }, mask[valid].unsqueeze(0)


def update_dense_jsd(actor, data):
    if actor.use_ulysses_sp or not actor.use_remove_padding:
        raise ValueError(
            "Dense distillation uses packed text inputs without sequence parallelism"
        )
    if (
        actor.config.ppo_epochs != 1
        or actor.config.use_kl_loss
        or actor.config.entropy_coeff
    ):
        raise ValueError(
            "Dense JSD requires one update with no extra KL or entropy objective"
        )
    actor.actor_module.train()
    temperature = data.meta_info["temperature"]
    alpha = linear_alpha(
        data.meta_info["global_steps"] - 1, data.meta_info["total_training_steps"]
    )
    if actor.config.dense_distill_mode not in {"rkl", "linear_jsd", "entropy_jsd"}:
        raise ValueError(f"Unknown dense objective: {actor.config.dense_distill_mode}")
    alpha_schedule = (
        "teacher_entropy"
        if actor.config.dense_distill_mode == "entropy_jsd"
        else "linear"
    )
    if actor.config.dense_distill_mode == "rkl":
        alpha = 1.0
    entropy_power = actor.config.dense_entropy_power
    beta = late_beta(alpha, actor.config.dense_beta_max)
    if alpha_schedule != "teacher_entropy" and (entropy_power != 1 or beta != 1):
        raise ValueError("Entropy extensions require the entropy_jsd actor mode")
    if not hasattr(actor, "dense_teacher"):
        actor.dense_teacher = (
            AutoModelForCausalLM.from_pretrained(
                actor.config.dense_teacher_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        apply_monkey_patch(
            actor.dense_teacher,
            use_remove_padding=True,
            ulysses_sp_size=1,
            use_fused_kernels=False,
        )
    device = get_device_id()
    actor.dense_teacher.to(device)
    mini_batches = data.select(
        batch_keys=[
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
    ).split(actor.config.ppo_mini_batch_size)
    if len(mini_batches) != 1:
        raise ValueError(
            "The schedule assumes exactly one optimizer update per outer batch"
        )
    metrics = {
        "actor/jsd_alpha": [alpha],
        "actor/jsd_entropy_power": [entropy_power],
        "actor/jsd_beta": [beta],
        "actor/jsd_loss_temperature": [1 / beta],
    }
    schedule_totals = torch.zeros(3, dtype=torch.float64, device=device)
    try:
        for mini_batch in mini_batches:
            micro_batches, _ = prepare_dynamic_batch(
                mini_batch, max_token_len=actor.config.ppo_max_token_len_per_gpu
            )
            actor.actor_optimizer.zero_grad()
            for micro in micro_batches:
                micro = micro.to(device)
                inputs, mask = packed_inputs_and_mask(micro.batch)
                with torch.autocast(
                    device_type=actor.device_name, dtype=torch.bfloat16
                ):
                    with torch.no_grad():
                        teacher_logits = (
                            actor.dense_teacher(**inputs).logits / temperature
                        )
                    student_logits = actor.actor_module(**inputs).logits / temperature
                    result = dense_jsd_loss(
                        student_logits,
                        teacher_logits,
                        mask,
                        alpha,
                        token_chunk_size=actor.config.dense_token_chunk_size,
                        alpha_schedule=alpha_schedule,
                        return_alpha_stats=alpha_schedule == "teacher_entropy",
                        entropy_power=entropy_power,
                        beta=beta,
                    )
                    if alpha_schedule == "teacher_entropy":
                        loss, alpha_stats = result
                        tokens = mask.sum().detach()
                        schedule_totals += torch.stack(
                            [
                                alpha_stats["mean"] * tokens,
                                alpha_stats["floor_fraction"] * tokens,
                                tokens,
                            ]
                        ).to(torch.float64)
                    else:
                        loss = result
                    del result
                # Preserve the native dynamic-microbatch aggregation convention.
                scale = micro.batch.batch_size[0] / actor.config.ppo_mini_batch_size
                (loss * scale).backward()
                append_to_dict(
                    metrics, {"actor/dense_distill_loss": loss.detach().item() * scale}
                )
                del teacher_logits, student_logits, loss
            grad_norm = actor._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        if alpha_schedule == "teacher_entropy":
            # Report the token-weighted mean over every microbatch and data-parallel rank.
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(schedule_totals)
            denominator = schedule_totals[2].clamp_min(1)
            append_to_dict(
                metrics,
                {
                    "actor/jsd_effective_alpha": (
                        schedule_totals[0] / denominator
                    ).item(),
                    "actor/jsd_alpha_floor_fraction": (
                        schedule_totals[1] / denominator
                    ).item(),
                },
            )
    finally:
        actor.actor_optimizer.zero_grad()
        actor.dense_teacher.to("cpu")
        torch.cuda.empty_cache()
    return metrics
