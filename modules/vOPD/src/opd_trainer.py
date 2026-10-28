from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from trl.experimental.minillm import MiniLLMTrainer
from trl.trainer.utils import selective_log_softmax

from modules.vOPD.src.grad_variance import GradientVarianceTracker
from modules.vOPD.src.progressive_jsd import dense_jsd_loss, entropy_alpha_totals, linear_alpha


class CustomMiniLLMTrainer(MiniLLMTrainer):
    def __init__(
        self,
        *args,
        use_baseline: bool = False,
        kl_top_k: int = -1,
        opd_top_k: int = -1,
        reward_clip_lambda: float | None = None,
        log_grad_variance: bool = False,
        grad_variance_logging_steps: int = 1,
        dense_objective: str = "upstream",
        alpha_start: float = 0.0,
        alpha_end: float = 1.0,
        alpha_schedule: str = "linear",
        loss_token_chunk_size: int = 128,
        prompt_length_limit: int = 1024,
        stop_weight: float = 0.0,
        stop_token_ids=(),
        **kwargs,
    ):
        if reward_clip_lambda is not None and not 0.0 < reward_clip_lambda < 1.0:
            raise ValueError("reward_clip_lambda must be between 0 and 1")
        super().__init__(*args, **kwargs)
        if dense_objective not in {"upstream", "progressive_jsd", "reverse_kl"}:
            raise ValueError(f"Unknown dense objective: {dense_objective}")
        linear_alpha(0, 2, alpha_start, alpha_end)
        self.dense_objective = dense_objective
        self.alpha_start, self.alpha_end = alpha_start, alpha_end
        if alpha_schedule not in {"linear", "teacher_entropy", "teacher_entropy_mean"}:
            raise ValueError(f"Unknown alpha schedule: {alpha_schedule}")
        if dense_objective == "reverse_kl" and alpha_schedule != "linear":
            raise ValueError("Reverse KL requires the unmodified endpoint schedule")
        self.alpha_schedule = alpha_schedule
        if alpha_schedule == "teacher_entropy_mean":
            if dense_objective != "progressive_jsd":
                raise ValueError("Aggregated entropy requires progressive_jsd")
            if self.args.steps_per_generation != self.args.gradient_accumulation_steps or self.num_iterations != 1:
                raise ValueError("Aggregated entropy requires one generation batch per optimizer update")
        self.loss_token_chunk_size = loss_token_chunk_size
        self.prompt_length_limit = prompt_length_limit
        self.stop_weight = float(stop_weight)
        self.stop_token_ids = tuple(stop_token_ids)
        if not math.isfinite(self.stop_weight) or self.stop_weight < 0:
            raise ValueError("stop_weight must be finite and nonnegative")
        if self.stop_weight and (dense_objective == "upstream" or not self.stop_token_ids):
            raise ValueError("Stop auxiliary requires a dense objective and explicit stop IDs")
        if dense_objective != "upstream":
            # GRPO disables Trainer's accumulation scaling for its own loss;
            # restore it for a standard differentiable mean divergence.
            self.compute_loss_func = None
        self.use_baseline = use_baseline
        self.kl_top_k = kl_top_k
        self.opd_top_k = opd_top_k
        self.use_top_k_opd = opd_top_k > 0
        self.reward_clip_boundary = (
            math.log(reward_clip_lambda) / (1.0 - reward_clip_lambda)
            if reward_clip_lambda is not None
            else None
        )
        self.grad_variance_tracker = (
            GradientVarianceTracker(every_n_steps=grad_variance_logging_steps)
            if log_grad_variance
            else None
        )

    @staticmethod
    def _topk_log_softmax(logits, indices):
        selected_logits = torch.gather(logits, dim=-1, index=indices)
        return selected_logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    def _compute_reward_clipped_advantage(
        self,
        student_log_probs_on_labels,
        teacher_log_probs_on_labels,
        mask,
    ):
        if self.reward_clip_boundary is None:
            return self._compute_advantage(
                student_log_probs_on_labels=student_log_probs_on_labels,
                teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                mask=mask,
            )
        student_log_probs_on_labels = student_log_probs_on_labels.detach()
        rewards = (
            (teacher_log_probs_on_labels - student_log_probs_on_labels)
            .detach()
            .clamp_min(self.reward_clip_boundary)
        )
        return self._compute_advantage(
            student_log_probs_on_labels=student_log_probs_on_labels,
            teacher_log_probs_on_labels=student_log_probs_on_labels + rewards,
            mask=mask,
        )

    def log(self, logs, start_time=None):
        if self.grad_variance_tracker is not None and self.model.training:
            logs.update(
                self.grad_variance_tracker.metrics(
                    self.optimizer,
                    accelerator=self.accelerator,
                    step=self.state.global_step,
                )
            )
        super().log(logs, start_time)

    def _topk_opd_single_step_loss(self, student_logits, teacher_logits, mask):
        top_k = min(self.opd_top_k, student_logits.size(-1))
        topk_ids = torch.topk(
            student_logits.detach(), top_k, dim=-1
        ).indices  # [B, T, K]
        student_topk_logp = self._topk_log_softmax(
            student_logits, topk_ids
        )  # [B, T, K]

        with torch.no_grad():
            teacher_topk_logp = self._topk_log_softmax(
                teacher_logits, topk_ids
            )  # [B, T, K]

        student_topk_logp = student_topk_logp - torch.logsumexp(
            student_topk_logp, dim=-1, keepdim=True
        )
        teacher_topk_logp = teacher_topk_logp - torch.logsumexp(
            teacher_topk_logp, dim=-1, keepdim=True
        )

        student_topk_prob = student_topk_logp.exp()
        token_loss = (
            student_topk_prob * (student_topk_logp - teacher_topk_logp.detach())
        ).sum(dim=-1)  # [B, T]

        loss = (token_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        return loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.dense_objective != "upstream":
            return self.compute_dense_loss(model, inputs, return_outputs)
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat(
            [inputs["prompt_mask"], inputs["completion_mask"]], dim=1
        )
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        student_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )

        prompt_lengths = inputs["prompt_ids"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = input_ids[:, prompt_lengths:]

        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        needs_full_log_probs = self.single_step_decomposition or (
            self.use_baseline and self.rkl_advantage and self.kl_top_k <= 0
        )
        if needs_full_log_probs:
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            student_log_probs_on_labels = torch.gather(
                student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
            ).squeeze(-1)
            teacher_log_probs_on_labels = torch.gather(
                teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
            ).squeeze(-1)
        else:
            student_log_probs = None
            teacher_log_probs = None
            student_log_probs_on_labels = selective_log_softmax(
                student_logits, shifted_labels
            )
            teacher_log_probs_on_labels = selective_log_softmax(
                teacher_logits, shifted_labels
            )

        mask = inputs["completion_mask"].bool()

        if self.use_baseline:
            if self.rkl_advantage:
                if self.kl_top_k > 0:
                    top_k = min(self.kl_top_k, student_logits.size(-1))
                    kl_indices = torch.topk(student_logits, top_k, dim=-1).indices

                    if student_log_probs is None:
                        student_log_probs_for_baseline = self._topk_log_softmax(
                            student_logits, kl_indices
                        )
                        teacher_log_probs_for_baseline = self._topk_log_softmax(
                            teacher_logits, kl_indices
                        )
                    else:
                        student_log_probs_for_baseline = torch.gather(
                            student_log_probs, dim=-1, index=kl_indices
                        )
                        teacher_log_probs_for_baseline = torch.gather(
                            teacher_log_probs, dim=-1, index=kl_indices
                        )

                    student_log_mass = torch.logsumexp(
                        student_log_probs_for_baseline, dim=-1, keepdim=True
                    )
                    teacher_log_mass = torch.logsumexp(
                        teacher_log_probs_for_baseline, dim=-1, keepdim=True
                    )
                    student_log_probs_for_baseline = (
                        student_log_probs_for_baseline - student_log_mass
                    )
                    teacher_log_probs_for_baseline = (
                        teacher_log_probs_for_baseline - teacher_log_mass
                    )
                    student_probs = student_log_probs_for_baseline.exp()
                else:
                    student_log_probs_for_baseline = student_log_probs
                    teacher_log_probs_for_baseline = teacher_log_probs
                    student_probs = student_log_probs_for_baseline.exp()

                baseline = (
                    student_probs
                    * (student_log_probs_for_baseline - teacher_log_probs_for_baseline)
                ).sum(dim=-1)
                centered_gap = (
                    student_log_probs_on_labels - teacher_log_probs_on_labels - baseline
                )
                reverse_kl_advantage = -centered_gap.detach()

                inputs["advantages"] = (
                    inputs["advantages"].unsqueeze(1) + reverse_kl_advantage
                )

            # Compute GRPO loss on verifiable reward
            loss = self._compute_loss(model, inputs)

            # Compute loss
            if self.single_step_decomposition:
                single_step_decomposition_loss = self._single_step_decomposition_loss(
                    student_log_probs=student_log_probs,
                    teacher_log_probs=teacher_log_probs,
                    mask=mask,
                )

                loss += single_step_decomposition_loss

        else:
            if self.rkl_advantage:
                reverse_kl_advantage = self._compute_reward_clipped_advantage(
                    student_log_probs_on_labels=student_log_probs_on_labels,
                    teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                    mask=mask,
                )

                inputs["advantages"] = (
                    inputs["advantages"].unsqueeze(1) + reverse_kl_advantage
                )

            # Compute GRPO loss on verifiable reward
            loss = self._compute_loss(model, inputs)

            if self.single_step_decomposition:
                if self.use_top_k_opd:
                    single_step_decomposition_loss = self._topk_opd_single_step_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        mask=mask,
                    )
                else:
                    single_step_decomposition_loss = (
                        self._single_step_decomposition_loss(
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_log_probs,
                            mask=mask,
                        )
                    )

                loss += single_step_decomposition_loss

        torch.cuda.empty_cache()
        return (loss, student_outputs) if return_outputs else loss

    def _tokenize_prompts(self, prompts):
        token_ids, images, multimodal_fields = super()._tokenize_prompts(prompts)
        return (
            [ids[-self.prompt_length_limit :] for ids in token_ids],
            images,
            multimodal_fields,
        )

    def _distill_progress(self):
        return (1.0 if self.dense_objective == "reverse_kl" else linear_alpha(
            self.state.global_step, self.state.max_steps, self.alpha_start, self.alpha_end,
        ))

    @torch.no_grad()
    def _aggregate_entropy_alpha(self, inputs, progress):
        """One token-weighted alpha for the full generation/optimizer batch.

        Precompute before GRPO shuffles/splits the generation batch. Teacher
        forwards stay microbatched; only two FP64 totals cross GPU boundaries.
        No student forward, sampling or RNG consumption is added here.
        """
        if progress in (0, 1):
            return progress
        totals = inputs["completion_ids"].new_zeros(2, dtype=torch.float64)
        batch_size = self.args.per_device_train_batch_size
        self.teacher_model.eval()
        for offset in range(0, inputs["prompt_ids"].shape[0], batch_size):
            sl = slice(offset, offset + batch_size)
            ids = torch.cat((inputs["prompt_ids"][sl], inputs["completion_ids"][sl]), dim=1)
            mask = torch.cat((inputs["prompt_mask"][sl], inputs["completion_mask"][sl]), dim=1)
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 0)
            output = self.teacher_model(
                input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False,
            )
            start = inputs["prompt_ids"].shape[1] - 1
            totals += entropy_alpha_totals(
                output.logits[:, start:-1, :] / self.kd_temperature,
                inputs["completion_mask"][sl], progress, self.loss_token_chunk_size,
            )
            del output
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        if totals[1].item() <= 0:
            raise ValueError("Aggregated entropy batch has no valid response tokens")
        return (totals[0] / totals[1]).item()

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        if self.alpha_schedule == "teacher_entropy_mean":
            alpha = self._aggregate_entropy_alpha(result, self._distill_progress())
            # A per-sequence field survives GRPO's shuffle and microbatch split;
            # every sequence/rank/accumulation slice carries the SAME scalar.
            result["distill_shared_alpha"] = torch.full(
                (result["prompt_ids"].shape[0],), alpha,
                device=result["prompt_ids"].device, dtype=torch.float64,
            )
        return result

    def compute_dense_loss(self, model, inputs, return_outputs=False):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat(
            [inputs["prompt_mask"], inputs["completion_mask"]], dim=1
        )
        prompt_length = inputs["prompt_ids"].shape[1]
        # Causal positions must match the left-padded generation prefixes.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        student = model(**kwargs)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher = self.teacher_model(**kwargs)
        q = student.logits[:, prompt_length - 1 : -1, :]
        p = teacher.logits[:, prompt_length - 1 : -1, :]
        progress = self._distill_progress()
        alpha, schedule = progress, self.alpha_schedule
        if schedule == "teacher_entropy_mean":
            alpha = inputs["distill_shared_alpha"][0].item()
            schedule = "linear"  # Apply the precomputed scalar without a second entropy transform.
        loss, alpha_stats = dense_jsd_loss(
            q / self.kd_temperature,
            p / self.kd_temperature,
            inputs["completion_mask"],
            alpha,
            self.loss_token_chunk_size,
            alpha_schedule=schedule,
            return_alpha_stats=True,
            stop_weight=self.stop_weight,
            stop_token_ids=self.stop_token_ids,
        )
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["distill/alpha"].append(alpha_stats["mean"].item())
        self._metrics[mode]["distill/alpha_progress"].append(progress)
        if self.alpha_schedule in {"teacher_entropy", "teacher_entropy_mean"}:
            self._metrics[mode]["distill/alpha_floor_fraction"].append(
                alpha_stats["floor_fraction"].item()
            )
        self._metrics[mode]["distill/loss"].append(loss.detach().item())
        if self.stop_weight:
            for key in ("primary_loss", "stop_loss", "teacher_stop_probability", "student_stop_probability"):
                self._metrics[mode]["distill/" + key].append(alpha_stats[key].item())
            self._metrics[mode]["distill/stop_weight"].append(self.stop_weight)
        self._metrics[mode]["distill/completion_tokens"].append(
            inputs["completion_mask"].sum().item()
        )
        self._metrics[mode]["distill/peak_memory_gib"].append(
            torch.cuda.max_memory_allocated() / 2**30
        )
        return (loss, student) if return_outputs else loss
