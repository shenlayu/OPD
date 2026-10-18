"""Dense-vocabulary normalized generalized JSD on fixed sampled prefixes.

Both models' full logits must already exist. Token chunking/checkpointing only
limits loss intermediates; there is no vocabulary streaming or sparse support.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def linear_alpha(
    step: int, total_steps: int, start: float = 0.0, end: float = 1.0
) -> float:
    """Zero-based optimizer updates: first uses start, last uses end."""
    if not 0 <= start <= end <= 1:
        raise ValueError("Require 0 <= alpha_start <= alpha_end <= 1")
    if total_steps < 2:
        return start
    fraction = min(max(step / (total_steps - 1), 0.0), 1.0)
    return start + fraction * (end - start)


def entropy_alpha(log_teacher_probs, progress, floor=1e-3):
    """Delay the transition by teacher uncertainty, measured in bits.

    A binary-uniform teacher gives a quadratic schedule; a certain teacher
    recovers linear progress. Endpoints are exact. The interior floor avoids
    cancellation in FP32 normalized JSD, and is not a tuned hyperparameter.
    """
    if not 0 <= progress <= 1 or not 0 < floor < 0.5:
        raise ValueError("Invalid progress or numerical alpha floor")
    log_p = log_teacher_probs.detach()
    if progress in (0, 1):
        return log_p.new_full(log_p.shape[:-1], progress)
    entropy_bits = (-(log_p.exp() * log_p).sum(-1)).clamp_min(0) / math.log(2)
    return torch.exp((1 + entropy_bits) * math.log(progress)).clamp(floor, 1 - floor)


def token_divergence(
    student_logits, teacher_logits, alpha, alpha_schedule="linear", return_alpha=False
):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if alpha_schedule not in {"linear", "teacher_entropy"}:
        raise ValueError(f"Unknown alpha schedule: {alpha_schedule}")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_q = F.log_softmax(student_logits, dim=-1, dtype=dtype)
    log_p = F.log_softmax(teacher_logits.detach(), dim=-1, dtype=dtype)
    effective_alpha = (
        entropy_alpha(log_p, alpha)
        if alpha_schedule == "teacher_entropy"
        else log_p.new_full(log_p.shape[:-1], alpha)
    )
    if alpha == 0:
        result = (log_p.exp() * (log_p - log_q)).sum(-1)
    elif alpha == 1:
        result = (log_q.exp() * (log_q - log_p)).sum(-1)
    else:
        # Keep q's contribution to m differentiable. Teacher and schedule are fixed.
        if alpha_schedule == "linear":
            log_m = torch.logaddexp(log_p + math.log(alpha), log_q + math.log1p(-alpha))
            denominator = alpha
        else:
            a = effective_alpha.unsqueeze(-1)
            log_m = torch.logaddexp(log_p + a.log(), log_q + torch.log1p(-a))
            denominator = effective_alpha
        kl_p_m = (log_p.exp() * (log_p - log_m)).sum(-1)
        kl_q_m = (log_q.exp() * (log_q - log_m)).sum(-1)
        result = kl_p_m / (1 - denominator) + kl_q_m / denominator
    return (result, effective_alpha) if return_alpha else result


def dense_jsd_loss(
    student_logits,
    teacher_logits,
    mask,
    alpha,
    token_chunk_size=128,
    alpha_schedule="linear",
    return_alpha_stats=False,
):
    if (
        student_logits.shape != teacher_logits.shape
        or mask.shape != student_logits.shape[:-1]
    ):
        raise ValueError("student, teacher, and completion mask shapes must agree")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    q = student_logits.reshape(-1, student_logits.shape[-1])
    p = teacher_logits.detach().reshape_as(q)
    weights = mask.reshape(-1).to(dtype=torch.float32)
    total = q.new_zeros((), dtype=torch.float32)
    alpha_sum = q.new_zeros((), dtype=torch.float32)
    floor_count = q.new_zeros((), dtype=torch.float32)

    def chunk_loss(s, t, w):
        divergence, actual_alpha = token_divergence(
            s, t, alpha, alpha_schedule, return_alpha=True
        )
        return (
            (divergence * w).sum(),
            (actual_alpha * w).sum(),
            ((actual_alpha <= 1e-3) * w).sum(),
        )

    for offset in range(0, q.shape[0], token_chunk_size):
        args = (
            q[offset : offset + token_chunk_size],
            p[offset : offset + token_chunk_size],
            weights[offset : offset + token_chunk_size],
        )
        if torch.is_grad_enabled() and q.requires_grad:
            part, part_alpha, part_floor = checkpoint(
                chunk_loss, *args, use_reentrant=False
            )
        else:
            part, part_alpha, part_floor = chunk_loss(*args)
        total = total + part
        alpha_sum = alpha_sum + part_alpha
        floor_count = floor_count + part_floor
    denominator = weights.sum().clamp_min(1)
    loss = total / denominator
    if return_alpha_stats:
        return loss, dict(
            mean=alpha_sum / denominator, floor_fraction=floor_count / denominator
        )
    return loss
