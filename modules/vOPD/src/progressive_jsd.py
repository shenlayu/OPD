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


@torch.no_grad()
def entropy_alpha_totals(teacher_logits, mask, progress, token_chunk_size=128):
    """Return sum(alpha) and valid-token count, ready to sum across ranks.

    Transform each token's entropy BEFORE averaging; ignore padding and retain
    the same clipping as teacher_entropy. FP64 totals avoid chunk-size drift.
    The caller supplies temperature-scaled response-prediction logits.
    """
    if teacher_logits.shape[:-1] != mask.shape or token_chunk_size < 1:
        raise ValueError("Invalid entropy aggregation shape or chunk size")
    logits = teacher_logits.detach().reshape(-1, teacher_logits.shape[-1])
    weights = mask.reshape(-1).to(dtype=torch.float64)
    total = logits.new_zeros((), dtype=torch.float64)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    for offset in range(0, logits.shape[0], token_chunk_size):
        w = weights[offset : offset + token_chunk_size]
        valid = w > 0
        log_p = F.log_softmax(
            logits[offset : offset + token_chunk_size][valid], dim=-1, dtype=dtype,
        )
        alpha = entropy_alpha(log_p, progress)
        total += (alpha.double() * w[valid]).sum()
    return torch.stack((total, weights.sum()))


def token_divergence(
    student_logits, teacher_logits, alpha, alpha_schedule="linear", return_alpha=False,
    stop_weight=0.0, stop_token_ids=(), return_components=False,
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
    stop = torch.zeros_like(result)
    teacher_stop = torch.zeros_like(result)
    student_stop = torch.zeros_like(result)
    primary = result
    if stop_weight:
        stop, teacher_stop, student_stop = stop_bernoulli_kl(log_q, log_p, stop_token_ids)
        result = result + stop_weight * stop
    if return_components:
        return result, effective_alpha, torch.stack((primary, stop, teacher_stop, student_stop), -1)
    return (result, effective_alpha) if return_alpha else result


def stop_bernoulli_kl(log_q, log_p, stop_token_ids):
    """Teacher-to-student KL of stop/continue at every prefix, with stable tails.

    Both categories are logsumexp reductions of the full distribution. This
    avoids cancellation from log(1-p_stop) when stop probability rounds to 1.
    The teacher stays detached; no sampled EOS or length-cap labels are used.
    """
    ids = tuple(stop_token_ids)
    size = log_q.shape[-1]
    if (not ids or len(set(ids)) != len(ids) or len(ids) >= size
            or any(type(i) is not int or not 0 <= i < size for i in ids)):
        raise ValueError("Stop IDs must be distinct valid IDs with a nonempty complement")
    selector = torch.zeros(size, dtype=torch.bool, device=log_q.device)
    selector[list(ids)] = True
    log_p = log_p.detach()
    grouped_q = torch.stack((torch.logsumexp(log_q[..., list(ids)], -1),
                            torch.logsumexp(log_q.masked_fill(selector, -torch.inf), -1)), -1)
    grouped_p = torch.stack((torch.logsumexp(log_p[..., list(ids)], -1),
                            torch.logsumexp(log_p.masked_fill(selector, -torch.inf), -1)), -1)
    loss = (grouped_p.exp() * (grouped_p - grouped_q)).sum(-1)
    return loss, grouped_p[..., 0].exp(), grouped_q[..., 0].exp()


def dense_jsd_loss(
    student_logits,
    teacher_logits,
    mask,
    alpha,
    token_chunk_size=128,
    alpha_schedule="linear",
    return_alpha_stats=False,
    stop_weight=0.0,
    stop_token_ids=(),
):
    if (
        student_logits.shape != teacher_logits.shape
        or mask.shape != student_logits.shape[:-1]
    ):
        raise ValueError("student, teacher, and completion mask shapes must agree")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if not math.isfinite(stop_weight) or stop_weight < 0:
        raise ValueError("stop_weight must be finite and nonnegative")
    q = student_logits.reshape(-1, student_logits.shape[-1])
    p = teacher_logits.detach().reshape_as(q)
    weights = mask.reshape(-1).to(dtype=torch.float32)
    total = q.new_zeros((), dtype=torch.float32)
    alpha_sum = q.new_zeros((), dtype=torch.float32)
    floor_count = q.new_zeros((), dtype=torch.float32)
    components_sum = q.new_zeros((4,), dtype=torch.float32)

    def chunk_loss(s, t, w):
        divergence, actual_alpha, components = token_divergence(
            s, t, alpha, alpha_schedule, return_alpha=True,
            stop_weight=stop_weight, stop_token_ids=stop_token_ids, return_components=True,
        )
        return (
            (divergence * w).sum(),
            (actual_alpha * w).sum(),
            ((actual_alpha <= 1e-3) * w).sum(),
            (components.detach() * w.unsqueeze(-1)).sum(0),
        )

    for offset in range(0, q.shape[0], token_chunk_size):
        args = (
            q[offset : offset + token_chunk_size],
            p[offset : offset + token_chunk_size],
            weights[offset : offset + token_chunk_size],
        )
        if torch.is_grad_enabled() and q.requires_grad:
            part, part_alpha, part_floor, part_components = checkpoint(
                chunk_loss, *args, use_reentrant=False
            )
        else:
            part, part_alpha, part_floor, part_components = chunk_loss(*args)
        total = total + part
        alpha_sum = alpha_sum + part_alpha
        floor_count = floor_count + part_floor
        components_sum = components_sum + part_components
    denominator = weights.sum().clamp_min(1)
    loss = total / denominator
    if return_alpha_stats:
        return loss, dict(
            mean=alpha_sum / denominator, floor_fraction=floor_count / denominator,
            primary_loss=components_sum[0] / denominator,
            stop_loss=components_sum[1] / denominator,
            teacher_stop_probability=components_sum[2] / denominator,
            student_stop_probability=components_sum[3] / denominator,
        )
    return loss
