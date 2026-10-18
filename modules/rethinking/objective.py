"""Full-vocabulary normalized generalized JSD, optionally vocabulary-parallel."""

import math

import torch
from torch.utils.checkpoint import checkpoint


def linear_alpha(update, total_updates):
    if total_updates < 2:
        return 0.0
    return min(1.0, max(0.0, update / (total_updates - 1)))


def late_beta(progress, max_beta=1.25):
    """Keep the first half at unit temperature, then ramp the inverse temperature."""
    if not 0 <= progress <= 1 or not math.isfinite(max_beta) or max_beta < 1:
        raise ValueError("Invalid progress or maximum inverse temperature")
    return 1 + (max_beta - 1) * max(0.0, 2 * progress - 1)


def entropy_alpha(log_teacher_probs, progress, floor=1e-3, entropy_power=1.0):
    """Use untempered teacher entropy in bits; optionally amplify 1 + H."""
    if not 0 <= progress <= 1 or not 0 < floor < 0.5:
        raise ValueError("Invalid progress or numerical alpha floor")
    if not math.isfinite(entropy_power) or entropy_power < 1:
        raise ValueError("Entropy power must be finite and at least one")
    log_p = log_teacher_probs.detach()
    if progress in (0, 1):
        return log_p.new_full(log_p.shape[:-1], progress)
    entropy = -(log_p.exp() * log_p)
    entropy = entropy.sum(-1)
    entropy_bits = entropy.clamp_min(0) / math.log(2)
    exponent = 1 + entropy_bits
    if entropy_power != 1:
        exponent = exponent.pow(entropy_power)
    return torch.exp(exponent * math.log(progress)).clamp(floor, 1 - floor)


def token_divergence(
    student,
    teacher,
    alpha,
    alpha_schedule="linear",
    return_alpha=False,
    entropy_power=1.0,
    beta=1.0,
):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    if alpha_schedule not in {"linear", "teacher_entropy"}:
        raise ValueError(f"Unknown alpha schedule: {alpha_schedule}")
    if not math.isfinite(beta) or beta < 1:
        raise ValueError("Inverse temperature must be finite and at least one")
    dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
    student, teacher = student.to(dtype), teacher.detach().to(dtype)
    log_q = (student if beta == 1 else student * beta).log_softmax(-1)
    base_log_p = teacher.log_softmax(-1)
    log_p = base_log_p if beta == 1 else (teacher * beta).log_softmax(-1)

    def sum_vocab(x):
        return x.sum(-1)

    effective_alpha = (
        entropy_alpha(base_log_p, alpha, entropy_power=entropy_power)
        if alpha_schedule == "teacher_entropy"
        else log_p.new_full(log_p.shape[:-1], alpha)
    )
    if alpha == 0:
        result = sum_vocab(log_p.exp() * (log_p - log_q))
    elif alpha == 1:
        result = sum_vocab(log_q.exp() * (log_q - log_p))
    else:
        # q must remain differentiable inside the mixture; teacher and schedule are fixed.
        if alpha_schedule == "teacher_entropy":
            a = effective_alpha.unsqueeze(-1)
            log_m = torch.logaddexp(log_p + a.log(), log_q + torch.log1p(-a))
            denominator = effective_alpha
        else:
            log_m = torch.logaddexp(log_p + math.log(alpha), log_q + math.log1p(-alpha))
            denominator = alpha
        result = (
            sum_vocab(log_p.exp() * (log_p - log_m)) / (1 - denominator)
            + sum_vocab(log_q.exp() * (log_q - log_m)) / denominator
        )
    if beta != 1:
        result = result / beta**2
    return (result, effective_alpha) if return_alpha else result


def dense_jsd_loss(
    student,
    teacher,
    mask,
    alpha,
    token_chunk_size=64,
    alpha_schedule="linear",
    return_alpha_stats=False,
    entropy_power=1.0,
    beta=1.0,
):
    if student.shape != teacher.shape or mask.shape != student.shape[:-1]:
        raise ValueError("Logits and completion mask shapes disagree")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    q = student.reshape(-1, student.shape[-1])
    p = teacher.detach().reshape_as(q)
    weights = mask.reshape(-1).to(torch.float32)
    total = q.new_zeros((), dtype=torch.float32)
    alpha_sum = q.new_zeros((), dtype=torch.float32)
    floor_count = q.new_zeros((), dtype=torch.float32)

    def chunk(s, t, w):
        divergence, actual_alpha = token_divergence(
            s,
            t,
            alpha,
            alpha_schedule=alpha_schedule,
            return_alpha=True,
            entropy_power=entropy_power,
            beta=beta,
        )
        return (
            (divergence * w).sum(),
            (actual_alpha * w).sum(),
            ((actual_alpha <= 1e-3) * w).sum(),
        )

    for offset in range(0, q.shape[0], token_chunk_size):
        pieces = (
            q[offset : offset + token_chunk_size],
            p[offset : offset + token_chunk_size],
            weights[offset : offset + token_chunk_size],
        )
        if torch.is_grad_enabled() and q.requires_grad:
            part, part_alpha, part_floor = checkpoint(
                chunk, *pieces, use_reentrant=False
            )
        else:
            part, part_alpha, part_floor = chunk(*pieces)
        total = total + part
        alpha_sum = alpha_sum + part_alpha
        floor_count = floor_count + part_floor
    denominator = weights.sum().clamp_min(1)
    loss = total / denominator
    if return_alpha_stats:
        return loss, {
            "mean": alpha_sum / denominator,
            "floor_fraction": floor_count / denominator,
        }
    return loss
