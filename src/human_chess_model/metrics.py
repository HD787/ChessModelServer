from __future__ import annotations

import torch
from torch.nn import functional as F


def mask_illegal_logits(
    logits: torch.Tensor,
    legal_indices: torch.Tensor,
    legal_counts: torch.Tensor,
) -> torch.Tensor:
    masked = torch.full_like(logits, float("-inf"))
    batch = torch.arange(logits.shape[0], device=logits.device)[:, None]
    positions = torch.arange(legal_indices.shape[1], device=logits.device)[None, :]
    valid = positions < legal_counts[:, None]
    safe_indices = legal_indices.clamp_min(0)
    masked[batch.expand_as(safe_indices)[valid], safe_indices[valid]] = logits[
        batch.expand_as(safe_indices)[valid], safe_indices[valid]
    ]
    return masked


@torch.no_grad()
def batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_indices: torch.Tensor | None = None,
    legal_counts: torch.Tensor | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    if legal_indices is not None and legal_counts is not None:
        logits = mask_illegal_logits(logits, legal_indices, legal_counts)
    loss = F.cross_entropy(logits, targets, reduction="sum")
    max_k = max(ks)
    top = torch.topk(logits, k=max_k, dim=1).indices
    metrics = {"loss_sum": loss.item(), "count": targets.numel()}
    for k in ks:
        metrics[f"top{k}"] = (top[:, :k] == targets[:, None]).any(dim=1).sum().item()
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).sum()
    metrics["entropy_sum"] = entropy.item()
    return metrics


def merge_metrics(items: list[dict]) -> dict:
    total = {}
    for item in items:
        for key, value in item.items():
            total[key] = total.get(key, 0.0) + value
    count = max(total.get("count", 1), 1)
    result = {
        "nll": total.get("loss_sum", 0.0) / count,
        "entropy": total.get("entropy_sum", 0.0) / count,
        "count": int(count),
    }
    for key, value in total.items():
        if key.startswith("top"):
            result[f"{key}_acc"] = value / count
    return result
