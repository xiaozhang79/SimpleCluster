import math

import torch
import torch.nn.functional as F


def build_3d_grid_positions(
    frame_indices: torch.Tensor,
    tokens_per_frame: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build ``[frame, row, column]`` coordinates for square token grids."""

    if frame_indices.ndim != 1:
        raise ValueError("frame_indices must be one-dimensional")
    grid_size = math.isqrt(int(tokens_per_frame))
    if grid_size * grid_size != int(tokens_per_frame):
        raise ValueError(f"Expected a square token grid, got {tokens_per_frame} tokens")

    target_device = device if device is not None else frame_indices.device
    frame_indices = frame_indices.to(device=target_device, dtype=torch.float32)
    rows = torch.arange(grid_size, device=target_device, dtype=torch.float32)
    columns = torch.arange(grid_size, device=target_device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    spatial = torch.stack((grid_y, grid_x), dim=-1).reshape(1, tokens_per_frame, 2)
    temporal = frame_indices.reshape(-1, 1, 1).expand(-1, tokens_per_frame, 1)
    positions = torch.cat(
        (temporal, spatial.expand(frame_indices.numel(), -1, -1)), dim=-1
    )
    return positions.reshape(1, frame_indices.numel() * tokens_per_frame, 3)


def apply_3d_rope(
    features: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Apply 3D rotary encoding used only for group assignment."""

    if features.ndim != 3:
        raise ValueError(
            f"Expected features [batch, tokens, hidden], got {tuple(features.shape)}"
        )
    if positions.shape != (*features.shape[:2], 3):
        raise ValueError(
            "positions must have shape [batch, tokens, 3], "
            f"got {tuple(positions.shape)}"
        )
    if rope_theta <= 0:
        raise ValueError("rope_theta must be positive")

    hidden_size = features.shape[-1]
    base_axis_dim, remainder = divmod(hidden_size, 3)
    axis_dims = (base_axis_dim + remainder, base_axis_dim, base_axis_dim)
    axis_starts = (0, axis_dims[0], axis_dims[0] + axis_dims[1])
    output = features.clone()
    positions = positions.to(device=features.device, dtype=torch.float32)

    for axis, (start, axis_dim) in enumerate(zip(axis_starts, axis_dims)):
        rotary_dim = axis_dim - axis_dim % 2
        if rotary_dim == 0:
            continue
        token_slice = output[..., start : start + rotary_dim]
        half_dim = rotary_dim // 2
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (
                torch.arange(half_dim, device=features.device, dtype=torch.float32)
                / max(half_dim, 1)
            )
        )
        angles = positions[..., axis].unsqueeze(-1) * inv_freq
        cosine = angles.cos().to(dtype=features.dtype)
        sine = angles.sin().to(dtype=features.dtype)
        even = token_slice[..., 0::2]
        odd = token_slice[..., 1::2]
        rotated = torch.empty_like(token_slice)
        rotated[..., 0::2] = even * cosine - odd * sine
        rotated[..., 1::2] = even * sine + odd * cosine
        output[..., start : start + rotary_dim] = rotated
    return output


def _initialize_prototypes(
    tokens: torch.Tensor, group_count: int
) -> torch.Tensor:
    """Deterministically initialize dispersed prototypes."""

    batch_size, _, hidden_size = tokens.shape
    batch_indices = torch.arange(batch_size, device=tokens.device)
    indices = torch.empty(
        batch_size, group_count, dtype=torch.long, device=tokens.device
    )
    indices[:, 0] = tokens.square().sum(dim=-1).argmax(dim=-1)
    first = tokens[batch_indices, indices[:, 0]]
    nearest = torch.einsum("bnc,bc->bn", tokens, first)
    for group_index in range(1, group_count):
        next_indices = nearest.argmin(dim=-1)
        indices[:, group_index] = next_indices
        prototype = tokens[batch_indices, next_indices]
        nearest = torch.maximum(
            nearest, torch.einsum("bnc,bc->bn", tokens, prototype)
        )
    return torch.gather(
        tokens,
        1,
        indices.unsqueeze(-1).expand(batch_size, group_count, hidden_size),
    )


def _assign(
    tokens: torch.Tensor, prototypes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    similarities = torch.bmm(tokens, prototypes.transpose(1, 2))
    return similarities.argmax(dim=-1), similarities


def _update(
    tokens: torch.Tensor,
    assignments: torch.Tensor,
    similarities: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    batch_size = tokens.shape[0]
    weights = F.one_hot(assignments, num_classes=group_count).to(tokens.dtype)
    counts = weights.sum(dim=1)
    sums = torch.einsum("bnk,bnc->bkc", weights, tokens)
    updated = F.normalize(
        sums / counts.clamp_min(1.0).unsqueeze(-1), dim=-1, eps=1e-6
    )

    empty = counts == 0
    if empty.any():
        batch_indices = torch.arange(batch_size, device=tokens.device)
        nearest = similarities.max(dim=-1).values
        for group_index in range(group_count):
            empty_mask = empty[:, group_index]
            if not empty_mask.any():
                continue
            replacement = nearest.argmin(dim=-1)
            replacement_tokens = tokens[batch_indices, replacement]
            updated[:, group_index] = torch.where(
                empty_mask.unsqueeze(-1), replacement_tokens, updated[:, group_index]
            )
            nearest[empty_mask, replacement[empty_mask]] = 1.0
    return updated


@torch.no_grad()
def simplecluster_partition(
    features: torch.Tensor,
    group_count: int,
    *,
    max_iters: int = 10,
    grouping_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw-feature group means and cosine-based assignments."""

    if features.ndim != 3:
        raise ValueError(
            f"Expected [batch, tokens, hidden], got {tuple(features.shape)}"
        )
    if not 1 <= int(group_count) <= features.shape[1]:
        raise ValueError(
            f"group_count must be in [1, {features.shape[1]}], got {group_count}"
        )
    if int(max_iters) < 1:
        raise ValueError("max_iters must be positive")
    if grouping_features is not None and grouping_features.shape != features.shape:
        raise ValueError("grouping_features must match features")

    raw = features.float()
    grouping = raw if grouping_features is None else grouping_features.float()
    work = F.normalize(grouping, dim=-1, eps=1e-6)
    prototypes = _initialize_prototypes(work, int(group_count))

    for _ in range(int(max_iters)):
        assignments, similarities = _assign(work, prototypes)
        updated = _update(work, assignments, similarities, int(group_count))
        if torch.allclose(updated, prototypes, atol=1e-6, rtol=0.0):
            prototypes = updated
            break
        prototypes = updated

    assignments, _ = _assign(work, prototypes)
    weights = F.one_hot(assignments, num_classes=int(group_count)).to(raw.dtype)
    counts = weights.sum(dim=1)
    if not bool((counts > 0).all()):
        raise RuntimeError("SimpleCluster produced an empty final group")
    sums = torch.einsum("bnk,bnc->bkc", weights, raw)
    means = (sums / counts.unsqueeze(-1)).to(dtype=features.dtype)
    return means, assignments


def order_by_first_member(
    compressed: torch.Tensor, assignments: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Order compressed tokens by the earliest source member."""

    group_count = compressed.shape[1]
    token_count = assignments.shape[1]
    positions = torch.arange(token_count, device=assignments.device)
    first = torch.full(
        (assignments.shape[0], group_count),
        token_count,
        dtype=torch.long,
        device=assignments.device,
    )
    first.scatter_reduce_(
        1, assignments, positions.unsqueeze(0).expand_as(assignments),
        reduce="amin", include_self=True
    )
    order = torch.argsort(first, dim=1)
    ordered = torch.gather(
        compressed,
        1,
        order.unsqueeze(-1).expand(-1, -1, compressed.shape[-1]),
    )
    ordered_positions = torch.gather(first, 1, order)
    return ordered, ordered_positions


def compress_video_tokens(
    features: torch.Tensor,
    frame_indices: torch.Tensor,
    retention_ratio: float,
    *,
    max_iters: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress a video-wide square-grid token sequence."""

    if features.ndim != 3:
        raise ValueError("features must have shape [frames, tokens, hidden]")
    if not 0.0 < float(retention_ratio) <= 1.0:
        raise ValueError("retention_ratio must be in (0, 1]")
    frames, tokens_per_frame, hidden_size = features.shape
    if frame_indices.shape != (frames,):
        raise ValueError("frame_indices must contain one entry per feature grid")

    original_count = frames * tokens_per_frame
    group_count = max(
        1, min(original_count, round(original_count * float(retention_ratio)))
    )
    raw = features.reshape(1, original_count, hidden_size)
    positions = build_3d_grid_positions(
        frame_indices, tokens_per_frame, device=features.device
    )
    grouping = apply_3d_rope(raw, positions)
    compressed, assignments = simplecluster_partition(
        raw,
        group_count,
        max_iters=max_iters,
        grouping_features=grouping,
    )
    ordered, first_positions = order_by_first_member(compressed, assignments)
    return ordered, assignments, first_positions
