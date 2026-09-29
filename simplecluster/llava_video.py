import json
import math
import os
import types
from typing import Optional, Union

import torch

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava_vid import LlavaVid
from simplecluster.core import compress_video_tokens


def _append_audit(path: str | None, payload: dict) -> None:
    if not path:
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
        )
    finally:
        os.close(descriptor)


def _validate_input_profile(wrapper: LlavaVid) -> None:
    expected = {
        "max_frames_num": 64,
        "force_sample": True,
        "mm_spatial_pool_mode": "average",
        "mm_resampler_location": "before",
        "video_fps": 1,
        "add_time_instruction": False,
        "mm_newline_position": "grid",
    }
    actual = {
        "max_frames_num": wrapper.max_frames_num,
        "force_sample": wrapper.force_sample,
        "mm_spatial_pool_mode": wrapper.mm_spatial_pool_mode,
        "mm_resampler_location": wrapper.mm_resampler_location,
        "video_fps": wrapper.fps,
        "add_time_instruction": wrapper.add_time_instruction,
        "mm_newline_position": getattr(
            wrapper.model.config, "mm_newline_position", None
        ),
    }
    if actual != expected:
        raise RuntimeError(
            "SimpleCluster requires the fixed-64 native-grid profile; "
            f"got {actual}"
        )


def _install_grid_layout(model, producer) -> None:
    """Preserve every native grid-row newline around compressed content."""

    original_pool = model.get_2dPool
    cache: list[tuple[list[torch.Tensor], list[torch.Tensor], int]] = []

    def pooled(model_self, image_feature, stride=2):
        features = original_pool(image_feature, stride)
        if tuple(features.shape[:2]) != (64, 169):
            raise RuntimeError(
                "Expected pooled features [64, 169, hidden], "
                f"got {tuple(features.shape)}"
            )
        grid_size = math.isqrt(int(features.shape[1]))
        frame_tokens, frame_anchors = producer(features)
        cache.append((frame_tokens, frame_anchors, grid_size))
        return features

    def grid(model_self, _unused_features):
        if not cache:
            raise RuntimeError("Missing compressed grid layout")
        frame_tokens, frame_anchors, grid_size = cache.pop(0)
        newline = model_self.model.image_newline
        output = []
        for tokens, anchors in zip(frame_tokens, frame_anchors):
            if anchors.numel() > 1:
                order = torch.argsort(anchors, stable=True)
                tokens, anchors = tokens[order], anchors[order]
            for row in range(grid_size):
                in_row = anchors.div(grid_size, rounding_mode="floor") == row
                if in_row.any():
                    output.append(tokens[in_row])
                output.append(
                    newline.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
                )
        return torch.cat(output, dim=0)

    model.get_2dPool = types.MethodType(pooled, model)
    model.add_token_per_grid = types.MethodType(grid, model)


@register_model("llava_video_simplecluster")
class LlavaVideoSimpleCluster(LlavaVid):
    """Compress 64-frame content while retaining native grid newlines."""

    def __init__(
        self,
        retention_ratio: float = 0.10,
        simplecluster_max_iters: int = 10,
        token_audit_path: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        **kwargs,
    ) -> None:
        ratio = float(retention_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1]")
        if int(simplecluster_max_iters) != 10:
            raise ValueError("SimpleCluster uses 10 refinement iterations")

        super().__init__(batch_size=batch_size, **kwargs)
        _validate_input_profile(self)
        call_index = 0

        def producer(features: torch.Tensor):
            nonlocal call_index
            frames, tokens_per_frame, _ = features.shape
            frame_indices = torch.arange(frames, device=features.device)
            compressed, _, first_positions = compress_video_tokens(
                features,
                frame_indices,
                ratio,
                max_iters=10,
            )
            compressed = compressed[0]
            first_positions = first_positions[0]
            output_tokens = []
            output_anchors = []
            for frame in range(frames):
                in_frame = (
                    first_positions.div(tokens_per_frame, rounding_mode="floor")
                    == frame
                )
                output_tokens.append(compressed[in_frame])
                output_anchors.append(first_positions[in_frame] % tokens_per_frame)

            original_count = frames * tokens_per_frame
            final_count = int(compressed.shape[0])
            _append_audit(
                token_audit_path,
                {
                    "call_index": call_index,
                    "method": "SimpleCluster",
                    "frames": int(frames),
                    "grid_size": 13,
                    "grid_newlines": int(frames * 13),
                    "original_content_tokens": int(original_count),
                    "final_content_tokens": final_count,
                    "actual_retention_ratio": final_count / original_count,
                },
            )
            call_index += 1
            return output_tokens, output_anchors

        _install_grid_layout(self.model, producer)
