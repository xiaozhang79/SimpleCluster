import json
import os
import types
from typing import Optional, Union

import torch

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava_onevision import Llava_OneVision
from simplecluster.core import compress_video_tokens


def _append_audit(path: str | None, payload: dict) -> None:
    if not path:
        return
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


@register_model("llava_onevision_simplecluster")
class LlavaOneVisionSimpleCluster(Llava_OneVision):
    """Compress all pooled video tokens in one shared feature space."""

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
        original_pool = self.model.get_2dPool
        call_index = 0

        def pooled_with_simplecluster(model_self, image_feature, stride=2):
            nonlocal call_index
            pooled = original_pool(image_feature, stride)
            if pooled.ndim != 3 or pooled.shape[1] != 196:
                raise RuntimeError(
                    "Expected pooled video features [frames, 196, hidden], "
                    f"got {tuple(pooled.shape)}"
                )
            frame_indices = torch.arange(pooled.shape[0], device=pooled.device)
            compressed, _, _ = compress_video_tokens(
                pooled,
                frame_indices,
                ratio,
                max_iters=10,
            )
            original_count = int(pooled.shape[0] * pooled.shape[1])
            final_count = int(compressed.shape[1])
            _append_audit(
                token_audit_path,
                {
                    "call_index": call_index,
                    "method": "SimpleCluster",
                    "frames": int(pooled.shape[0]),
                    "grid_size": 14,
                    "original_content_tokens": original_count,
                    "final_content_tokens": final_count,
                    "actual_retention_ratio": final_count / original_count,
                },
            )
            call_index += 1
            return compressed

        self.model.get_2dPool = types.MethodType(
            pooled_with_simplecluster, self.model
        )
