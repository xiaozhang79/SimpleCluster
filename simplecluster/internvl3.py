import json
import os
import types
from typing import Optional, Union

import torch

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import InternVL3
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


@register_model("internvl3_simplecluster")
class InternVL3SimpleCluster(InternVL3):
    """Compress all frame-tile tokens before InternVL3 language generation."""

    def __init__(
        self,
        pretrained: str,
        modality: str = "video",
        batch_size: Union[str, int] = 1,
        retention_ratio: float = 0.10,
        simplecluster_max_iters: int = 10,
        token_audit_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        ratio = float(retention_ratio)
        if modality != "video":
            raise ValueError("SimpleCluster supports InternVL3 video evaluation only")
        if int(batch_size) != 1:
            raise ValueError("SimpleCluster requires batch_size=1")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1]")
        if int(simplecluster_max_iters) != 10:
            raise ValueError("SimpleCluster uses 10 refinement iterations")

        super().__init__(
            pretrained=pretrained,
            modality=modality,
            batch_size=batch_size,
            **kwargs,
        )
        model = self.model
        original_chat = model.chat
        get_conv_template = original_chat.__globals__.get("get_conv_template")
        if get_conv_template is None:
            raise RuntimeError("The loaded InternVL model exposes an unsupported chat API")
        call_index = 0

        def chat_with_simplecluster(
            model_self,
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=False,
            num_patches_list=None,
            IMG_START_TOKEN="<img>",
            IMG_END_TOKEN="</img>",
            IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
            verbose=False,
        ):
            nonlocal call_index
            if pixel_values is None:
                return original_chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=history,
                    return_history=return_history,
                    num_patches_list=num_patches_list,
                    IMG_START_TOKEN=IMG_START_TOKEN,
                    IMG_END_TOKEN=IMG_END_TOKEN,
                    IMG_CONTEXT_TOKEN=IMG_CONTEXT_TOKEN,
                    verbose=verbose,
                )
            if num_patches_list is None:
                raise ValueError("num_patches_list is required for video inputs")
            patch_counts = [int(value) for value in num_patches_list]
            if not patch_counts or any(value <= 0 for value in patch_counts):
                raise ValueError(f"Invalid num_patches_list: {patch_counts}")
            if int(pixel_values.shape[0]) != sum(patch_counts):
                raise ValueError("pixel_values and num_patches_list disagree")

            image_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            model_self.img_context_token_id = image_context_id
            template = get_conv_template(model_self.template)
            template.system_message = model_self.system_message
            eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

            current_history = [] if history is None else list(history)
            for old_question, old_answer in current_history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            for patch_count in patch_counts:
                image_tokens = (
                    IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * model_self.num_image_token * patch_count
                    + IMG_END_TOKEN
                )
                query = query.replace("<image>", image_tokens, 1)

            dense_features = model_self.extract_feature(pixel_values)
            if dense_features.ndim != 3:
                raise RuntimeError(
                    "Expected InternVL visual features [tiles, tokens, hidden], "
                    f"got {tuple(dense_features.shape)}"
                )
            frame_indices = torch.repeat_interleave(
                torch.arange(len(patch_counts), device=dense_features.device),
                torch.tensor(patch_counts, device=dense_features.device),
            )
            compressed, _, first_positions = compress_video_tokens(
                dense_features,
                frame_indices,
                ratio,
                max_iters=10,
            )
            compressed = compressed[0]
            first_positions = first_positions[0]

            model_inputs = tokenizer(query, return_tensors="pt")
            device = next(model_self.language_model.parameters()).device
            dense_ids = model_inputs["input_ids"].to(device)
            dense_context_positions = torch.nonzero(
                dense_ids[0] == image_context_id, as_tuple=False
            ).squeeze(1)
            expected_dense_count = int(
                dense_features.shape[0] * dense_features.shape[1]
            )
            if dense_context_positions.numel() != expected_dense_count:
                raise RuntimeError(
                    "Prompt/image token mismatch: "
                    f"prompt={dense_context_positions.numel()} visual={expected_dense_count}"
                )

            selected_sequence_positions = dense_context_positions[first_positions]
            keep = dense_ids[0] != image_context_id
            keep[selected_sequence_positions] = True
            compact_ids = dense_ids[:, keep]
            compact_embeddings = model_self.language_model.get_input_embeddings()(
                compact_ids
            )
            compact_context = compact_ids[0] == image_context_id
            if int(compact_context.sum()) != int(compressed.shape[0]):
                raise RuntimeError("Compressed prompt and visual features disagree")
            compact_embeddings[0, compact_context] = compressed.to(
                device=compact_embeddings.device,
                dtype=compact_embeddings.dtype,
            )
            attention_mask = torch.ones_like(compact_ids)
            generation_kwargs = dict(generation_config)
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_output = model_self.language_model.generate(
                inputs_embeds=compact_embeddings,
                attention_mask=attention_mask,
                use_cache=True,
                **generation_kwargs,
            )
            response = tokenizer.batch_decode(
                generation_output, skip_special_tokens=True
            )[0]
            response = response.split(template.sep.strip())[0].strip()
            current_history.append((question, response))

            final_count = int(compressed.shape[0])
            _append_audit(
                token_audit_path,
                {
                    "call_index": call_index,
                    "method": "SimpleCluster",
                    "frames": len(patch_counts),
                    "frame_tile_counts": patch_counts,
                    "grid_size": int(dense_features.shape[1] ** 0.5),
                    "original_content_tokens": expected_dense_count,
                    "final_content_tokens": final_count,
                    "actual_retention_ratio": final_count / expected_dense_count,
                },
            )
            call_index += 1
            if return_history:
                return response, current_history
            return response

        model.chat = types.MethodType(chat_with_simplecluster, model)
