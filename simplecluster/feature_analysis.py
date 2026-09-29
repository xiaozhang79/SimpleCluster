import argparse
import csv
import json
from pathlib import Path

import torch
import torch.distributed as distributed

from simplecluster.core import compress_video_tokens
from simplecluster.feature_metrics import compute_feature_metrics
from simplecluster.result_logs import RATIOS, rebuild_feature_log


TASKS = {
    "MVBench": "mvbench",
    "EgoSchema": "egoschema",
    "LongVideoBench": "longvideobench_val_v",
    "VideoMME": "videomme",
}


def _tasks(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _tasks(nested)
    else:
        yield value


def video_paths(benchmark: str) -> list[str]:
    """Resolve each unique video once within a benchmark, without reading answers."""

    from lmms_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager(model_name="llava_onevision")
    loaded = get_task_dict([TASKS[benchmark]], task_manager=manager)
    seen = set()
    paths = []
    for task in _tasks(loaded):
        for document in task.task_docs:
            visual = task.doc_to_visual(document)
            if not isinstance(visual, list) or len(visual) != 1 or not isinstance(visual[0], str):
                raise RuntimeError(f"{benchmark} contains a non-video sample")
            path = str(Path(visual[0]).resolve())
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths:
        raise RuntimeError(f"No videos found for {benchmark}")
    return paths


def visual_features(adapter, video_path: str) -> torch.Tensor:
    """Return projected, pooled visual content tokens before LLM insertion."""

    frames = adapter.load_video([video_path], adapter.max_frames_num)
    pixels = adapter._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    pixels = pixels.half().to(adapter.device)
    encoded = adapter.model.encode_images(pixels)
    pooled = adapter.model.get_2dPool(encoded)
    if pooled.ndim != 3 or pooled.shape[0] != 32 or pooled.shape[1] != 196:
        raise RuntimeError(f"Expected 32 x 196 pooled content tokens, got {tuple(pooled.shape)}")
    return pooled


def summarize(records: list[dict], output_dir: Path) -> list[dict]:
    fields = [
        "benchmark", "videos", "nqe_l2", "cosine_coverage_at_0.90",
        "mean_retained_tokens", "mean_actual_retention",
    ]
    rows = []
    for benchmark in TASKS:
        subset = [record for record in records if record["benchmark"] == benchmark]
        if not subset:
            raise RuntimeError(f"No completed videos for {benchmark}")
        mean = lambda name: sum(record[name] for record in subset) / len(subset)
        rows.append({
            "benchmark": benchmark,
            "videos": len(subset),
            "nqe_l2": mean("nqe_l2"),
            "cosine_coverage_at_0.90": mean("cosine_coverage_at_0.90"),
            "mean_retained_tokens": mean("retained_tokens"),
            "mean_actual_retention": mean("actual_retention"),
        })
    rows.append({
        "benchmark": "Macro",
        "videos": sum(row["videos"] for row in rows),
        **{
            field: sum(row[field] for row in rows) / len(rows)
            for field in fields[2:]
        },
    })
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--ratio-percent", type=int, choices=RATIOS, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for the FP16/FlashAttention 2 visual path")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from lmms_eval.models.simple.llava_onevision import Llava_OneVision

    adapter = Llava_OneVision(
        pretrained=args.model_path,
        conv_template="qwen_1_5",
        model_name="llava_qwen",
        max_frames_num=32,
        attn_implementation="flash_attention_2",
        batch_size=1,
    )
    rank, world_size = adapter.rank, adapter.world_size
    expected = {}
    with (args.output_dir / f"rank_{rank:02d}.jsonl").open("w", encoding="utf-8") as stream:
        with torch.inference_mode():
            for benchmark in TASKS:
                paths = video_paths(benchmark)
                expected[benchmark] = set(paths)
                for path in paths[rank::world_size]:
                    raw = visual_features(adapter, path)
                    compressed, _, _ = compress_video_tokens(
                        raw,
                        torch.arange(raw.shape[0], device=raw.device),
                        args.ratio_percent / 100,
                        max_iters=10,
                    )
                    metrics = compute_feature_metrics(
                        raw.reshape(-1, raw.shape[-1]),
                        compressed.reshape(-1, compressed.shape[-1]),
                    )
                    record = {
                        "benchmark": benchmark,
                        "video_path": path,
                        "nominal_retention": args.ratio_percent / 100,
                        "actual_retention": metrics.retained_tokens / metrics.original_tokens,
                        **metrics.to_dict(),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
    if distributed.is_available() and distributed.is_initialized():
        distributed.barrier()
    if rank != 0:
        return

    records = []
    for worker in range(world_size):
        path = args.output_dir / f"rank_{worker:02d}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing worker records: {path}")
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    for benchmark, expected_paths in expected.items():
        actual_paths = [record["video_path"] for record in records if record["benchmark"] == benchmark]
        if len(actual_paths) != len(expected_paths) or set(actual_paths) != expected_paths:
            raise RuntimeError(f"Incomplete or duplicated {benchmark} videos")

    rows = summarize(records, args.output_dir)
    summary_path = args.run_root / "feature_analysis" / f"r{args.ratio_percent}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"ratio_percent": args.ratio_percent, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    rebuild_feature_log(args.run_root, args.log_path)


if __name__ == "__main__":
    main()
