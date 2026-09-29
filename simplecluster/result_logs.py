import argparse
import json
import math
from pathlib import Path


BACKBONES = ("llava_onevision", "llava_video", "internvl3")
BENCHMARKS = ("MVBench", "EgoSchema", "LongVideoBench", "VideoMME")
RATIOS = (1, 5, 10, 15, 20, 25)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _complete_samples(payload: dict, task: str) -> int:
    counts = payload.get("n-samples", {}).get(task)
    if not isinstance(counts, dict):
        raise ValueError(f"Missing sample count for {task}")
    original = int(counts["original"])
    effective = int(counts["effective"])
    if original <= 0 or effective != original:
        raise ValueError(f"Incomplete task {task}: {effective}/{original} samples")
    return original


def performance_scores(payload: dict) -> dict[str, float]:
    """Read the four primary metrics from a complete lmms-eval result JSON."""

    results = payload["results"]
    mv_tasks = sorted(
        name for name in results
        if name.startswith("mvbench_") and "mvbench_accuracy,none" in results[name]
    )
    if len(mv_tasks) != 20:
        raise ValueError(f"Expected 20 MVBench subtasks, found {len(mv_tasks)}")
    weighted = 0.0
    count = 0
    for task in mv_tasks:
        n = _complete_samples(payload, task)
        weighted += n * _number(results[task]["mvbench_accuracy,none"], task)
        count += n

    _complete_samples(payload, "egoschema")
    _complete_samples(payload, "longvideobench_val_v")
    _complete_samples(payload, "videomme")
    longvideo = _number(results["longvideobench_val_v"]["lvb_acc,none"], "LongVideoBench")
    if not 0.0 <= longvideo <= 1.0:
        raise ValueError("LongVideoBench accuracy must be in [0, 1]")
    scores = {
        "MVBench": weighted / count,
        "EgoSchema": _number(results["egoschema"]["submission,none"], "EgoSchema"),
        "LongVideoBench": 100.0 * longvideo,
        "VideoMME": _number(results["videomme"]["videomme_perception_score,none"], "VideoMME"),
    }
    if any(not 0.0 <= value <= 100.0 for value in scores.values()):
        raise ValueError(f"A benchmark score is outside [0, 100]: {scores}")
    scores["Avg"] = sum(scores[name] for name in BENCHMARKS) / len(BENCHMARKS)
    return scores


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def rebuild_performance_log(root: Path, model: str, log_path: Path) -> None:
    if model not in BACKBONES:
        raise ValueError(f"Unsupported model: {model}")
    lines = ["Model                  Ratio   MVBench  EgoSchema  LongVideoBench  VideoMME   Avg"]
    for ratio in RATIOS:
        path = root / "evaluation" / model / f"r{ratio}" / "scores.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        scores = record["scores"]
        lines.append(
            f"{model:22s} {ratio:>3d}%   "
            f"{scores['MVBench']:7.2f}  {scores['EgoSchema']:9.2f}  "
            f"{scores['LongVideoBench']:14.2f}  {scores['VideoMME']:8.2f}  "
            f"{scores['Avg']:6.2f}"
        )
    _write_atomic(log_path, "\n".join(lines) + "\n")


def record_performance(root: Path, model: str, ratio: int, result_path: Path, log_path: Path) -> None:
    if model not in BACKBONES or ratio not in RATIOS:
        raise ValueError("Unsupported model or retention ratio")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    scores = performance_scores(payload)
    destination = root / "evaluation" / model / f"r{ratio}" / "scores.json"
    _write_atomic(
        destination,
        json.dumps({"model": model, "ratio_percent": ratio, "scores": scores}, indent=2) + "\n",
    )
    rebuild_performance_log(root, model, log_path)


def rebuild_feature_log(root: Path, log_path: Path) -> None:
    lines = ["Ratio  Benchmark       Videos  L2-NQE  Cosine Coverage@0.90  Actual retention"]
    for ratio in RATIOS:
        path = root / "feature_analysis" / f"r{ratio}" / "summary.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        for row in record["rows"]:
            lines.append(
                f"{ratio:>3d}%   {row['benchmark']:15s} {row['videos']:>6d}  "
                f"{row['nqe_l2']:6.3f}  {100 * row['cosine_coverage_at_0.90']:>19.2f}%  "
                f"{100 * row['mean_actual_retention']:>15.2f}%"
            )
        lines.append("")
    _write_atomic(log_path, "\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", choices=BACKBONES, required=True)
    parser.add_argument("--ratio-percent", type=int, choices=RATIOS, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()
    record_performance(args.run_root, args.model, args.ratio_percent, args.result_json, args.log_path)


if __name__ == "__main__":
    main()
