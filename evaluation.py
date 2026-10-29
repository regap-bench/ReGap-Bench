from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from data_loading import (
    DEFAULT_DATA_ROOT,
    DEFAULT_EVALUATION_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REQUESTED_SETTINGS,
    VALID_TASKS,
    load_sample,
    output_dir_for,
    resolve_sample_dir,
    safe_model_name,
    validate_setting,
    validate_task,
)


MODEL_OUTPUT_KEYS = (
    "raw_model_output",
    "model_output",
    "output",
    "model_answer",
    "answer",
    "response",
)
MULTIPLE_CHOICE_TASKS = ("transformation_prediction",)
INTEGER_COUNT_TASKS = ("shape_counting",)

def _coerce_output_text(output: Any) -> str | None:
    if isinstance(output, dict):
        for key in MODEL_OUTPUT_KEYS:
            if isinstance(output.get(key), str):
                return output[key]
        return None
    if not isinstance(output, str):
        return None
    try:
        data = json.loads(output)
    except Exception:
        return output
    if isinstance(data, dict):
        return _coerce_output_text(data) or output
    return output


def _answer_tag_contents(text: str) -> list[str]:
    return re.findall(r"<answer>\s*([^<>]*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)


def _clean_candidate(candidate: str) -> str:
    candidate = re.sub(r"\x60{3}(?:json|python)?\s*|\s*\x60{3}", "", candidate)
    return candidate.strip()


def extract_list_from_output(output: Any) -> list[int] | None:
    text = _coerce_output_text(output)
    if not text:
        return None

    candidates = list(reversed(_answer_tag_contents(text)))
    candidates.extend(reversed(re.findall(r"\[\s*[^\]]*?\s*\]", text, re.DOTALL)))

    for candidate in candidates:
        candidate = _clean_candidate(candidate)
        for parser in (ast.literal_eval, json.loads):
            try:
                to_parse = candidate
                if not to_parse.startswith("[") and ("," in to_parse or to_parse.isdigit()):
                    to_parse = f"[{to_parse}]"
                parsed = parser(to_parse)
            except Exception:
                continue
            if isinstance(parsed, list):
                try:
                    return [int(item) for item in parsed]
                except (TypeError, ValueError):
                    return None

        numbers = re.findall(r"-?\d+", candidate)
        if numbers:
            return [int(number) for number in numbers]

    return None


def extract_count_from_output(output: Any) -> int:
    text = _coerce_output_text(output)
    if not text:
        return -1

    match = re.search(r"<answer>\s*(\d+)\s*</answer>", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1


def normalize_choice_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    match = re.fullmatch(r"\(?\s*([A-D])\s*\)?\.?", text)
    if match:
        return match.group(1)
    return None


def _choice_from_candidate(candidate: str, *, allow_bare_letter: bool) -> str | None:
    candidate = _clean_candidate(candidate).strip().upper()
    if allow_bare_letter:
        normalized = normalize_choice_label(candidate)
        if normalized:
            return normalized

    patterns = (
        r"\b(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|=|-)?\s*\(?([A-D])\)?\b",
        r"\b(?:I\s+CHOOSE|I\s+SELECT|SELECTED|SELECT)\s*\(?([A-D])\)?\b",
    )
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, candidate, re.IGNORECASE))
    if matches:
        return matches[-1].upper()
    return None


def extract_choice_from_output(output: Any) -> str | None:
    text = _coerce_output_text(output)
    if not text:
        return None

    for candidate in reversed(_answer_tag_contents(text)):
        choice = _choice_from_candidate(candidate, allow_bare_letter=True)
        if choice:
            return choice
    return _choice_from_candidate(text, allow_bare_letter=False)


def select_answer_extractor(task: str):
    task = validate_task(task)
    if task in INTEGER_COUNT_TASKS:
        return extract_count_from_output
    if task in MULTIPLE_CHOICE_TASKS:
        return extract_choice_from_output
    return extract_list_from_output


def _numeric_json_files(path: Path) -> list[Path]:
    numbered = []
    for file_path in path.iterdir():
        if file_path.suffix != ".json":
            continue
        try:
            file_id = int(file_path.stem)
        except ValueError:
            continue
        numbered.append((file_id, file_path))
    return [file_path for _, file_path in sorted(numbered, key=lambda item: item[0])]


def _ground_truth_for(inference_data: dict, sample_dir: Path, result_file: Path) -> Any:
    if "answer" in inference_data and inference_data["answer"] not in (None, -1, []):
        return inference_data["answer"]
    sample_path = sample_dir / result_file.name
    return load_sample(sample_path).get("answer")


def _infer_metadata_from_result(inference_dir: Path) -> dict[str, str]:
    for result_file in _numeric_json_files(inference_dir):
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        metadata = {}
        if data.get("task") in VALID_TASKS:
            metadata["task"] = data["task"]
        if data.get("setting") in REQUESTED_SETTINGS:
            metadata["setting"] = data["setting"]
        if isinstance(data.get("model_name"), str):
            metadata["model"] = data["model_name"]
        return metadata
    return {}


def infer_metadata_from_output_path(output_path: Path | str) -> dict[str, str]:
    output_path = Path(output_path).expanduser().resolve()
    parts = output_path.parts
    metadata = _infer_metadata_from_result(output_path) if output_path.is_dir() else {}

    for index in range(len(parts) - 1, -1, -1):
        part = parts[index]
        if part not in VALID_TASKS:
            continue
        metadata.setdefault("task", part)
        if index + 1 < len(parts) and parts[index + 1] in REQUESTED_SETTINGS:
            metadata.setdefault("setting", parts[index + 1])
        if index + 2 < len(parts):
            metadata.setdefault("model", parts[index + 2])
        break

    return metadata


def evaluate_inference_dir(task: str, inference_dir: Path | str, sample_dir: Path | str) -> dict:
    task = validate_task(task)
    inference_dir = Path(inference_dir).expanduser().resolve()
    sample_dir = Path(sample_dir).expanduser().resolve()
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    extractor = select_answer_extractor(task)
    result_files = _numeric_json_files(inference_dir)
    correct = 0
    details = {}

    for result_file in result_files:
        with result_file.open("r", encoding="utf-8") as f:
            inference_data = json.load(f)

        raw_output = inference_data.get("raw_model_output", "")
        ground_truth = _ground_truth_for(inference_data, sample_dir, result_file)
        prediction = extractor(raw_output)
        if task in MULTIPLE_CHOICE_TASKS:
            ground_truth = normalize_choice_label(ground_truth)

        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1
        details[result_file.name] = {
            "extracted_prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
        }

    total = len(result_files)
    metrics = {
        "task": task,
        "total_evaluated": total,
        "correct_predictions": correct,
        "accuracy_percentage": (correct / total * 100.0) if total else 0.0,
    }

    return {"metrics": metrics, "details": details}


def run_evaluation(
    task: str | None = None,
    setting: str | None = None,
    model_name_or_path: str | None = None,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    eval_root: Path | str = DEFAULT_EVALUATION_ROOT,
    inference_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    eval_output_dir: Path | str | None = None,
    sample_dir: Path | str | None = None,
) -> Path:
    if output_path is not None:
        if inference_dir is not None and Path(inference_dir) != Path(output_path):
            raise ValueError("Use either inference_dir or output_path, not both.")
        inference_dir = output_path

    if inference_dir is None:
        if not task:
            raise ValueError("task is required when no output path is provided.")
        if not setting:
            raise ValueError("setting is required when no output path is provided.")
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required when inference_dir is not provided.")
        task = validate_task(task)
        setting = validate_setting(setting)
        inference_dir = output_dir_for(output_root, task, setting, model_name_or_path)
    else:
        inference_dir = Path(inference_dir)
        inferred = infer_metadata_from_output_path(inference_dir)
        task = task or inferred.get("task")
        setting = setting or inferred.get("setting")
        model_name_or_path = model_name_or_path or inferred.get("model")
        if not task:
            raise ValueError(
                "Could not infer task from output path or result files. Pass --task explicitly."
            )
        task = validate_task(task)
        if setting:
            setting = validate_setting(setting)

    setting_label = setting or "unknown_setting"
    if eval_output_dir is None:
        if model_name_or_path:
            model_dir_name = safe_model_name(model_name_or_path)
        else:
            model_dir_name = Path(inference_dir).name
        eval_output_dir = Path(eval_root).expanduser().resolve() / task / setting_label / model_dir_name
    else:
        eval_output_dir = Path(eval_output_dir).expanduser().resolve()

    sample_dir = Path(sample_dir).expanduser().resolve() if sample_dir else resolve_sample_dir(task, data_root)
    report = evaluate_inference_dir(task, inference_dir, sample_dir)
    report["metrics"]["setting"] = setting_label
    report["metrics"]["model_name"] = model_name_or_path or Path(inference_dir).name
    report["paths"] = {
        "inference_dir": str(Path(inference_dir).expanduser().resolve()),
        "sample_dir": str(sample_dir),
    }

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_output_dir / "final_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    metrics = report["metrics"]
    print("Evaluation complete.")
    print(f"Report: {report_path}")
    print(
        f"Accuracy: {metrics['accuracy_percentage']:.2f}% "
        f"({metrics['correct_predictions']}/{metrics['total_evaluated']})"
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate consolidated experiment outputs.")
    parser.add_argument("--task", choices=VALID_TASKS)
    parser.add_argument("--setting", choices=REQUESTED_SETTINGS)
    parser.add_argument("--model", help="Model name/path used during inference.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument(
        "--output-path",
        "--inference-dir",
        dest="output_path",
        type=Path,
        help="Evaluate this existing model output directory.",
    )
    parser.add_argument("--eval-output-dir", type=Path, help="Override where the evaluation report is written.")
    parser.add_argument(
        "--sample-dir",
        "--question-dir",
        dest="sample_dir",
        type=Path,
        help="Override the sample metadata directory.",
    )
    args = parser.parse_args()
    if args.output_path is None and not (args.task and args.setting and args.model):
        parser.error("--task, --setting, and --model are required unless --output-path is provided")
    return args


def main() -> None:
    args = parse_args()
    run_evaluation(
        task=args.task,
        setting=args.setting,
        model_name_or_path=args.model,
        data_root=args.data_root,
        output_root=args.output_root,
        eval_root=args.eval_root,
        output_path=args.output_path,
        eval_output_dir=args.eval_output_dir,
        sample_dir=args.sample_dir,
    )


if __name__ == "__main__":
    main()
