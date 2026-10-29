from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data_loading import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REQUESTED_SETTINGS,
    VALID_TASKS,
    iter_examples,
    output_dir_for,
    resolve_data_selection,
)
from modeling import PROVIDERS, get_model
from prompting import select_prompt


# Providers whose model.run takes a list of prompts and images and returns a list of outputs.
BATCHED_PROVIDERS = ("qwen35",)


def infer_provider_from_model(model_name_or_path: str) -> str | None:
    model_lower = model_name_or_path.lower()
    if "gemini" in model_lower:
        return "gemini"
    if "qwen" in model_lower:
        return "qwen35"
    if "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    return None


def has_nonempty_model_output(result_file: Path) -> bool:
    try:
        with result_file.open("r", encoding="utf-8") as f:
            result_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    output = result_data.get("raw_model_output")
    if output is None:
        return False
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, (list, dict)):
        return bool(output)
    return True


def iter_batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def write_result(
    result_file: Path,
    example,
    task: str,
    setting: str,
    source_image_dir_name: str,
    prompt: str,
    provider: str,
    model_name_or_path: str,
    model_output: str,
) -> None:
    output_payload = dict(example.question_data)
    output_payload.update(
        {
            "task": task,
            "setting": setting,
            "source_image_dir": source_image_dir_name,
            "image_file": example.image_path.name,
            "image_path": str(example.image_path),
            "sample_path": str(example.sample_path),
            "prompt": prompt,
            "provider": provider,
            "model_name": model_name_or_path,
            "raw_model_output": model_output,
        }
    )
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4, ensure_ascii=False)


def run_inference(
    task: str,
    setting: str,
    model_name_or_path: str,
    provider: str | None = None,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    start_id: int | None = None,
    end_id: int | None = None,
    sleep_seconds: float = 0.0,
    overwrite: bool = False,
    batch_size: int = 8,
    **model_kwargs,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    provider = provider or infer_provider_from_model(model_name_or_path)
    if not provider:
        raise ValueError(
            "Could not infer provider from model name. Pass --provider explicitly."
        )

    selection = resolve_data_selection(task, setting, data_root)
    final_output_dir = output_dir_for(output_root, task, setting, model_name_or_path)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: {selection.task}")
    print(f"Setting: {selection.setting} (source images: {selection.source_image_dir_name})")
    print(f"Images: {selection.image_dir}")
    print(f"Samples: {selection.sample_dir}")
    print(f"Output: {final_output_dir}")

    examples = list(iter_examples(selection, start_id=start_id, end_id=end_id))
    if not examples:
        print("No matching examples found.")
        return final_output_dir

    pending_examples = []
    for example in examples:
        result_file = final_output_dir / f"{example.example_id}.json"
        if result_file.exists() and not overwrite:
            if has_nonempty_model_output(result_file):
                print(f"Skipping {example.example_id}: result already has non-empty output.")
                continue
            print(f"Regenerating {example.example_id}: existing result has empty output.")
        pending_examples.append(example)

    if not pending_examples:
        print(f"All {len(examples)} examples already have non-empty outputs.")
        return final_output_dir

    model = get_model(provider, model_name_or_path, **model_kwargs)

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for image loading. Install the PIL/Pillow package.") from exc

    # API models take one sample per call; batched models take a whole batch.
    batched = provider in BATCHED_PROVIDERS
    if not batched:
        batch_size = 1
    print(f"Found {len(examples)} examples; {len(pending_examples)} require inference.")
    for batch_examples in iter_batches(pending_examples, batch_size):
        print(f"Inferencing {', '.join(example.image_path.name for example in batch_examples)} ...")

        prompts = []
        images = []
        for example in batch_examples:
            prompts.append(select_prompt(task, example.question_data))
            with Image.open(example.image_path) as img:
                images.append(img.convert("RGB"))

        if batched:
            model_outputs = model.run(prompts, images)
        else:
            model_outputs = [model.run(prompts[0], images[0])]

        if len(model_outputs) != len(batch_examples):
            raise RuntimeError(
                f"Model returned {len(model_outputs)} outputs for "
                f"{len(batch_examples)} inputs."
            )

        for example, prompt, model_output in zip(batch_examples, prompts, model_outputs):
            write_result(
                result_file=final_output_dir / f"{example.example_id}.json",
                example=example,
                task=task,
                setting=setting,
                source_image_dir_name=selection.source_image_dir_name,
                prompt=prompt,
                provider=provider,
                model_name_or_path=model_name_or_path,
                model_output=model_output,
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print(f"Inference complete. Raw outputs saved to: {final_output_dir}")
    return final_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReGap-Bench inference.")
    parser.add_argument("--task", required=True, choices=VALID_TASKS)
    parser.add_argument("--setting", required=True, choices=REQUESTED_SETTINGS)
    parser.add_argument("--model", required=True, help="Model name or local path passed to modeling.get_model.")
    parser.add_argument("--provider", choices=PROVIDERS, help="Inferred from the model name when omitted.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--end-id", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result JSON files.")

    api = parser.add_argument_group("openai / gemini")
    api.add_argument("--request-timeout", type=int, help="Per-request timeout in seconds (openai).")
    api.add_argument("--max-retries", type=int, help="Maximum request attempts per sample (openai).")
    api.add_argument("--retry-sleep-seconds", type=float, help="Initial sleep before retrying (openai).")
    api.add_argument("--retry-backoff", type=float, help="Retry sleep multiplier (openai).")
    api.add_argument("--max-completion-tokens", type=int, help="Maximum output tokens per sample (openai).")
    api.add_argument("--temperature", type=float)

    local = parser.add_argument_group("qwen35 (vLLM)")
    local.add_argument("--batch-size", type=int, default=8)
    local.add_argument("--max-new-tokens", type=int)
    local.add_argument("--top-p", type=float)
    local.add_argument("--top-k", type=int)
    local.add_argument("--tensor-parallel-size", type=int)
    local.add_argument("--gpu-memory-utilization", type=float)
    local.add_argument("--max-model-len", type=int)
    local.add_argument("--max-num-seqs", type=int)
    local.add_argument("--dtype")
    local.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    local.add_argument("--stop-on-answer", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        task=args.task,
        setting=args.setting,
        model_name_or_path=args.model,
        provider=args.provider,
        data_root=args.data_root,
        output_root=args.output_root,
        start_id=args.start_id,
        end_id=args.end_id,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        retry_backoff=args.retry_backoff,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        enable_thinking=args.enable_thinking,
        stop_on_answer=args.stop_on_answer,
    )


if __name__ == "__main__":
    main()
