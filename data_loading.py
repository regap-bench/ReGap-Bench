from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


VALID_TASKS = (
    "shape_counting",
    "angle_ranking",
    "size_ranking",
    "clock_ranking",
    "transformation_prediction",
)
REQUESTED_SETTINGS = (
    "images_plain",
    "images_grid",
    "images_shading",
    "images_number",
)
VALID_SETTINGS = REQUESTED_SETTINGS
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DEFAULT_EVALUATION_ROOT = Path(__file__).resolve().parent / "evaluation_results"


@dataclass(frozen=True)
class DataSelection:
    task: str
    setting: str
    data_root: Path
    task_dir: Path
    image_dir: Path
    sample_dir: Path
    source_image_dir_name: str

    @property
    def question_dir(self) -> Path:
        return self.sample_dir


@dataclass(frozen=True)
class ExperimentExample:
    example_id: int
    image_path: Path
    sample_path: Path
    question_data: dict

    @property
    def question_path(self) -> Path:
        return self.sample_path


def validate_task(task: str) -> str:
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {', '.join(VALID_TASKS)}")
    return task


def validate_setting(setting: str) -> str:
    if setting not in VALID_SETTINGS:
        raise ValueError(
            f"Unknown setting '{setting}'. Expected one of: {', '.join(VALID_SETTINGS)}"
        )
    return setting


def resolve_task_dir(task: str, data_root: Path | str = DEFAULT_DATA_ROOT) -> Path:
    task = validate_task(task)
    task_dir = Path(data_root).expanduser().resolve() / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    return task_dir


def resolve_sample_dir(task: str, data_root: Path | str = DEFAULT_DATA_ROOT) -> Path:
    task_dir = resolve_task_dir(task, data_root)
    sample_dir = task_dir / "samples"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {sample_dir}")
    return sample_dir


def resolve_question_dir(task: str, data_root: Path | str = DEFAULT_DATA_ROOT) -> Path:
    return resolve_sample_dir(task, data_root)


def discover_image_dirs(task: str, data_root: Path | str = DEFAULT_DATA_ROOT) -> list[str]:
    task_dir = resolve_task_dir(task, data_root)
    image_dirs = []
    for child in sorted(task_dir.iterdir()):
        if child.is_dir() and any(p.suffix.lower() in IMAGE_EXTENSIONS for p in child.iterdir()):
            image_dirs.append(child.name)
    return image_dirs


def resolve_data_selection(
    task: str,
    setting: str,
    data_root: Path | str = DEFAULT_DATA_ROOT,
) -> DataSelection:
    task = validate_task(task)
    setting = validate_setting(setting)
    task_dir = resolve_task_dir(task, data_root)
    sample_dir = resolve_sample_dir(task, data_root)

    image_dir = task_dir / setting
    if image_dir.exists():
        return DataSelection(
            task=task,
            setting=setting,
            data_root=Path(data_root).expanduser().resolve(),
            task_dir=task_dir,
            image_dir=image_dir,
            sample_dir=sample_dir,
            source_image_dir_name=setting,
        )

    available = discover_image_dirs(task, data_root)
    available_text = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"No image directory found for task='{task}' setting='{setting}'. "
        f"Looked for: {image_dir}. "
        f"Available image directories: {available_text}."
    )


def load_sample(sample_path: Path | str) -> dict:
    sample_path = Path(sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")
    with sample_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_question(question_path: Path | str) -> dict:
    return load_sample(question_path)


def numeric_image_files(
    image_dir: Path | str,
    start_id: int | None = None,
    end_id: int | None = None,
) -> list[Path]:
    image_dir = Path(image_dir)
    files = []
    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            example_id = int(image_path.stem)
        except ValueError:
            continue
        if start_id is not None and example_id < start_id:
            continue
        if end_id is not None and example_id > end_id:
            continue
        files.append((example_id, image_path))
    return [path for _, path in sorted(files, key=lambda item: item[0])]


def iter_examples(
    selection: DataSelection,
    start_id: int | None = None,
    end_id: int | None = None,
) -> Iterator[ExperimentExample]:
    for image_path in numeric_image_files(selection.image_dir, start_id, end_id):
        example_id = int(image_path.stem)
        sample_path = selection.sample_dir / f"{example_id}.json"
        yield ExperimentExample(
            example_id=example_id,
            image_path=image_path,
            sample_path=sample_path,
            question_data=load_sample(sample_path),
        )


def safe_model_name(model_name_or_path: str) -> str:
    normalized = model_name_or_path.strip().replace("\\", "/").rstrip("/")
    basename = normalized.split("/")[-1] if normalized else "model"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in basename)
    return safe or "model"


def output_dir_for(
    output_root: Path | str,
    task: str,
    setting: str,
    model_name_or_path: str,
) -> Path:
    validate_task(task)
    validate_setting(setting)
    return Path(output_root).expanduser().resolve() / task / setting / safe_model_name(model_name_or_path)
