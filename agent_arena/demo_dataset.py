"""Helpers for creating and persisting the standard Agent Arena demo set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from .task_catalog import generate_demo_tasks
from .tasking import TaskDefinition, tasks_from_json, tasks_to_json


DEFAULT_DEMO_COUNT = 50
DEFAULT_DEMO_SEED = 20261004


def load_or_create_demo_dataset(
    path: str | Path,
    *,
    count: int = DEFAULT_DEMO_COUNT,
    seed: int = DEFAULT_DEMO_SEED,
    adversarial_rate: float = 0.30,
) -> List[TaskDefinition]:
    """Load a dataset if present, otherwise create it and write it atomically."""

    target = Path(path)
    if target.exists():
        return load_dataset(target)
    tasks = generate_demo_tasks(count=count, seed=seed, adversarial_rate=adversarial_rate)
    save_dataset(tasks, target)
    return tasks


def save_dataset(tasks: Iterable[TaskDefinition], path: str | Path) -> Path:
    """Write task JSON, creating parent directories as needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = tasks_to_json(tasks, indent=2)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_dataset(path: str | Path) -> List[TaskDefinition]:
    target = Path(path)
    return tasks_from_json(target.read_text(encoding="utf-8"))


def dataset_summary(tasks: Iterable[TaskDefinition]) -> dict:
    """Return stable counts used by the dashboard and smoke tests."""

    task_list = list(tasks)
    by_category: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    adversarial = 0
    for task in task_list:
        by_category[task.category] = by_category.get(task.category, 0) + 1
        by_difficulty[task.difficulty] = by_difficulty.get(task.difficulty, 0) + 1
        if task.metadata.get("is_adversarial") or "adversarial" in task.tags:
            adversarial += 1
    return {
        "count": len(task_list),
        "categories": dict(sorted(by_category.items())),
        "difficulties": dict(sorted(by_difficulty.items())),
        "adversarial": adversarial,
    }


__all__ = [
    "DEFAULT_DEMO_COUNT",
    "DEFAULT_DEMO_SEED",
    "dataset_summary",
    "load_dataset",
    "load_or_create_demo_dataset",
    "save_dataset",
]

