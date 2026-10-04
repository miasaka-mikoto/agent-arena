"""Compatibility facade for task catalogue imports.

Some integrations naturally import ``agent_arena.tasks`` while the package
keeps models and generation logic in separate modules.  Re-exporting here
keeps those integrations stable without introducing a second task schema.
"""

from .tasking import Task, TaskDefinition, TaskResult, TaskStatus, tasks_from_json, tasks_to_json
from .task_catalog import (
    ADVERSARIAL_VARIANTS,
    BUILTIN_CATEGORIES,
    CATEGORY_ALIASES,
    build_task,
    canonical_category,
    generate_demo_tasks,
    generate_tasks,
    generate_task,
    generate,
    demo_dataset,
    adversarial_task,
    make_adversarial_variant,
)

__all__ = [
    "TaskDefinition",
    "Task",
    "TaskResult",
    "TaskStatus",
    "tasks_from_json",
    "tasks_to_json",
    "ADVERSARIAL_VARIANTS",
    "BUILTIN_CATEGORIES",
    "CATEGORY_ALIASES",
    "build_task",
    "canonical_category",
    "generate_demo_tasks",
    "generate_tasks",
    "generate_task",
    "generate",
    "demo_dataset",
    "adversarial_task",
    "make_adversarial_variant",
]
