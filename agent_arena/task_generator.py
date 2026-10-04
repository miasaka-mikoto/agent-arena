"""Public Task Generator API.

``TaskGenerator`` wraps the pure functions in :mod:`task_catalog` for callers
that prefer an object with a fixed seed.  It also supports registering small
custom templates, allowing experiments to grow without changing the built-in
benchmark catalogue.
"""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .task_catalog import (
    ADVERSARIAL_VARIANTS,
    BUILTIN_CATEGORIES,
    build_task,
    generate_demo_tasks,
    generate_tasks,
    make_adversarial_variant,
)
from .tasking import TaskDefinition


Template = Callable[[int, str, str], TaskDefinition]


class TaskGenerator:
    """Deterministic generator suitable for benchmark scripts and a UI."""

    def __init__(self, *, seed: int = 0, templates: Optional[Mapping[str, Template]] = None):
        self.seed = int(seed)
        self._templates: Dict[str, Template] = dict(templates or {})

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*BUILTIN_CATEGORIES, *self._templates]))

    def register_template(self, name: str, builder: Template) -> None:
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            raise ValueError("template name cannot be empty")
        if not callable(builder):
            raise TypeError("builder must be callable")
        self._templates[key] = builder

    def one(self, category: str, *, index: int = 0, difficulty: str = "easy",
            task_id: Optional[str] = None) -> TaskDefinition:
        key = str(category).strip().lower().replace("-", "_").replace(" ", "_")
        child_seed = self.seed + int(index)
        if key in self._templates:
            generated_id = task_id or f"custom-{key}-{self.seed:08x}-{index:03d}"
            return self._templates[key](child_seed, difficulty, generated_id)
        generated_id = task_id or f"task-{index + 1:03d}-{self.seed:08x}"
        return build_task(key, seed=child_seed, difficulty=difficulty, task_id=generated_id)

    def generate(self, count: int, *, categories: Optional[Sequence[str]] = None,
                 difficulties: Optional[Sequence[str]] = None,
                 adversarial_rate: float = 0.0,
                 adversarial_variants: Optional[Sequence[str]] = None) -> List[TaskDefinition]:
        # Built-ins can use the optimized shared generator.  For custom
        # templates we perform the same round-robin logic locally.
        selected = list(categories or self.categories)
        if selected and all(c in BUILTIN_CATEGORIES for c in selected):
            return generate_tasks(
                count,
                seed=self.seed,
                categories=selected,
                difficulties=difficulties,
                adversarial_rate=adversarial_rate,
                adversarial_variants=adversarial_variants,
            )
        if count < 0:
            raise ValueError("count must be >= 0")
        diff = list(difficulties or ("easy", "medium", "hard"))
        if not selected:
            raise ValueError("at least one category is required")
        if not diff:
            raise ValueError("at least one difficulty is required")
        if not 0.0 <= adversarial_rate <= 1.0:
            raise ValueError("adversarial_rate must be between 0 and 1")
        variants = list(adversarial_variants or ADVERSARIAL_VARIANTS)
        unknown = [variant for variant in variants if variant not in ADVERSARIAL_VARIANTS]
        if unknown:
            raise ValueError(f"unknown adversarial variant(s): {unknown}")
        rng = random.Random(self.seed)
        output: List[TaskDefinition] = []
        for index in range(count):
            task = self.one(selected[index % len(selected)], index=index,
                            difficulty=diff[index % len(diff)])
            if variants and rng.random() < adversarial_rate:
                variant = variants[rng.randrange(len(variants))]
                task = make_adversarial_variant(task, variant)
            output.append(task)
        return output

    def demo(self, count: int = 50, *, adversarial_rate: float = 0.30) -> List[TaskDefinition]:
        return generate_demo_tasks(count=count, seed=self.seed, adversarial_rate=adversarial_rate)

    @staticmethod
    def adversarial(task: TaskDefinition, variant: str) -> TaskDefinition:
        return make_adversarial_variant(task, variant)


__all__ = ["TaskGenerator", "Template"]
