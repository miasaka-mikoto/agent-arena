"""Serializable task models used by the catalogue and benchmark runner.

The execution engine only needs a small contract.  Keeping that contract in a
plain dataclass makes task files portable (JSON/YAML adapters can be layered on
later) and, importantly, prevents hidden ground truth from being accidentally
mixed into an agent observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional


class TaskStatus(str, Enum):
    """Lifecycle status for a task run or a generated task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TaskDefinition:
    """A complete benchmark task specification.

    ``initial_state`` is public environment input.  ``hidden_ground_truth`` is
    evaluator-only data and should never be copied into an Observation.  The
    two dictionaries intentionally remain independent via ``from_dict`` and
    ``clone`` deep copies.
    """

    task_id: str
    instruction: str
    initial_state: Dict[str, Any] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    hidden_ground_truth: Dict[str, Any] = field(default_factory=dict)
    success_conditions: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = 30.0
    max_steps: int = 20
    difficulty: str = "easy"
    tags: List[str] = field(default_factory=list)
    category: str = "generic"
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_id is None or str(self.task_id) == "":
            raise ValueError("task_id must be a non-empty string")
        self.task_id = str(self.task_id)
        self.instruction = str(self.instruction)
        try:
            self.max_steps = int(self.max_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_steps must be >= 1") from exc
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.timeout is not None:
            try:
                self.timeout = float(self.timeout)
            except (TypeError, ValueError) as exc:
                raise ValueError("timeout must be > 0") from exc
            if self.timeout <= 0:
                raise ValueError("timeout must be > 0")
        self.initial_state = _deepcopy(self.initial_state if self.initial_state is not None else {})
        self.hidden_ground_truth = _deepcopy(self.hidden_ground_truth if self.hidden_ground_truth is not None else {})
        self.success_conditions = _deepcopy(self.success_conditions if self.success_conditions is not None else {})
        self.metadata = _deepcopy(self.metadata if self.metadata is not None else {})
        self.allowed_tools = list(dict.fromkeys(self.allowed_tools or []))
        self.tags = list(dict.fromkeys(self.tags or []))
        if self.category == "generic" and self.tags:
            # A useful default for older task JSON that only supplied tags.
            category_tags = {
                "files", "email", "calendar", "code", "grid", "retrieval"
            }
            for tag in self.tags:
                if tag in category_tags:
                    self.category = tag
                    break

    @property
    def id(self) -> str:
        """Compatibility alias used by some runners."""

        return self.task_id

    @property
    def initial_observation(self) -> Dict[str, Any]:
        """A defensive copy of public state for environment reset."""

        return _deepcopy(self.initial_state)

    def clone(self, *, task_id: Optional[str] = None, **changes: Any) -> "TaskDefinition":
        """Return an independent task with optional field overrides."""

        values = self.to_dict()
        values.update(changes)
        if task_id is not None:
            values["task_id"] = task_id
        return type(self).from_dict(values)

    def to_dict(self, *, include_hidden: bool = True) -> Dict[str, Any]:
        """Convert to JSON-compatible data.

        ``include_hidden=False`` is the safe form for agent observations and
        replay payloads.  The default retains ground truth for dataset files
        and evaluator persistence.
        """

        result = _deepcopy(asdict(self))
        if not include_hidden:
            result.pop("hidden_ground_truth", None)
        return result

    def public_dict(self) -> Dict[str, Any]:
        """Return an agent-safe task mapping with no hidden world state.

        ``initial_state`` is the complete seed used to initialise the
        simulator and may reveal a full map, inbox, or filesystem.  Agents
        receive the instruction and tool contract, then learn the rest via
        observations and tools.
        """

        result = self.to_dict(include_hidden=False)
        result.pop("initial_state", None)
        result.pop("success_conditions", None)
        return result

    def copy(self, **changes: Any) -> "TaskDefinition":
        """Compatibility alias for :meth:`clone`."""

        return self.clone(**changes)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskDefinition":
        fields = {
            "task_id": data.get("task_id", data.get("id")),
            "instruction": data.get("instruction", ""),
            "initial_state": data.get("initial_state", {}),
            "allowed_tools": data.get("allowed_tools", []),
            "hidden_ground_truth": data.get("hidden_ground_truth", {}),
            "success_conditions": data.get("success_conditions", {}),
            "timeout": data.get("timeout", 30.0),
            "max_steps": data.get("max_steps", 20),
            "difficulty": data.get("difficulty", "easy"),
            "tags": data.get("tags", []),
            "category": data.get("category", "generic"),
            "seed": data.get("seed"),
            "metadata": data.get("metadata", {}),
        }
        return cls(**fields)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskDefinition":
        """Alias accepted by environment/runner adapters."""

        return cls.from_dict(data)


@dataclass
class TaskResult:
    """Small, runner-neutral result object for generated task smoke tests."""

    task_id: str
    status: TaskStatus | str
    score: float = 0.0
    steps: int = 0
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value if isinstance(self.status, TaskStatus) else str(self.status)
        return result


def _deepcopy(value: Any) -> Any:
    """Copy JSON-like values without requiring callers to import ``copy``."""

    # JSON round-tripping is deliberate: task state is required to be portable
    # and this catches accidental non-serializable values early.
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        import copy

        return copy.deepcopy(value)


def tasks_to_json(tasks: Iterable[TaskDefinition], *, indent: Optional[int] = 2) -> str:
    return json.dumps([task.to_dict() for task in tasks], ensure_ascii=False, indent=indent, sort_keys=True)


def tasks_from_json(payload: str) -> List[TaskDefinition]:
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("task dataset JSON must contain a list")
    return [TaskDefinition.from_dict(item) for item in raw]


# Friendly compatibility alias used by lightweight benchmark notebooks.
Task = TaskDefinition
