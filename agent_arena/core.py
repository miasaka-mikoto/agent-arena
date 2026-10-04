"""Core contracts and offline agents for Agent Arena.

The project intentionally keeps this module dependency free.  Environments and
the benchmark runner exchange small, serialisable dataclasses instead of
framework-specific objects.  Agents never receive a complete environment
unless an environment explicitly puts it in the public observation.

The module also contains four deterministic/offline reference agents.  They
are useful as baselines and test fixtures; none of them call a paid model API.
"""

from __future__ import annotations

import copy
import inspect
import random
import time
import uuid
from abc import ABC
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, ClassVar, Optional, Union


JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, list["JSONValue"], dict[str, "JSONValue"]]


def _jsonable(value: Any) -> Any:
    """Convert a model value into a JSON-friendly copy.

    Hidden ground truth and arbitrary environment payloads are deliberately
    not inspected here.  This is only a safe serializer for the public model
    objects; callers decide whether a field is safe to expose.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    # Keep serialization total: an environment may put a small custom object
    # in metadata.  Repr is preferable to failing a benchmark report.
    return repr(value)


def _copy(value: Any) -> Any:
    """Best-effort defensive copy used at API boundaries."""

    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _first_identifier(value: Any, keys: Sequence[str], *, nested_keys: Sequence[str] = ()) -> Any:
    """Find the first useful identifier in a tool result or observation.

    Rule-based agents use this small, deliberately conservative extractor to
    carry a path/message/object id from one tool result into the next call.
    It only follows mappings and sequences supplied by the public observation;
    no hidden environment state is consulted.
    """

    wanted = {str(key) for key in keys}
    nested = {str(key) for key in nested_keys}

    def visit(node: Any) -> Any:
        if isinstance(node, Mapping):
            for key in wanted:
                candidate = node.get(key)
                if candidate not in (None, "", [], {}):
                    return candidate
            for key, child in node.items():
                if str(key) in nested:
                    found = visit(child)
                    if found not in (None, "", [], {}):
                        return found
            # A result may wrap a single item under an arbitrary key.  Only
            # recurse one level into non-control values to avoid accidentally
            # selecting an unrelated number or boolean.
            for child in node.values():
                if isinstance(child, (Mapping, list, tuple)):
                    found = visit(child)
                    if found not in (None, "", [], {}):
                        return found
        elif isinstance(node, (list, tuple)):
            for child in node:
                found = visit(child)
                if found not in (None, "", [], {}):
                    return found
        return None

    return visit(value)


@dataclass
class TaskDefinition:
    """A benchmark task specification.

    ``hidden_ground_truth`` is retained by the runner/evaluator and should not
    be included in an observation sent to an agent.  ``success_conditions`` is
    intentionally untyped: simple tasks use mappings/lists while custom tasks
    may keep a callable predicate in the runner process.
    """

    task_id: str
    instruction: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    hidden_ground_truth: Any = None
    success_conditions: Any = field(default_factory=dict)
    timeout: float | None = None
    max_steps: int = 50
    difficulty: str | int = "easy"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_id = str(self.task_id)
        self.instruction = str(self.instruction)
        self.initial_state = dict(self.initial_state or {})
        self.allowed_tools = _normalise_names(self.allowed_tools)
        self.tags = _normalise_names(self.tags)
        self.metadata = dict(self.metadata or {})
        try:
            self.max_steps = max(0, int(self.max_steps))
        except (TypeError, ValueError):
            self.max_steps = 50
        if self.timeout is not None:
            try:
                self.timeout = max(0.0, float(self.timeout))
            except (TypeError, ValueError):
                self.timeout = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskDefinition":
        """Build a task from JSON/YAML-like data.

        A few common key spellings are accepted to make task datasets easy to
        author by hand (``id``/``task_id``, ``tools``/``allowed_tools``).
        """

        data = dict(value)
        if "task_id" not in data:
            data["task_id"] = data.pop("id", "task")
        if "allowed_tools" not in data and "tools" in data:
            data["allowed_tools"] = data.pop("tools")
        if "success_conditions" not in data and "success" in data:
            data["success_conditions"] = data.pop("success")
        return cls(**{k: v for k, v in data.items() if k in _field_names(cls)})

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskDefinition":
        return cls.from_dict(value)

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "initial_state": _jsonable(self.initial_state),
            "allowed_tools": list(self.allowed_tools),
            "success_conditions": _jsonable(self.success_conditions),
            "timeout": self.timeout,
            "max_steps": self.max_steps,
            "difficulty": self.difficulty,
            "tags": list(self.tags),
            "metadata": _jsonable(self.metadata),
        }
        if include_hidden:
            result["hidden_ground_truth"] = _jsonable(self.hidden_ground_truth)
        return result

    def public_dict(self) -> dict[str, Any]:
        """Return the agent-safe task view (never exposes ground truth)."""

        data = self.to_dict(include_hidden=False)
        data.pop("initial_state", None)
        data.pop("success_conditions", None)
        return data

    def copy(self, **changes: Any) -> "TaskDefinition":
        data = self.to_dict(include_hidden=True)
        data.update(changes)
        return TaskDefinition.from_dict(data)


@dataclass
class ToolResult:
    """Result returned by an environment tool."""

    tool: str = ""
    success: bool = True
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tool = str(self.tool or "")
        self.success = bool(self.success)
        self.output = _copy(self.output)
        try:
            self.latency_ms = max(0.0, float(self.latency_ms))
        except (TypeError, ValueError):
            self.latency_ms = 0.0
        self.metadata = _copy(dict(self.metadata or {}))

    @property
    def ok(self) -> bool:
        return self.success

    @property
    def result(self) -> Any:
        return self.output

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "success": self.success,
            "output": _jsonable(self.output),
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_any(cls, value: Any, *, tool: str = "") -> "ToolResult":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
            if "success" not in data and "ok" in data:
                data["success"] = data.pop("ok")
            if "output" not in data and "result" in data:
                data["output"] = data.pop("result")
            if "error" not in data and data.get("success") is False and data.get("message") not in (None, ""):
                data["error"] = str(data["message"])
            data.setdefault("tool", tool)
            return cls(**{k: v for k, v in data.items() if k in _field_names(cls)})
        # Environment adapters often return a small dataclass/object rather
        # than a mapping.  Preserve its public result fields instead of
        # wrapping the object as opaque output (which would hide failures).
        if value is not None and any(hasattr(value, key) for key in ("success", "ok", "output", "result", "error")):
            success = getattr(value, "success", getattr(value, "ok", True))
            output = getattr(value, "output", getattr(value, "result", None))
            return cls(
                tool=str(getattr(value, "tool", tool) or tool),
                success=bool(success),
                output=output,
                error=getattr(value, "error", None),
                latency_ms=getattr(value, "latency_ms", getattr(value, "latency", 0.0)),
                metadata=dict(getattr(value, "metadata", {}) or {}),
            )
        return cls(tool=tool, success=True, output=value)


@dataclass
class Action:
    """A single agent action.

    Most Arena environments expose actions through tools, so ``tool`` is the
    primary field.  Native environment actions (for example ``move`` in a
    grid world) can use ``action_type`` with ``tool=None``.  ``reason`` is a
    short public summary only; private chain-of-thought is never requested or
    stored.
    """

    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    action_type: str = "tool"
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    def __post_init__(self) -> None:
        if self.tool is not None:
            self.tool = str(self.tool)
        self.arguments = _copy(dict(self.arguments or {}))
        self.action_type = str(self.action_type or ("tool" if self.tool else "action"))
        if self.reason is not None:
            # Reasons are public summaries, not a channel for private
            # chain-of-thought.  Keep traces/report payloads bounded.
            self.reason = str(self.reason)[:2000]
        self.metadata = _copy(dict(self.metadata or {}))
        if self.call_id is not None:
            self.call_id = str(self.call_id)

    @property
    def type(self) -> str:
        """Alias commonly used by environment adapters."""

        return self.action_type

    @property
    def args(self) -> dict[str, Any]:
        return self.arguments

    @property
    def name(self) -> str | None:
        return self.tool

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool)

    @property
    def finish(self) -> bool:
        """Whether this action explicitly terminates an episode."""

        value = str(self.action_type or "").lower()
        return value in {"finish", "done", "stop", "terminate", "end"} or bool(self.metadata.get("finish", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": _jsonable(self.arguments),
            "action_type": self.action_type,
            "reason": self.reason,
            "metadata": _jsonable(self.metadata),
            "call_id": self.call_id,
            "finish": self.finish,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Mapping-like access for lightweight runners."""

        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    @classmethod
    def from_any(cls, value: Any, *, default_tool: str | None = None) -> "Action | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, ToolCall):
            return value.to_action()
        if isinstance(value, str):
            return cls(tool=value, action_type="tool")
        if isinstance(value, Mapping):
            data = dict(value)
            tool = data.get("tool", data.get("name", default_tool))
            action_type = data.get("action_type", data.get("type", data.get("action", "tool")))
            if data.get("finish", data.get("done", data.get("terminate", False))):
                action_type = "finish"
            arguments = data.get("arguments", data.get("args", data.get("parameters", {})))
            return cls(
                tool=tool,
                arguments=dict(arguments or {}),
                action_type=action_type,
                reason=data.get("reason", data.get("summary")),
                metadata=dict(data.get("metadata") or {}),
                call_id=data.get("call_id", data.get("id")),
            )
        raise TypeError(f"Cannot convert {type(value).__name__} to Action")


@dataclass
class ToolCall:
    """Explicit tool-call form accepted by the runner."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tool = str(self.tool)
        self.arguments = dict(self.arguments or {})
        self.metadata = dict(self.metadata or {})

    @property
    def name(self) -> str:
        return self.tool

    @property
    def args(self) -> dict[str, Any]:
        return self.arguments

    def to_action(self) -> Action:
        return Action(
            tool=self.tool,
            arguments=self.arguments,
            action_type="tool",
            reason=self.reason,
            metadata=self.metadata,
            call_id=self.call_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_action().to_dict()


@dataclass
class Observation:
    """Agent-visible environment observation.

    ``state`` is public state only.  The environment may keep its full ground
    truth elsewhere; this model does not have a hidden-state field.
    """

    state: Any = field(default_factory=dict)
    available_tools: list[str] = field(default_factory=list)
    step: int = 0
    done: bool = False
    last_action: Action | None = None
    last_result: ToolResult | None = None
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = _copy(self.state)
        self.available_tools = _normalise_names(self.available_tools)
        try:
            self.step = max(0, int(self.step))
        except (TypeError, ValueError):
            self.step = 0
        self.done = bool(self.done)
        if self.last_action is not None:
            self.last_action = Action.from_any(self.last_action)
        if self.last_result is not None:
            self.last_result = ToolResult.from_any(self.last_result)
        self.metadata = dict(self.metadata or {})

    @property
    def visible_state(self) -> Any:
        return self.state

    @property
    def tools(self) -> list[str]:
        return self.available_tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": _jsonable(self.state),
            "available_tools": list(self.available_tools),
            "step": self.step,
            "done": self.done,
            "last_action": self.last_action.to_dict() if self.last_action else None,
            "last_result": self.last_result.to_dict() if self.last_result else None,
            "reward": self.reward,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_any(cls, value: Any) -> "Observation":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            data = dict(value)
            # Environments commonly return their public state directly (for
            # example ``{"position": ..., "available_tools": [...]}``)
            # rather than wrapping it under ``state``.  Keep that raw mapping
            # intact; dropping it would make every baseline agent effectively
            # blind after the first observation.
            if "state" in data:
                state = data["state"]
            elif "visible_state" in data:
                state = data["visible_state"]
            elif "observation" in data:
                state = data["observation"]
            else:
                state = dict(data)
            tools = data.get("available_tools", data.get("tools", data.get("allowed_tools", [])))
            return cls(
                state=state,
                available_tools=_extract_tool_names(tools),
                step=data.get("step", 0),
                done=data.get("done", data.get("finished", False)),
                last_action=Action.from_any(data["last_action"]) if data.get("last_action") else None,
                last_result=ToolResult.from_any(data["last_result"]) if data.get("last_result") else None,
                reward=data.get("reward"),
                metadata=dict(data.get("metadata") or {}),
            )
        # A bare state is legal and convenient for tiny test environments.
        return cls(state=value)


@dataclass
class Plan:
    """A short, public plan summary and candidate actions."""

    actions: list[Action] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actions = [a for a in (Action.from_any(v) for v in self.actions) if a is not None]
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "summary": self.summary,
            "metadata": _jsonable(self.metadata),
        }

    def __iter__(self):
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class AgentResult:
    """Public result emitted when an agent stops acting."""

    status: str = "finished"
    success: bool | None = None
    message: str = ""
    steps: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "message": self.message,
            "steps": self.steps,
            "metrics": _jsonable(self.metrics),
            "artifacts": _jsonable(self.artifacts),
        }


@dataclass
class TraceStep:
    """One safe replay step.

    ``reason`` is intentionally a public short summary, never private
    chain-of-thought.  ``observation`` and ``result`` are already sanitized by
    the runner/environment boundary.
    """

    step: int
    observation: Any = None
    action: Action | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    reason: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action is not None:
            self.action = Action.from_any(self.action)
            if self.tool is None and self.action is not None:
                self.tool = self.action.tool
            if not self.arguments and self.action is not None:
                self.arguments = dict(self.action.arguments)
            if self.reason is None and self.action is not None:
                self.reason = self.action.reason
        self.arguments = dict(self.arguments or {})
        if self.result is not None:
            self.result = ToolResult.from_any(self.result, tool=self.tool or "")
        try:
            self.latency_ms = max(0.0, float(self.latency_ms))
        except (TypeError, ValueError):
            self.latency_ms = 0.0
        self.metadata = dict(self.metadata or {})

    @property
    def observation_before(self) -> Any:
        return self.observation

    def to_dict(self) -> dict[str, Any]:
        obs = self.observation.to_dict() if hasattr(self.observation, "to_dict") else _jsonable(self.observation)
        return {
            "step": self.step,
            "observation": obs,
            "action": self.action.to_dict() if self.action else None,
            "tool": self.tool,
            "arguments": _jsonable(self.arguments),
            "result": self.result.to_dict() if self.result else None,
            "reason": self.reason,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class AgentRun:
    """Replayable run record shared by runner, evaluator and report writers."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = ""
    task_id: str = ""
    status: str = "running"
    success: bool | None = None
    steps: list[TraceStep] = field(default_factory=list)
    result: AgentResult | None = None
    started_at: float | None = None
    finished_at: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def trace(self) -> list[TraceStep]:
        return self.steps

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def finish(self, result: AgentResult | None = None, *, success: bool | None = None, status: str | None = None) -> None:
        self.result = result or self.result
        if self.result is not None:
            if success is None:
                success = self.result.success
            if status is None:
                status = self.result.status
        self.success = success
        self.status = status or ("success" if success else "failure" if success is False else "finished")
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "status": self.status,
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "result": self.result.to_dict() if self.result else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": _jsonable(self.metrics),
            "metadata": _jsonable(self.metadata),
        }


# Name used by some callers.
RunResult = AgentRun


class AgentProvider(ABC):
    """Stable provider contract implemented by every Arena agent.

    The methods are intentionally concrete and permissive so a minimal custom
    provider can override only what it needs.  ``observe`` stores the latest
    public observation; ``record_result`` lets a runner feed tool outcomes back
    without exposing hidden environment state.
    """

    provider_type: ClassVar[str] = "base"

    def __init__(self, *, name: str | None = None, seed: int | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.seed = seed
        self.rng = random.Random(seed)
        self.task: TaskDefinition | None = None
        self.observation: Observation = Observation()
        self.observation_history: list[Observation] = []
        self.plan_state: Plan = Plan()
        self.history: list[Action] = []
        self.results: list[ToolResult] = []
        self.finished: bool = False
        self._finish_result: AgentResult | None = None
        self.metrics: dict[str, Any] = {}

    def reset(self, task: TaskDefinition | Mapping[str, Any] | None = None, seed: int | None = None, **_: Any) -> None:
        """Reset all episode-local state.

        ``task`` is optional for compatibility with runners that reset first
        and attach a task later.  A supplied seed makes baseline runs fully
        reproducible.
        """

        if task is not None and not isinstance(task, TaskDefinition):
            task = TaskDefinition.from_dict(task)
        # Keep only the agent-visible task contract.  In particular, do not
        # retain evaluator-only hidden_ground_truth or success_conditions on a
        # provider object where a custom policy could accidentally inspect it.
        self.task = _sanitize_task(task)
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.observation = Observation()
        self.observation_history.clear()
        self.plan_state = Plan()
        self.history.clear()
        self.results.clear()
        self.finished = False
        self._finish_result = None
        self.metrics = {}

    def observe(self, observation: Observation | Mapping[str, Any] | Any | None = None, **_: Any) -> Observation:
        if observation is not None:
            self.observation = Observation.from_any(observation)
            self.observation_history.append(self.observation)
        return self.observation

    def plan(self, task: TaskDefinition | Mapping[str, Any] | None = None, observation: Observation | Mapping[str, Any] | None = None, **_: Any) -> Plan:
        if task is not None:
            candidate = task if isinstance(task, TaskDefinition) else TaskDefinition.from_dict(task)
            self.task = _sanitize_task(candidate)
        if observation is not None:
            self.observe(observation)
        self.plan_state = Plan(summary="Act from the current public observation.")
        return self.plan_state

    def act(self, observation: Observation | Mapping[str, Any] | None = None, **_: Any) -> Action | None:
        if observation is not None:
            self.observe(observation)
        if self.observation.done or self.finished:
            return None
        raise NotImplementedError(f"{self.__class__.__name__}.act() must be implemented")

    def record_result(self, result: ToolResult | Mapping[str, Any] | Any, *, action: Action | None = None, **_: Any) -> ToolResult:
        """Record a tool result and update the public observation boundary."""

        if action is not None and not isinstance(action, Action):
            action = Action.from_any(action)
        converted = ToolResult.from_any(result, tool=action.tool if action else "")
        self.results.append(converted)
        if action is not None and (not self.history or self.history[-1] is not action):
            self.history.append(action)
        self.observation.last_result = converted
        if action is not None:
            self.observation.last_action = action
        return converted

    # Common aliases used by adapters.
    on_result = record_result
    observe_result = record_result

    def finish(self, result: AgentResult | Mapping[str, Any] | Any | None = None, **_: Any) -> AgentResult:
        if isinstance(result, AgentResult):
            converted = result
        elif isinstance(result, Mapping):
            data = dict(result)
            converted = AgentResult(**{k: v for k, v in data.items() if k in _field_names(AgentResult)})
        elif result is None:
            converted = AgentResult(steps=len(self.history))
        else:
            converted = AgentResult(message=str(result), steps=len(self.history))
        if converted.steps == 0 and self.history:
            converted.steps = len(self.history)
        self.finished = True
        self._finish_result = converted
        self.metrics = dict(converted.metrics or {})
        return converted

    @property
    def last_action(self) -> Action | None:
        return self.history[-1] if self.history else None

    def _available_tools(self, observation: Observation | None = None) -> list[str]:
        obs = observation or self.observation
        names = _extract_tool_names(obs.available_tools)
        if not names and self.task is not None:
            names = list(self.task.allowed_tools)
        # Some simple environments expose tools in state metadata.
        if not names and isinstance(obs.state, Mapping):
            names = _extract_tool_names(obs.state.get("available_tools", obs.state.get("tools", [])))
        return _normalise_names(names)

    def _action(self, tool: str | None, arguments: Mapping[str, Any] | None = None, *, reason: str | None = None, action_type: str = "tool") -> Action:
        return Action(tool=tool, arguments=dict(arguments or {}), action_type=action_type, reason=reason)


class RandomAgent(AgentProvider):
    """Seeded random baseline with conservative argument generation."""

    provider_type = "random"

    def act(self, observation: Observation | Mapping[str, Any] | None = None, **kwargs: Any) -> Action | None:
        if observation is not None:
            self.observe(observation)
        if self.observation.done or self.finished:
            return None
        tools = self._available_tools()
        if not tools:
            # Grid environments may accept a native wait action without tools.
            return self._action(None, {}, action_type="wait", reason="No public tool available; wait.")
        tool = self.rng.choice(tools)
        args = _default_arguments(tool, self.observation.state, self.rng)
        return self._action(tool, args, reason=f"Random baseline selected {tool}.")


class RuleBasedAgent(AgentProvider):
    """Small deterministic heuristic baseline.

    The policy deliberately uses only public observations and tool names.  It
    is not intended to be optimal; its value is as a transparent reference
    against which more capable providers can be compared.
    """

    provider_type = "rule_based"

    _priority: ClassVar[tuple[str, ...]] = (
        "list_files",
        "search_mail",
        "list_mail",
        "calendar_lookup",
        "calendar_find_conflicts",
        "browser_open",
        "read_file",
        "read_mail",
        "calculator",
        "execute_mock_code",
        "run_mock_tests",
        "sandbox_list_files",
        "sandbox_read_file",
        "sandbox_write_file",
        "move_file",
        "write_file",
        "grid_move",
        "grid_turn",
        "grid_pick_up",
        "grid_use",
        "grid_interact",
        "grid_wait",
        "interact",
        "pick_up",
        "use",
        "drop",
        "wait",
    )

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._used: set[str] = set()
        self._last_failed = False
        self._last_result: ToolResult | None = None
        self._file_queue: list[tuple[str, str]] = []
        self._category_complete = False
        self._empty_retries: set[str] = set()

    def observe(self, observation: Observation | Mapping[str, Any] | Any | None = None, **kwargs: Any) -> Observation:
        obs = super().observe(observation, **kwargs)
        # A runner may choose to include the most recent result in the next
        # observation instead of calling record_result directly.  Consume that
        # public result here so the baseline can recover in either integration
        # style.
        if obs.last_result is not None:
            self._last_result = obs.last_result
            self._last_failed = not obs.last_result.success
            if obs.last_action is not None and obs.last_action.tool:
                self._used.add(obs.last_action.tool)
        return obs

    def plan(self, task: TaskDefinition | Mapping[str, Any] | None = None, observation: Observation | Mapping[str, Any] | None = None, **kwargs: Any) -> Plan:
        super().plan(task, observation, **kwargs)
        tools = self._available_tools()
        ordered = [t for t in self._priority if t in tools] + [t for t in tools if t not in self._priority]
        self.plan_state = Plan(
            actions=[self._action(t, self._rule_arguments(t), reason=f"Rule priority: {t}.") for t in ordered],
            summary="Use the most informative available tool, then complete the task.",
        )
        return self.plan_state

    def record_result(self, result: ToolResult | Mapping[str, Any] | Any, *, action: Action | None = None, **kwargs: Any) -> ToolResult:
        converted = super().record_result(result, action=action, **kwargs)
        if action and action.tool:
            self._used.add(action.tool)
        self._last_failed = not converted.success
        self._last_result = converted
        return converted

    def act(self, observation: Observation | Mapping[str, Any] | None = None, **kwargs: Any) -> Action | None:
        if observation is not None:
            self.observe(observation)
        if self.observation.done or self.finished:
            return None
        handled, action = self._category_action()
        if handled:
            return action
        tools = self._available_tools()
        if not tools:
            return self._action(None, {}, action_type="wait", reason="No public tool available; wait.")

        state = self.observation.state
        # Prefer recovery after a tool failure: retry only once, then select a
        # different informative tool if one exists.
        if self._last_failed and self.last_action and self.last_action.tool in tools:
            failed_tool = self.last_action.tool
            alternatives = [t for t in tools if t != failed_tool and t not in self._used]
            if alternatives:
                tool = self._choose_tool(alternatives)
            else:
                tool = failed_tool
        else:
            unused = [t for t in tools if t not in self._used]
            tool = self._choose_tool(unused or tools)
        args = self._rule_arguments(tool)
        return self._action(tool, args, reason=f"Rule-based policy selected {tool} from public state.")

    def _choose_tool(self, tools: Sequence[str]) -> str:
        rank = {name: i for i, name in enumerate(self._priority)}
        return min(tools, key=lambda name: (rank.get(name, len(rank)), name))

    def _category_action(self) -> tuple[bool, Action | None]:
        """Run concise workflows for the built-in safe environments.

        Returning ``(False, None)`` delegates to the generic transparent
        priority policy.  Returning ``(True, None)`` asks the runner to finish
        the episode after a complete public workflow.
        """

        tools = set(self._available_tools())
        state = self.observation.state if isinstance(self.observation.state, Mapping) else {}
        environment = str(state.get("environment", "")).lower()
        category = str(getattr(self.task, "category", "")).lower() if self.task else ""
        tags = {str(tag).lower() for tag in (getattr(self.task, "tags", []) or [])} if self.task else set()
        last_tool = self._last_result.tool if self._last_result else None
        last_output = self._last_result.output if self._last_result else None

        # File organization: inspect /inbox, then move every visible file to
        # the instruction-derived archive/notes destination.  This never
        # reads ground truth or touches a host filesystem.
        if environment == "virtual_file_system" or category in {"files", "file_organization"} or "files" in tags:
            if "list_files" not in tools or "move_file" not in tools:
                return False, None
            if self._last_failed:
                if self.last_action and self.last_action.tool:
                    return True, self._action(self.last_action.tool, self.last_action.arguments, reason="Retry the failed virtual file operation once.")
            if last_tool == "list_files" and isinstance(last_output, Mapping):
                if self._last_result and self._last_result.metadata.get("adversarial") == "empty_result" and "list_files" not in self._empty_retries:
                    self._empty_retries.add("list_files")
                    return True, self._action("list_files", {"path": "/inbox"}, reason="Retry the empty virtual listing before deciding it is complete.")
                entries = last_output.get("entries", [])
                files = [entry.get("path") for entry in entries if isinstance(entry, Mapping) and entry.get("type") == "file" and str(entry.get("path", "")).startswith("/inbox/")]
                if files:
                    self._file_queue = [(str(path), _organization_destination(str(path))) for path in files]
                elif self.last_action and self.last_action.arguments.get("path") == "/inbox":
                    self._category_complete = True
            if last_tool == "move_file" and isinstance(last_output, Mapping):
                moved_source = str(last_output.get("source", ""))
                self._file_queue = [item for item in self._file_queue if item[0] != moved_source]
            if self._file_queue:
                source, destination = self._file_queue[0]
                return True, self._action("move_file", {"source": source, "destination": destination}, reason="Move the next inbox file to its requested virtual folder.")
            if self._category_complete:
                return True, None
            return True, self._action("list_files", {"path": "/inbox"}, reason="Inspect the virtual inbox before organizing it.")

        # Email retrieval: search the public inbox for the security message,
        # read the matching message, then finish.  The trace contains the
        # public tool result for a report/reviewer without private reasoning.
        if environment == "virtual_email_inbox" or category in {"email", "email_retrieval"} or "email" in tags:
            if "search_mail" not in tools or "read_mail" not in tools:
                return False, None
            if self._last_failed and self.last_action and self.last_action.tool:
                return True, self._action(self.last_action.tool, self.last_action.arguments, reason="Retry the failed simulated inbox operation once.")
            if last_tool in {"search_mail", "list_mail"}:
                if self._last_result and self._last_result.metadata.get("adversarial") == "empty_result" and "search_mail" not in self._empty_retries:
                    self._empty_retries.add("search_mail")
                    return True, self._action("search_mail", {"query": "security"}, reason="Retry the empty simulated search result.")
                message_id = _find_message_id(last_output)
                if message_id:
                    return True, self._action("read_mail", {"message_id": message_id, "mark_read": True}, reason="Read the matching security message.")
            if last_tool == "read_mail":
                return True, None
            return True, self._action("search_mail", {"query": "security"}, reason="Find the security message in the simulated inbox.")

        # Calendar scheduling: inspect the requested day, create the meeting
        # using only date/time text from the instruction, then finish.
        if environment == "virtual_calendar" or category in {"calendar", "calendar_scheduling"} or "calendar" in tags:
            if "calendar_lookup" not in tools or "calendar_create" not in tools:
                return False, None
            if self._last_failed and self.last_action and self.last_action.tool:
                return True, self._action(self.last_action.tool, self.last_action.arguments, reason="Retry the failed simulated calendar operation once.")
            if last_tool == "calendar_lookup":
                return True, self._action("calendar_create", self._rule_arguments("calendar_create"), reason="Create the requested meeting after checking the calendar.")
            if last_tool == "calendar_create":
                return True, None
            return True, self._action("calendar_lookup", self._rule_arguments("calendar_lookup"), reason="Check the requested calendar window first.")

        # Code repair mock: read the virtual file, apply the small known-safe
        # correction, and submit it to the non-executing code checker.
        if environment == "code_sandbox_mock" or category in {"code", "code_repair"} or "code" in tags:
            read_tool = "sandbox_read_file" if "sandbox_read_file" in tools else "read_file" if "read_file" in tools else None
            write_tool = "sandbox_write_file" if "sandbox_write_file" in tools else "write_file" if "write_file" in tools else None
            execute_tool = "execute_mock_code" if "execute_mock_code" in tools else "run_mock_tests" if "run_mock_tests" in tools else None
            if not (read_tool and write_tool and execute_tool):
                return False, None
            if self._last_failed and self.last_action and self.last_action.tool:
                return True, self._action(self.last_action.tool, self.last_action.arguments, reason="Retry the failed mock code operation once.")
            if last_tool == read_tool:
                path = _first_identifier(last_output, ("path",), nested_keys=("files",)) or "/workspace/main.py"
                repaired = _repair_code(_first_content(last_output), self.task.instruction if self.task else "")
                return True, self._action(write_tool, {"path": path, "content": repaired}, reason="Write the conservative code correction to the virtual file.")
            if last_tool == write_tool:
                path = _first_identifier(last_output, ("path",), nested_keys=("files",)) or "/workspace/main.py"
                return True, self._action(execute_tool, {"path": path}, reason="Run the non-executing mock checker on the repaired file.")
            if last_tool == execute_tool:
                if self._last_result and self._last_result.metadata.get("adversarial") == "empty_result" and "execute_mock_code" not in self._empty_retries:
                    self._empty_retries.add("execute_mock_code")
                    path = _first_identifier(last_output, ("path",), nested_keys=("files",)) or "/workspace/main.py"
                    return True, self._action(execute_tool, {"path": path}, reason="Retry the empty mock execution result before finishing.")
                return True, None
            return True, self._action(read_tool, {"path": "/workspace/main.py"}, reason="Inspect the virtual code file before repairing it.")

        # Grid world: use only the visible cells.  On the simple baseline map
        # this reaches the visible goal; on harder maps it remains a valid
        # partial-observation navigator rather than reading the hidden grid.
        if environment == "simple_grid_world" or category in {"grid", "grid_navigation"} or "grid" in tags:
            if "grid_move" not in tools:
                return False, None
            if bool(state.get("done")) or bool(state.get("goal_reached")):
                return True, None
            return True, self._action("grid_move", {"direction": _grid_direction(state)}, reason="Move toward an observed safe route or visible goal.")

        return False, None

    def _rule_arguments(self, tool: str) -> dict[str, Any]:
        """Derive conservative arguments from public observation/results.

        The heuristic intentionally does not inspect hidden ground truth.  It
        handles the built-in simulated environments well enough to provide a
        meaningful transparent baseline while remaining safe for arbitrary
        custom tools.
        """

        name = str(tool).lower()
        state = self.observation.state
        result = self._last_result.output if self._last_result is not None else None
        if name == "list_files":
            return {"path": "/"}
        if name in {"sandbox_list_files", "list_mail", "calendar_lookup"}:
            if name == "calendar_lookup":
                date = _extract_date(self.task.instruction if self.task else "")
                return {"date": date} if date else {}
            return {}
        if name in {"search_mail", "search_files", "browser_search", "search"}:
            text = (self.task.instruction if self.task else "").lower()
            query = "security" if "security" in text else "important" if "important" in text else ""
            if isinstance(state, Mapping):
                query = str(state.get("query", state.get("keyword", query)))
            return {"query": query}
        if name in {"read_file", "sandbox_read_file", "open_file"}:
            path = _first_identifier(result, ("path",), nested_keys=("entries", "files", "matches"))
            if path is None:
                path = _first_identifier(state, ("path",), nested_keys=("root_entries", "entries", "files"))
            if path is None:
                path = "/workspace/main.py" if "code" in (self.task.instruction.lower() if self.task else "") else ""
            return {"path": path}
        if name in {"read_mail", "mark_mail_read"}:
            message_id = _first_identifier(result, ("id", "message_id"), nested_keys=("messages",))
            if message_id is None:
                message_id = _first_identifier(state, ("id", "message_id"), nested_keys=("messages",))
            return {"message_id": message_id or "", **({"mark_read": True} if name == "read_mail" else {})}
        if name in {"move_file", "copy_file"}:
            source = _first_identifier(result, ("path", "source"), nested_keys=("entries", "files", "matches"))
            if source is None:
                source = _first_identifier(state, ("path", "source"), nested_keys=("root_entries", "entries", "files"))
            source = str(source or "")
            basename = source.rsplit("/", 1)[-1]
            destination = "/archive/" + basename if any(token in basename.lower() for token in ("report", "invoice")) else "/notes/" + basename
            return {"source": source, "destination": destination}
        if name in {"calendar_find_conflicts"}:
            text = self.task.instruction if self.task else ""
            date = _extract_date(text)
            times = _extract_times(text)
            return {"start": f"{date}T{times[0]}" if date and times else None, "end": f"{date}T{times[1]}" if date and len(times) > 1 else None}
        if name in {"calendar_create"}:
            text = self.task.instruction if self.task else ""
            date = _extract_date(text) or "2026-01-01"
            times = _extract_times(text) or ("10:00", "11:00")
            title = "Synthetic planning meeting"
            if "schedule a " in text.lower() and " on " in text.lower():
                title = text.lower().split("schedule a ", 1)[1].split(" on ", 1)[0].strip().capitalize()
            return {"title": title, "start": f"{date}T{times[0]}", "end": f"{date}T{times[1]}"}
        if name in {"write_file", "sandbox_write_file"}:
            path = _first_identifier(result, ("path",), nested_keys=("entries", "files")) or "/workspace/main.py"
            content = _repair_code(_first_content(result), self.task.instruction if self.task else "")
            return {"path": path, "content": content}
        if name in {"execute_mock_code", "execute_code", "run_code", "run_mock_tests"}:
            path = _first_identifier(result, ("path",), nested_keys=("entries", "files"))
            if name == "run_mock_tests":
                return {"path": path} if path else {}
            content = _first_content(result)
            if content:
                return {"code": content}
            return {"path": path} if path else {"code": ""}
        if name in {"grid_move", "move", "navigate", "walk"}:
            direction = _grid_direction(state)
            return {"direction": direction}
        if name in {"grid_turn", "turn"}:
            return {"direction": "E"}
        if name in {"grid_pick_up", "pick_up", "pickup", "take", "grid_use", "use", "grid_interact", "interact", "grid_drop", "drop"}:
            object_id = _first_identifier(state, ("target", "object_id", "object"), nested_keys=("visible_objects", "objects"))
            key = "object_id" if name.startswith("grid_") else "object"
            return {key: object_id} if object_id else {}
        if name in {"grid_wait", "wait"}:
            return {"ticks": 1} if name == "grid_wait" else {}
        return _default_arguments(tool, state, self.rng)


class ScriptedAgent(AgentProvider):
    """Replay a fixed list of actions or action-producing callables."""

    provider_type = "scripted"

    def __init__(self, script: Iterable[Any] | Callable[..., Any] | None = None, *, name: str | None = None, seed: int | None = None, stop_on_error: bool = False, retry_on_error: bool = True) -> None:
        super().__init__(name=name, seed=seed)
        self.script_source = script
        self.stop_on_error = stop_on_error
        self.retry_on_error = bool(retry_on_error)
        self._script: list[Any] = []
        self._cursor = 0
        self._current_index: int | None = None
        self._retried_indices: set[int] = set()
        self._script_callable: Callable[..., Any] | None = script if callable(script) else None
        if script is not None and not callable(script):
            self._script = list(script)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._cursor = 0
        self._current_index = None
        self._retried_indices.clear()

    def plan(self, task: TaskDefinition | Mapping[str, Any] | None = None, observation: Observation | Mapping[str, Any] | None = None, **kwargs: Any) -> Plan:
        super().plan(task, observation, **kwargs)
        if self._script_callable:
            self.plan_state = Plan(summary="Execute the configured action script.")
        else:
            remaining = [Action.from_any(item) for item in self._script[self._cursor :]]
            self.plan_state = Plan(actions=[a for a in remaining if a is not None], summary="Execute the configured action script.")
        return self.plan_state

    def act(self, observation: Observation | Mapping[str, Any] | None = None, **kwargs: Any) -> Action | None:
        if observation is not None:
            self.observe(observation)
        if self.observation.done or self.finished:
            return None
        if self._script_callable is not None:
            self._current_index = self._cursor
            value = _call_policy(self._script_callable, self.observation, self.task, self._cursor)
            self._cursor += 1
            return Action.from_any(value)
        if self._cursor >= len(self._script):
            return None
        self._current_index = self._cursor
        value = self._script[self._cursor]
        self._cursor += 1
        if callable(value):
            value = _call_policy(value, self.observation, self.task, self._cursor - 1)
        return Action.from_any(value)

    def record_result(self, result: ToolResult | Mapping[str, Any] | Any, *, action: Action | None = None, **kwargs: Any) -> ToolResult:
        converted = super().record_result(result, action=action, **kwargs)
        transient = (not converted.success) or converted.metadata.get("adversarial") in {"failed_tool", "empty_result"}
        if transient and self.retry_on_error and self._current_index is not None and self._current_index not in self._retried_indices:
            self._retried_indices.add(self._current_index)
            self._cursor = min(self._cursor, self._current_index)
        if self.stop_on_error and not converted.success:
            self._cursor = len(self._script)
        return converted


class MockLLMAgent(RuleBasedAgent):
    """Offline deterministic stand-in for a language-model provider.

    ``responses`` may be a sequence or a callable.  If it does not provide an
    action, the transparent rule-based fallback is used.  Token estimates are
    rough and are explicitly marked as estimates in metrics; no network/API
    call occurs.
    """

    provider_type = "mock_llm"

    def __init__(self, responses: Iterable[Any] | Callable[..., Any] | None = None, *, name: str | None = None, seed: int | None = 0, model_name: str = "mock-llm-v1") -> None:
        super().__init__(name=name or "MockLLMAgent", seed=seed)
        self.model_name = model_name
        self.model = model_name
        self.responses_source = responses
        self._responses = list(responses) if responses is not None and not callable(responses) else []
        self._response_cursor = 0
        self._response_callable = responses if callable(responses) else None
        self.prompt_tokens_estimate = 0
        self.completion_tokens_estimate = 0

    @property
    def token_usage_estimate(self) -> dict[str, Any]:
        total = self.prompt_tokens_estimate + self.completion_tokens_estimate
        return {
            "prompt_tokens": self.prompt_tokens_estimate,
            "completion_tokens": self.completion_tokens_estimate,
            "total_tokens": total,
            "estimated": True,
        }

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._response_cursor = 0
        self.prompt_tokens_estimate = 0
        self.completion_tokens_estimate = 0

    def _next_response(self) -> Any:
        if self._response_callable is not None:
            value = _call_policy(self._response_callable, self.observation, self.task, self._response_cursor)
            self._response_cursor += 1
            return value
        if self._response_cursor >= len(self._responses):
            return None
        value = self._responses[self._response_cursor]
        self._response_cursor += 1
        return value

    def act(self, observation: Observation | Mapping[str, Any] | None = None, **kwargs: Any) -> Action | None:
        if observation is not None:
            self.observe(observation)
        # Estimate usage without storing private prompt content.
        self.prompt_tokens_estimate += max(1, len(str(self.observation.state)) // 4)
        value = self._next_response()
        action = Action.from_any(value) if value is not None else None
        if action is None:
            action = super().act()
        if action is not None:
            self.completion_tokens_estimate += max(1, len(str(action.to_dict())) // 4)
            if not action.reason:
                action.reason = "Mock model selected an action from the public observation."
        return action

    def finish(self, result: AgentResult | Mapping[str, Any] | Any | None = None, **kwargs: Any) -> AgentResult:
        finished = super().finish(result, **kwargs)
        finished.metrics.setdefault("model", self.model_name)
        finished.metrics.setdefault("token_usage_estimate", self.token_usage_estimate)
        finished.metrics.setdefault("token_usage", self.token_usage_estimate["total_tokens"])
        # The offline provider has no billable API calls; keep an explicit
        # zero-valued estimate so reports never confuse missing data with cost.
        finished.metrics.setdefault("estimated_cost", 0.0)
        self.metrics = dict(finished.metrics)
        return finished


class DeferredProvider(AgentProvider):
    """Safe placeholder for a future external/local model adapter.

    Construction is allowed so benchmark configuration can mention a provider,
    but acting is intentionally disabled until an explicit adapter is supplied.
    No SDK, network client, credential, or paid API is touched by this class.
    """

    provider_type = "deferred"

    def __init__(self, *, provider_name: str = "deferred", enabled: bool = False, **kwargs: Any) -> None:
        super().__init__(name=kwargs.pop("name", provider_name), seed=kwargs.pop("seed", None))
        self.provider_name = provider_name
        self.enabled = bool(enabled)
        self.config = dict(kwargs)

    def act(self, observation: Observation | Mapping[str, Any] | None = None, **kwargs: Any) -> Action | None:
        if observation is not None:
            self.observe(observation)
        if not self.enabled:
            raise RuntimeError(
                f"Provider '{self.provider_name}' is a reserved extension point and is disabled in offline mode."
            )
        raise NotImplementedError(f"No adapter implementation registered for provider '{self.provider_name}'.")


class OpenAIProvider(DeferredProvider):
    provider_type = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider_name="openai", **kwargs)


class AnthropicProvider(DeferredProvider):
    provider_type = "anthropic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider_name="anthropic", **kwargs)


class GoogleProvider(DeferredProvider):
    provider_type = "google"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider_name="google", **kwargs)


class LocalModelProvider(DeferredProvider):
    provider_type = "local_model"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider_name="local_model", **kwargs)


class CustomHTTPProvider(DeferredProvider):
    provider_type = "custom_http"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(provider_name="custom_http", **kwargs)


def _sanitize_task(task: TaskDefinition | None) -> TaskDefinition | None:
    """Return an agent-safe task view without evaluator-only fields."""

    if task is None:
        return None
    try:
        data = task.to_dict()
    except Exception:
        data = {
            "task_id": getattr(task, "task_id", getattr(task, "id", "task")),
            "instruction": getattr(task, "instruction", ""),
            "initial_state": getattr(task, "initial_state", {}),
            "allowed_tools": getattr(task, "allowed_tools", []),
            "timeout": getattr(task, "timeout", 30.0),
            "max_steps": getattr(task, "max_steps", 20),
            "difficulty": getattr(task, "difficulty", "easy"),
            "tags": getattr(task, "tags", []),
            "category": getattr(task, "category", "generic"),
            "seed": getattr(task, "seed", None),
            "metadata": getattr(task, "metadata", {}),
        }
    data = dict(data)
    data["hidden_ground_truth"] = {}
    data["success_conditions"] = {}
    # ``TaskDefinition`` from tasking accepts these fields; custom standalone
    # embeddings may not, so fall back to a minimal mapping if necessary.
    try:
        return TaskDefinition.from_dict(data)
    except Exception:
        try:
            return TaskDefinition(
                task_id=str(data.get("task_id", "task")),
                instruction=str(data.get("instruction", "")),
                initial_state=dict(data.get("initial_state", {}) or {}),
                allowed_tools=list(data.get("allowed_tools", []) or []),
                timeout=data.get("timeout", 30.0),
                max_steps=data.get("max_steps", 20),
                difficulty=data.get("difficulty", "easy"),
                tags=list(data.get("tags", []) or []),
                metadata=dict(data.get("metadata", {}) or {}),
            )
        except Exception:
            return task


def _field_names(cls: type[Any]) -> set[str]:
    return {f.name for f in cls.__dataclass_fields__.values()} if hasattr(cls, "__dataclass_fields__") else set()


def _normalise_names(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    result: list[str] = []
    try:
        iterator = iter(values)
    except TypeError:
        return [str(values)]
    for value in iterator:
        if isinstance(value, Mapping):
            name = value.get("name", value.get("tool", value.get("id")))
            if name is not None:
                result.append(str(name))
        elif hasattr(value, "name"):
            name = getattr(value, "name", None)
            if name is not None:
                result.append(str(name))
        elif value is not None:
            result.append(str(value))
    return list(dict.fromkeys(result))


def _extract_tool_names(values: Any) -> list[str]:
    return _normalise_names(values)


def _state_lookup(state: Any, *keys: str) -> Any:
    if not isinstance(state, Mapping):
        return None
    for key in keys:
        if key in state:
            return state[key]
    return None


def _first_content(value: Any) -> str | None:
    if isinstance(value, Mapping):
        content = value.get("content", value.get("body", value.get("code")))
        if content is not None:
            return str(content)
        for nested in ("result", "output", "file"):
            if nested in value:
                found = _first_content(value[nested])
                if found:
                    return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found = _first_content(item)
            if found:
                return found
    elif isinstance(value, str) and value.strip():
        return value
    return None


def _organization_destination(path: str) -> str:
    basename = str(path).rsplit("/", 1)[-1]
    folder = "/archive" if any(token in basename.lower() for token in ("report", "invoice")) else "/notes"
    return f"{folder}/{basename}"


def _find_message_id(value: Any) -> str | None:
    """Prefer an important/security message over an arbitrary inbox item."""

    candidates: list[Mapping[str, Any]] = []

    def collect(node: Any) -> None:
        if isinstance(node, Mapping):
            if node.get("id") is not None and any(key in node for key in ("subject", "labels", "from", "body")):
                candidates.append(node)
            for child in node.values():
                if isinstance(child, (Mapping, list, tuple)):
                    collect(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                collect(child)

    collect(value)
    if not candidates:
        return _first_identifier(value, ("id", "message_id"), nested_keys=("messages",))
    ranked = sorted(
        candidates,
        key=lambda item: (
            not any("important" in str(label).lower() for label in item.get("labels", []) or []),
            "security" not in str(item.get("subject", "")).lower() and "security" not in str(item.get("body", "")).lower(),
        ),
    )
    return str(ranked[0].get("id")) if ranked[0].get("id") is not None else None


def _extract_date(text: str) -> str | None:
    import re

    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", str(text))
    return match.group(1) if match else None


def _extract_times(text: str) -> tuple[str, ...]:
    import re

    # ISO timestamps commonly appear as ``T11:00``; a word boundary would not
    # match between the word characters ``T`` and ``1``.  A digit look-behind
    # correctly handles both ISO and plain ``from 11:00`` forms.
    matches = re.findall(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)", str(text))
    return tuple(f"{hour.zfill(2)}:{minute}" for hour, minute in matches)


def _repair_code(content: str | None, instruction: str) -> str:
    """Apply a tiny offline repair heuristic for generated code tasks."""

    source = content or ""
    if "add" in source and "return a - b" in source:
        return source.replace("return a - b", "return a + b")
    if "is_even" in source and "return n % 2 == 1" in source:
        return source.replace("return n % 2 == 1", "return n % 2 == 0")
    if "clamp" in source and "max(hi, min(lo, x))" in source:
        return source.replace("max(hi, min(lo, x))", "max(lo, min(hi, x))")
    return source


def _grid_direction(state: Any) -> str:
    """Choose a visible, non-wall direction toward a visible goal."""

    if not isinstance(state, Mapping):
        return "E"
    position = state.get("position")
    cells = state.get("visible_cells", [])
    if not isinstance(position, Mapping):
        return "E"
    px, py = int(position.get("x", 0)), int(position.get("y", 0))
    goals = [c for c in cells if isinstance(c, Mapping) and c.get("terrain") in {"goal", "G"}]
    candidates: list[tuple[int, str]] = []
    if goals:
        gx, gy = int(goals[0].get("x", px)), int(goals[0].get("y", py))
        if gx > px:
            candidates.append((abs(gx - px), "E"))
        if gx < px:
            candidates.append((abs(gx - px), "W"))
        if gy > py:
            candidates.append((abs(gy - py), "S"))
        if gy < py:
            candidates.append((abs(gy - py), "N"))
    # Prefer a direction with an observed open neighboring cell.  The tie
    # order makes the baseline deterministic across Python versions.
    open_cells = {(int(c.get("x")), int(c.get("y"))): c for c in cells if isinstance(c, Mapping)}
    for _, direction in sorted(candidates):
        dx, dy = {"E": (1, 0), "W": (-1, 0), "S": (0, 1), "N": (0, -1)}[direction]
        cell = open_cells.get((px + dx, py + dy))
        if cell is None or cell.get("terrain") != "wall":
            return direction
    for direction, (dx, dy) in (("E", (1, 0)), ("S", (0, 1)), ("W", (-1, 0)), ("N", (0, -1))):
        cell = open_cells.get((px + dx, py + dy))
        if cell is not None and cell.get("terrain") != "wall":
            return direction
    return "E"


def _default_arguments(tool: str, state: Any, rng: random.Random | None = None) -> dict[str, Any]:
    """Generate harmless, observation-derived defaults for baseline agents."""

    name = str(tool).lower()
    rng = rng or random.Random(0)
    if name in {"list_files", "calendar_lookup", "list_events", "inbox", "list_mail"}:
        return {}
    if name in {"search_mail", "search_files", "search", "browser_open"}:
        query = _state_lookup(state, "query", "keyword", "search", "target")
        return {"query": str(query)} if query is not None else {"query": ""}
    if name in {"read_file", "read_mail", "open_file"}:
        items = _state_lookup(state, "files", "messages", "emails", "items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)) and items:
            first = items[0]
            if isinstance(first, Mapping):
                ident = first.get("path", first.get("id", first.get("subject")))
            else:
                ident = first
            if ident is not None:
                return {"path" if "file" in name else "id": ident}
        return {"path" if "file" in name else "id": ""}
    if name in {"move_file", "copy_file"}:
        return {"source": "", "destination": ""}
    if name in {"execute_mock_code", "execute_code", "run_code"}:
        code = _state_lookup(state, "code", "snippet", "program")
        return {"code": str(code or "")}
    if name in {"calculator", "calculate"}:
        expression = _state_lookup(state, "expression", "formula", "calculation")
        return {"expression": str(expression or "0")}
    if name in {"move", "navigate", "walk"}:
        direction = _state_lookup(state, "suggested_direction", "direction") or rng.choice(["up", "down", "left", "right"])
        return {"direction": direction}
    if name in {"pick_up", "pickup", "take"}:
        target = _state_lookup(state, "target", "object", "item") or ""
        return {"object": target}
    if name in {"drop", "use", "interact"}:
        target = _state_lookup(state, "target", "object", "item") or ""
        return {"object": target}
    return {}


def _call_policy(policy: Callable[..., Any], observation: Observation, task: TaskDefinition | None, index: int) -> Any:
    """Call a user policy with the richest supported signature."""

    try:
        signature = inspect.signature(policy)
        positional = [p for p in signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if any(p.kind == p.VAR_POSITIONAL for p in signature.parameters.values()) or len(positional) >= 3:
            return policy(observation, task, index)
        if len(positional) == 2:
            return policy(observation, task)
        if len(positional) == 1:
            return policy(observation)
        return policy()
    except (TypeError, ValueError):
        # Builtins and some callable objects do not expose signatures.
        for args in ((observation, task, index), (observation, task), (observation,), ()):
            try:
                return policy(*args)
            except TypeError:
                continue
        return None


# Prefer the canonical task/trace contracts when the rest of the package has
# provided them.  This module remains importable on its own (the fallbacks
# above are useful for a single-file embedding), while package users get one
# shared TaskDefinition/TraceStep/RunResult type across runner and agents.
try:  # pragma: no cover - import order depends on the embedding application
    from .tasking import TaskDefinition as _CanonicalTaskDefinition

    TaskDefinition = _CanonicalTaskDefinition  # type: ignore[assignment,misc]
    if not hasattr(TaskDefinition, "public_dict"):
        def _canonical_public_dict(self: Any) -> dict[str, Any]:
            data = dict(self.to_dict())
            # Initial state may contain a complete map or all inbox contents;
            # agents should receive it through partial observations only.
            data.pop("initial_state", None)
            data.pop("hidden_ground_truth", None)
            data.pop("success_conditions", None)
            return data

        setattr(TaskDefinition, "public_dict", _canonical_public_dict)
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - trace.py may be loaded after this module in embeds
    from .trace import AgentTrace as _CanonicalAgentTrace
    from .trace import RunResult as _CanonicalRunResult
    from .trace import TraceStep as _CanonicalTraceStep

    TraceStep = _CanonicalTraceStep  # type: ignore[assignment,misc]
    RunResult = _CanonicalRunResult  # type: ignore[assignment,misc]
    AgentTrace = _CanonicalAgentTrace
    AgentRun = _CanonicalAgentTrace
except Exception:  # pragma: no cover
    AgentTrace = AgentRun


# Compatibility aliases used in examples and older drafts.
BaseAgent = AgentProvider
RandomProvider = RandomAgent
RuleAgent = RuleBasedAgent
MockAgent = MockLLMAgent
OpenAI = OpenAIProvider
Anthropic = AnthropicProvider
Google = GoogleProvider
LocalModel = LocalModelProvider
CustomHTTP = CustomHTTPProvider
Task = TaskDefinition
ActionRequest = Action
ObservationState = Observation
ToolCallResult = ToolResult


__all__ = [
    "Action",
    "ActionRequest",
    "AgentProvider",
    "AgentResult",
    "AgentTrace",
    "AgentRun",
    "AnthropicProvider",
    "Anthropic",
    "BaseAgent",
    "MockAgent",
    "MockLLMAgent",
    "CustomHTTPProvider",
    "CustomHTTP",
    "DeferredProvider",
    "GoogleProvider",
    "Google",
    "LocalModelProvider",
    "LocalModel",
    "OpenAIProvider",
    "OpenAI",
    "Observation",
    "ObservationState",
    "Plan",
    "RandomAgent",
    "RandomProvider",
    "RuleAgent",
    "RuleBasedAgent",
    "RunResult",
    "ScriptedAgent",
    "Task",
    "TaskDefinition",
    "ToolCall",
    "ToolCallResult",
    "ToolResult",
    "TraceStep",
]
