"""Execution trace models for Agent Arena.

The trace is deliberately a boring, JSON-serialisable data structure.  It is
the contract between the runner, evaluator, replay viewer and report writer.
Only a *public* reason summary is recorded; agents must never put private
chain-of-thought in a trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert common model objects to JSON-compatible values.

    Trace recording must not fail because a provider returned a small custom
    object.  We intentionally keep this conversion conservative: unknown
    objects become a useful ``repr`` rather than leaking private state or
    raising while handling an earlier failure.
    """

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    if is_dataclass(value):
        try:
            return json_safe(asdict(value))
        except Exception:
            pass
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float)):
        return getattr(value, "value")
    # Avoid serialising arbitrary provider internals.  repr is useful in a
    # replay, while still making the boundary explicit.
    return repr(value)


@dataclass
class TraceStep:
    """One agent/environment interaction.

    ``step`` starts at one.  ``observation`` is the observation *before* the
    action and ``result`` is the tool/environment response.  ``state_snapshot``
    is optional and should contain only public state suitable for replay.
    """

    step: int
    observation: Any = None
    action: Any = None
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    public_reason_summary: Optional[str] = None
    error: Optional[str] = None
    latency: float = 0.0
    timestamp: Optional[str] = None
    state_snapshot: Any = None
    valid: bool = True
    action_type: Optional[str] = None
    tokens: Optional[int] = None

    @property
    def reason(self) -> Optional[str]:
        """Compatibility alias for clients that call the field ``reason``."""

        return self.public_reason_summary

    @reason.setter
    def reason(self, value: Optional[str]) -> None:
        self.public_reason_summary = value

    @property
    def duration(self) -> float:
        return self.latency

    @property
    def latency_ms(self) -> float:
        """Compatibility view used by the replay/reporting modules."""

        return float(self.latency) * 1000.0

    @latency_ms.setter
    def latency_ms(self, value: float) -> None:
        try:
            self.latency = max(0.0, float(value)) / 1000.0
        except (TypeError, ValueError):
            self.latency = 0.0

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        # A public reason is explicitly bounded to discourage accidental
        # dumping of a provider's private reasoning into traces.
        if raw.get("public_reason_summary") is not None:
            raw["public_reason_summary"] = str(raw["public_reason_summary"])[:2000]
        return json_safe(raw)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceStep":
        known = {
            "step", "observation", "action", "tool", "arguments", "result",
            "public_reason_summary", "error", "latency", "timestamp",
            "state_snapshot", "valid", "action_type", "tokens",
        }
        payload = {k: data.get(k) for k in known if k in data}
        if "public_reason_summary" not in payload and "reason" in data:
            payload["public_reason_summary"] = data.get("reason")
        if "latency" not in payload and "latency_ms" in data:
            try:
                payload["latency"] = float(data.get("latency_ms") or 0.0) / 1000.0
            except (TypeError, ValueError):
                payload["latency"] = 0.0
        payload.setdefault("step", 0)
        payload.setdefault("arguments", {})
        payload.setdefault("latency", 0.0)
        payload.setdefault("valid", True)
        return cls(**payload)


@dataclass
class AgentTrace:
    """Complete trace for one task/agent run."""

    run_id: str
    task_id: str
    agent_id: str
    seed: Optional[int] = None
    started_at: str = field(default_factory=utc_now)
    finished_at: Optional[str] = None
    status: str = "running"
    steps: List[TraceStep] = field(default_factory=list)
    final_observation: Any = None
    final_state: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def events(self) -> List[TraceStep]:
        """Alias used by replay clients."""

        return self.steps

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_latency(self) -> float:
        return float(sum(max(0.0, float(s.latency or 0.0)) for s in self.steps))

    def append(self, step: TraceStep | Mapping[str, Any], **kwargs: Any) -> TraceStep:
        if not isinstance(step, TraceStep):
            step = TraceStep.from_dict(step)
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(step, key):
                    setattr(step, key, value)
        self.steps.append(step)
        return step

    def finish(self, status: str, *, final_observation: Any = None,
               final_state: Any = None, error: Optional[str] = None) -> None:
        self.status = str(status)
        self.finished_at = utc_now()
        if final_observation is not None:
            self.final_observation = final_observation
        if final_state is not None:
            self.final_state = final_state
        if error:
            self.error = str(error)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "seed": self.seed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "steps": [s.to_dict() if isinstance(s, TraceStep) else json_safe(s) for s in self.steps],
            "final_observation": self.final_observation,
            "final_state": self.final_state,
            "metadata": self.metadata,
            "error": self.error,
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentTrace":
        trace = cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            agent_id=str(data.get("agent_id", "")),
            seed=data.get("seed"),
            started_at=str(data.get("started_at", utc_now())),
            finished_at=data.get("finished_at"),
            status=str(data.get("status", "unknown")),
            final_observation=data.get("final_observation"),
            final_state=data.get("final_state"),
            metadata=dict(data.get("metadata", {}) or {}),
            error=data.get("error"),
        )
        trace.steps = [TraceStep.from_dict(item) for item in data.get("steps", [])]
        return trace

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "AgentTrace":
        return cls.from_dict(json.loads(payload))

    def save(self, path: str | Path, *, indent: Optional[int] = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target


@dataclass
class RunResult:
    """Runner output, with evaluator fields attached when available."""

    run_id: str
    task_id: str
    agent_id: str
    status: str
    success: Optional[bool]
    trace: AgentTrace
    score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    final_observation: Any = None
    final_state: Any = None
    error: Optional[str] = None
    seed: Optional[int] = None

    @property
    def steps(self) -> List[TraceStep]:
        return self.trace.steps

    @property
    def step_count(self) -> int:
        return self.trace.step_count

    def to_dict(self, *, include_trace: bool = True) -> Dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "seed": self.seed,
            "status": self.status,
            "success": self.success,
            "score": self.score,
            "metrics": self.metrics,
            "failures": self.failures,
            "final_observation": self.final_observation,
            "final_state": self.final_state,
            "error": self.error,
        }
        if include_trace:
            data["trace"] = self.trace.to_dict()
        return json_safe(data)


__all__ = ["TraceStep", "AgentTrace", "RunResult", "json_safe", "utc_now"]
