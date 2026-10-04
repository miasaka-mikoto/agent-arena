"""Replay data model and a dependency-free replay viewer.

The arena deliberately keeps replay files boring: a replay is a JSON object
containing run metadata and an ordered list of public trace events.  This makes
the files useful outside the application (for example in notebooks or CI)
while the :class:`ReplayViewer` gives users a small, self-contained HTML
viewer that works without a web server.

The module does not import a particular runner implementation.  ``from_trace``
and ``from_run`` accept dataclasses, dictionaries, or ordinary objects and
normalise the common fields used by Agent Arena's runner.  This loose boundary
is intentional: experiment code can evolve without invalidating old replay
files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import uuid


def _get(value: Any, *names: str, default: Any = None) -> Any:
    """Read the first present field from a mapping or an object."""

    if value is None:
        return default
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Convert arbitrary experiment values into safe, JSON-friendly values.

    Environment observations occasionally contain sets, enums, paths, or
    custom dataclasses.  Replays should never fail merely because one such
    value was returned by a tool, so unknown leaf values fall back to ``repr``.
    A depth guard also prevents accidental cycles from taking down reporting.
    """

    if _depth > 12:
        return "<max-depth>"
    if value is None or isinstance(value, (str, int, float, bool)):
        # JSON has no representation for NaN/Infinity.  Stringifying them is
        # more useful than writing invalid replay files.
        if isinstance(value, float) and (value != value or abs(value) == float("inf")):
            return repr(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, _depth=_depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v, _depth=_depth + 1) for v in value), key=repr)
    if is_dataclass(value):
        return _jsonable(asdict(value), _depth=_depth + 1)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict(), _depth=_depth + 1)
        except Exception:
            pass
    # Enum-like values are preferable as their value, if it is primitive.
    enum_value = getattr(value, "value", None)
    if enum_value is not None and isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return repr(value)


def _events_from(source: Any) -> list[Any]:
    """Find an ordered event/step collection on ``source``."""

    if source is None:
        return []
    if isinstance(source, Mapping):
        for key in ("events", "steps", "trace", "records"):
            candidate = source.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                return list(candidate)
        return []
    for key in ("events", "steps", "trace", "records"):
        candidate = getattr(source, key, None)
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return list(candidate)
    return []


@dataclass
class ReplayEvent:
    """One public agent/environment transition.

    ``public_reason_summary`` is intentionally a short, user-visible summary;
    private chain-of-thought must never be placed in this field.  The optional
    state fields allow an environment to expose snapshots without requiring a
    particular environment implementation.
    """

    step: int = 0
    observation: Any = None
    action: Any = None
    tool: str | None = None
    arguments: Any = None
    result: Any = None
    public_reason_summary: str | None = None
    error: Any = None
    latency_ms: float | None = None
    state_before: Any = None
    state_after: Any = None
    timestamp: str | None = None

    @property
    def reason(self) -> str | None:
        """Backward-compatible alias used by a few early runner prototypes."""

        return self.public_reason_summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "observation": _jsonable(self.observation),
            "action": _jsonable(self.action),
            "tool": self.tool,
            "arguments": _jsonable(self.arguments),
            "result": _jsonable(self.result),
            "public_reason_summary": self.public_reason_summary,
            "error": _jsonable(self.error),
            "latency_ms": self.latency_ms,
            "state_before": _jsonable(self.state_before),
            "state_after": _jsonable(self.state_after),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_any(cls, value: Any, index: int = 0) -> "ReplayEvent":
        if isinstance(value, cls):
            return value
        step = _get(value, "step", "step_index", "index", default=index)
        try:
            step = int(step)
        except (TypeError, ValueError):
            step = index
        # Runner versions have used both ``args`` and ``arguments``.
        return cls(
            step=step,
            observation=_get(value, "observation", "obs"),
            action=_get(value, "action"),
            tool=_get(value, "tool", "tool_name"),
            arguments=_get(value, "arguments", "args", "tool_args"),
            result=_get(value, "result", "tool_result", "output"),
            public_reason_summary=_get(
                value,
                "public_reason_summary",
                "reason_summary",
                "public_reason",
                "reason",
            ),
            error=_get(value, "error", "exception"),
            latency_ms=_get(value, "latency_ms", "latency", "duration_ms"),
            state_before=_get(value, "state_before", "environment_before", "world_before"),
            # TraceStep calls the public post-action snapshot ``state_snapshot``;
            # accept that name so no environment state is lost in a replay.
            state_after=_get(value, "state_after", "environment_after", "world_after", "state_snapshot"),
            timestamp=_get(value, "timestamp", "time"),
        )


@dataclass
class Replay:
    """A complete, serialisable run replay."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_id: str = ""
    seed: int | str | None = None
    status: str = "completed"
    success: bool | None = None
    events: list[ReplayEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def steps(self) -> list[ReplayEvent]:
        """Alias for APIs that refer to events as steps."""

        return self.events

    @property
    def step_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "seed": _jsonable(self.seed),
            "status": self.status,
            "success": self.success,
            "events": [event.to_dict() for event in self.events],
            "metadata": _jsonable(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Replay":
        events = value.get("events", value.get("steps", value.get("trace", [])))
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            events = []
        return cls(
            run_id=str(value.get("run_id", value.get("id", uuid.uuid4()))),
            task_id=str(value.get("task_id", "")),
            agent_id=str(value.get("agent_id", value.get("agent", ""))),
            seed=value.get("seed"),
            status=str(value.get("status", "completed")),
            success=value.get("success"),
            events=[ReplayEvent.from_any(event, i) for i, event in enumerate(events)],
            metadata=dict(value.get("metadata", {})) if isinstance(value.get("metadata", {}), Mapping) else {},
            created_at=str(value.get("created_at", "")),
        )

    @classmethod
    def from_trace(
        cls,
        trace: Any,
        *,
        run: Any = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        seed: int | str | None = None,
    ) -> "Replay":
        """Build a replay from an ``AgentTrace`` or compatible object."""

        source = run if run is not None else trace
        events = _events_from(trace)
        metadata = _get(source, "metadata", default={})
        if not isinstance(metadata, Mapping):
            metadata = {}
        success = _get(source, "success", "succeeded")
        status = _get(source, "status", default="completed")
        source_seed = _get(source, "seed", default=_get(trace, "seed"))
        return cls(
            run_id=str(_get(source, "run_id", "id", default=uuid.uuid4())),
            task_id=str(task_id if task_id is not None else _get(source, "task_id", default="")),
            agent_id=str(agent_id if agent_id is not None else _get(source, "agent_id", "agent", default="")),
            seed=seed if seed is not None else source_seed,
            status=str(status),
            success=success,
            events=[ReplayEvent.from_any(event, i) for i, event in enumerate(events)],
            metadata=dict(metadata),
            created_at=str(_get(source, "created_at", default=datetime.now(timezone.utc).isoformat())),
        )

    @classmethod
    def from_run(cls, run: Any) -> "Replay":
        """Build from a RunResult, including its nested trace if present."""

        trace = _get(run, "trace", "agent_trace", "replay", default=run)
        if isinstance(trace, Replay):
            return trace
        return cls.from_trace(trace, run=run)


class ReplayStore:
    """Small in-memory/indexed store used by the demo and report generator."""

    def __init__(self, replays: Iterable[Replay | Mapping[str, Any]] | None = None):
        self._items: dict[str, Replay] = {}
        for replay in replays or ():
            self.add(replay)

    def add(self, replay: Replay | Mapping[str, Any] | Any) -> Replay:
        if not isinstance(replay, Replay):
            replay = Replay.from_dict(replay) if isinstance(replay, Mapping) else Replay.from_run(replay)
        self._items[replay.run_id] = replay
        return replay

    def get(self, run_id: str) -> Replay | None:
        return self._items.get(str(run_id))

    def all(self) -> list[Replay]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "replays": [item.to_dict() for item in self.all()]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReplayStore":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        items = value.get("replays", []) if isinstance(value, Mapping) else value
        return cls(Replay.from_dict(item) for item in (items if isinstance(items, list) else []))

    def jsonl(self, path: str | Path) -> Path:
        """Write one replay per line for streaming/large tournament outputs."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in self.all()) + "\n", encoding="utf-8")
        return path


class ReplayViewer:
    """Render a replay as a standalone, keyboard-friendly HTML document."""

    @staticmethod
    def render(replay: Replay | Mapping[str, Any] | Any, *, title: str = "Agent Arena Replay") -> str:
        if not isinstance(replay, Replay):
            replay = Replay.from_dict(replay) if isinstance(replay, Mapping) else Replay.from_run(replay)
        payload = json.dumps(replay.to_dict(), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
        summary = f"{html.escape(replay.agent_id or 'unknown agent')} · {html.escape(replay.task_id or 'unknown task')} · {replay.step_count} steps"
        title_escaped = html.escape(title)
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_escaped}</title>
<style>
:root{{--bg:#0b1020;--panel:#141b2e;--panel2:#1a2440;--text:#e8edf8;--muted:#96a3bf;--accent:#66d9ef;--ok:#64d38a;--bad:#ff7c91;--line:#293653}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}}
header{{padding:22px 28px;border-bottom:1px solid var(--line);background:#0f162a;position:sticky;top:0;z-index:2}}
h1{{font-size:22px;margin:0 0 5px}} .muted{{color:var(--muted)}} .layout{{display:grid;grid-template-columns:270px minmax(0,1fr);min-height:calc(100vh - 95px)}}
aside{{border-right:1px solid var(--line);padding:16px;overflow:auto}} main{{padding:24px;max-width:1100px;width:100%}}
button.step{{display:block;width:100%;text-align:left;background:transparent;color:var(--text);border:1px solid transparent;border-radius:8px;padding:10px;margin:3px 0;cursor:pointer}}
button.step:hover,button.step.active{{background:var(--panel2);border-color:var(--accent)}} .step-error{{color:var(--bad)}} .step-ok{{color:var(--ok)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:17px;margin-bottom:16px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.kv b{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}} pre{{white-space:pre-wrap;overflow:auto;background:#0a0f1d;border:1px solid var(--line);border-radius:7px;padding:12px;margin:7px 0 0;color:#dce6fb}}
.section{{margin-top:16px}} .section h3{{margin:0 0 4px;font-size:14px;color:var(--accent)}} .badge{{display:inline-block;padding:2px 8px;border-radius:99px;background:var(--panel2);font-size:12px}}
@media(max-width:720px){{.layout{{grid-template-columns:1fr}}aside{{border-right:0;border-bottom:1px solid var(--line);max-height:220px}}main{{padding:14px}}}}
</style></head><body>
<header><h1>{title_escaped}</h1><div class="muted">{summary} · Run <code>{html.escape(replay.run_id)}</code></div></header>
<div class="layout"><aside><div class="muted" style="margin:0 0 8px">Timeline</div><div id="timeline"></div></aside><main id="detail"></main></div>
<script>
const replay={payload};
const timeline=document.getElementById('timeline'), detail=document.getElementById('detail');
const esc=(v)=>{{const d=document.createElement('div');d.textContent=v===undefined||v===null?'':String(v);return d.innerHTML}};
const pretty=(v)=>{{if(v===undefined||v===null)return '—';try{{return JSON.stringify(v,null,2)}}catch(e){{return String(v)}}}};
function show(i){{const e=replay.events[i]||{{}};document.querySelectorAll('.step').forEach((b,j)=>b.classList.toggle('active',j===i));
 const err=e.error?'<span class="badge step-error">error</span>':'<span class="badge step-ok">ok</span>';
 detail.innerHTML=`<div class="card"><div class="grid"><div class="kv"><b>Step</b>${{esc(e.step)}}</div><div class="kv"><b>Tool</b>${{esc(e.tool||'—')}}</div><div class="kv"><b>Latency</b>${{esc(e.latency_ms==null?'—':e.latency_ms+' ms')}}</div><div class="kv"><b>Status</b>${{err}}</div></div></div>
 <div class="card"><div class="section"><h3>Observation</h3><pre>${{esc(pretty(e.observation))}}</pre></div>
 <div class="section"><h3>Action</h3><pre>${{esc(pretty(e.action))}}</pre></div>
 <div class="section"><h3>Tool arguments</h3><pre>${{esc(pretty(e.arguments))}}</pre></div>
 <div class="section"><h3>Tool result</h3><pre>${{esc(pretty(e.result))}}</pre></div>
 ${{e.public_reason_summary?'<div class="section"><h3>Public reason summary</h3><p>'+esc(e.public_reason_summary)+'</p></div>':''}}
 ${{e.error?'<div class="section"><h3>Error</h3><pre>'+esc(pretty(e.error))+'</pre></div>':''}}
 <div class="section"><h3>Environment state after</h3><pre>${{esc(pretty(e.state_after))}}</pre></div></div>`}}
replay.events.forEach((e,i)=>{{const b=document.createElement('button');b.className='step';b.innerHTML=`<strong>#${{esc(e.step??i)}}</strong> <span class="muted">${{esc(e.tool||e.action||'transition')}}</span>${{e.error?' <span class="step-error">●</span>':''}}`;b.onclick=()=>show(i);timeline.appendChild(b)}});
if(replay.events.length)show(0);else detail.innerHTML='<div class="card">This replay contains no events.</div>';
document.addEventListener('keydown',(ev)=>{{const active=[...document.querySelectorAll('.step')].findIndex((b)=>b.classList.contains('active'));if(ev.key==='ArrowDown'&&active<replay.events.length-1)show(active+1);if(ev.key==='ArrowUp'&&active>0)show(active-1)}});
</script></body></html>'''

    @classmethod
    def write(cls, replay: Replay | Mapping[str, Any] | Any, path: str | Path, *, title: str = "Agent Arena Replay") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cls.render(replay, title=title), encoding="utf-8")
        return path


def render_replay(replay: Replay | Mapping[str, Any] | Any, *, title: str = "Agent Arena Replay") -> str:
    """Convenience function kept stable for CLI/report integrations."""

    return ReplayViewer.render(replay, title=title)


__all__ = ["ReplayEvent", "Replay", "ReplayStore", "ReplayViewer", "render_replay"]
