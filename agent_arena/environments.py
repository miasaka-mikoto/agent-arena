"""Safe, deterministic environments used by Agent Arena.

The environments in this module are deliberately *not* adapters around a
real filesystem, mailbox, calendar, browser, interpreter, or operating
system.  Every object is an in-memory simulation.  This makes benchmark runs
reproducible and, more importantly, prevents a task or an agent from touching
user data by accident.

Each environment implements the small protocol used by the runner::

    reset(seed=None) -> Observation-like mapping
    observe() -> public observation mapping
    execute_tool(name, arguments) -> ToolResult
    execute_action(action) -> ToolResult

``core.ToolResult`` is imported lazily when the core module is available.  A
small fallback result type keeps this module useful while the package is
being assembled and in standalone tests.
"""

from __future__ import annotations

import ast
import copy
import datetime as _dt
import math
import operator
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Result and protocol helpers


@dataclass
class LocalToolResult:
    """Fallback equivalent of :class:`agent_arena.core.ToolResult`.

    The main project uses ``core.ToolResult``.  Keeping this fallback local is
    useful when environments are imported in isolation (for example while a
    new task is being authored).
    """

    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _result(
    tool: str,
    success: bool,
    output: Any = None,
    error: str | None = None,
    *,
    started: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Create the project's shared ToolResult without a hard import cycle."""

    latency = 0.0 if started is None else round((time.perf_counter() - started) * 1000, 3)
    meta = dict(metadata or {})
    try:
        # Imported lazily because core.py is intentionally authored in
        # parallel with this module.
        from .core import ToolResult as SharedToolResult  # type: ignore

        return SharedToolResult(
            tool=tool,
            success=bool(success),
            output=output,
            error=error,
            latency_ms=latency,
            metadata=meta,
        )
    except (ImportError, AttributeError, TypeError):
        return LocalToolResult(
            tool=tool,
            success=bool(success),
            output=output,
            error=error,
            latency_ms=latency,
            metadata=meta,
        )


class EnvironmentProtocol:
    """Informal protocol implemented by every simulated environment."""

    name = "environment"
    available_tools: tuple[str, ...] = ()

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def observe(self) -> Mapping[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def execute_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:  # pragma: no cover
        raise NotImplementedError

    def execute_action(self, action: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


class BaseEnvironment(EnvironmentProtocol):
    """Common bookkeeping and safe dispatch for simulated environments."""

    name = "environment"
    available_tools: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.seed: int | None = None
        self.step_count = 0
        self.done = False
        self.success = False
        self.invalid_actions = 0
        self.last_error: str | None = None

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        self.seed = seed
        self.step_count = 0
        self.done = False
        self.success = False
        self.invalid_actions = 0
        self.last_error = None
        # Subclasses restore their private state immediately after this call
        # and then return ``observe()``.  Calling observe here would race that
        # restoration during construction (for example before ``files`` has
        # been created in VirtualFileSystem).
        return {}

    def _after_action(self, result: Any) -> Any:
        """Update generic counters while preserving the result object."""

        self.step_count += 1
        ok = bool(getattr(result, "success", False))
        if not ok:
            self.invalid_actions += 1
            self.last_error = getattr(result, "error", None)
        return result

    def execute_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        requested_name = str(name)
        # Friendly action names are accepted at the environment boundary too;
        # task wrappers may still enforce their canonical allow-list before
        # this point.
        action_aliases = {
            "move": "grid_move",
            "turn": "grid_turn",
            "pick_up": "grid_pick_up",
            "pickup": "grid_pick_up",
            "drop": "grid_drop",
            "use": "grid_use",
            "interact": "grid_interact",
            "wait": "grid_wait",
        }
        canonical_name = action_aliases.get(requested_name, requested_name)
        args = dict(arguments or {})
        # Normalize a few provider-friendly spellings at the boundary.  The
        # canonical names remain visible in traces and tool specifications.
        aliases: dict[str, dict[str, str]] = {
            "list_files": {"directory": "path", "dir": "path"},
            "read_file": {"file": "path", "file_path": "path"},
            "move_file": {"src": "source", "from_path": "source", "dest": "destination", "to": "destination", "to_path": "destination"},
            "delete_file": {"file": "path", "file_path": "path"},
            "read_mail": {"id": "message_id", "mail_id": "message_id"},
            "mark_mail_read": {"id": "message_id", "mail_id": "message_id"},
            "calendar_find_conflicts": {"id": "event_id"},
            "browser_open": {"href": "url"},
            "sandbox_read_file": {"file": "path", "file_path": "path"},
            "sandbox_write_file": {"file": "path", "file_path": "path"},
        }
        for old, new in aliases.get(canonical_name, {}).items():
            if new not in args and old in args:
                args[new] = args.pop(old)
        started = time.perf_counter()
        method = getattr(self, canonical_name, None)
        if not callable(method) or canonical_name.startswith("_"):
            return self._after_action(
                _result(requested_name, False, error=f"Unknown tool: {requested_name}", started=started, metadata={"environment": self.name})
            )
        try:
            output = method(**args)
            # Environment methods conventionally return a mapping; if a method
            # already returned a ToolResult, keep it intact.
            if hasattr(output, "success") and hasattr(output, "tool"):
                if requested_name != canonical_name and getattr(output, "tool", "") == canonical_name:
                    try:
                        output.tool = requested_name
                    except Exception:
                        pass
                return self._after_action(output)
            return self._after_action(
                _result(requested_name, True, output=output, started=started, metadata={"environment": self.name})
            )
        except TypeError as exc:
            return self._after_action(
                _result(requested_name, False, error=f"Invalid arguments: {exc}", started=started, metadata={"environment": self.name})
            )
        except Exception as exc:  # simulated tools must never crash a run
            return self._after_action(
                _result(requested_name, False, error=f"Tool error: {exc}", started=started, metadata={"environment": self.name})
            )

    def execute_action(self, action: Any) -> Any:
        """Accept an Action object, a mapping, or a bare tool name.

        This adapter lets the runner use the same call for core ``Action``
        instances and lightweight scripted actions.
        """

        if isinstance(action, str):
            return self.execute_tool(action, {})
        if isinstance(action, Mapping):
            name = action.get("tool") or action.get("name") or action.get("action")
            args = action.get("arguments") or action.get("args") or {}
        else:
            name = getattr(action, "tool", None) or getattr(action, "name", None) or getattr(action, "action_type", None)
            args = getattr(action, "arguments", None) or getattr(action, "args", None) or {}
        if not name:
            self.step_count += 1
            self.invalid_actions += 1
            return _result("", False, error="Action did not specify a tool", metadata={"environment": self.name})
        return self.execute_tool(str(name), args)

    def snapshot(self) -> Mapping[str, Any]:
        """Return the public state used by replay viewers."""

        return dict(self.observe())

    # Runner/evaluator compatibility aliases.  They intentionally expose only
    # the same public snapshot an agent would receive.
    def public_state(self) -> Mapping[str, Any]:
        return self.snapshot()

    def snapshot_public(self) -> Mapping[str, Any]:
        return self.snapshot()

    def is_done(self) -> bool:
        return bool(self.done)

    def is_success(self, task: Any | None = None) -> bool:
        return bool(self.success)

    def ground_truth(self) -> Mapping[str, Any]:
        """Return a copy of complete state for evaluators, never agents."""

        return copy.deepcopy(dict(self.snapshot()))

    def get_ground_truth(self) -> Mapping[str, Any]:
        return self.ground_truth()

    def tool_specs(self) -> list[dict[str, Any]]:
        """Describe callable tools without exposing implementation objects."""

        return [{"name": name, "description": name.replace("_", " ").title()} for name in self.available_tools]


def _state_mapping(
    initial_state: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    task: Any | None = None,
) -> Mapping[str, Any]:
    """Extract public task state accepted by flexible ``reset`` calls."""

    value: Any = initial_state if initial_state is not None else state
    if value is None and task is not None:
        value = task.get("initial_state", {}) if isinstance(task, Mapping) else getattr(task, "initial_state", {})
    return value if isinstance(value, Mapping) else {}


# ---------------------------------------------------------------------------
# Virtual file system


def _safe_path(path: str | PurePosixPath, *, allow_root: bool = True) -> str:
    if not isinstance(path, (str, PurePosixPath)):
        raise ValueError("path must be a string")
    raw = str(path).replace("\\", "/")
    if "\x00" in raw:
        raise ValueError("path contains NUL")
    if not raw.startswith("/"):
        raw = "/" + raw
    parts: list[str] = []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError("path traversal is not allowed")
        parts.append(part)
    normal = "/" + "/".join(parts)
    if not normal and allow_root:
        return "/"
    return normal or "/"


def _coerce_file_mapping(files: Any) -> dict[str, str]:
    """Accept mapping or ``[{path, content}]`` task-authoring shapes."""

    if isinstance(files, Mapping):
        return {_safe_path(str(key)): str(value) for key, value in files.items()}
    result: dict[str, str] = {}
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        for item in files:
            if isinstance(item, Mapping):
                key = item.get("path", item.get("name"))
                if key is not None:
                    result[_safe_path(str(key))] = str(item.get("content", item.get("text", "")))
    return result


class VirtualFileSystem(BaseEnvironment):
    """An in-memory POSIX-like filesystem with no host I/O."""

    name = "virtual_file_system"
    available_tools = (
        "list_files",
        "read_file",
        "write_file",
        "move_file",
        "delete_file",
        "search_files",
    )

    def __init__(self, files: Mapping[str, str] | None = None, directories: Iterable[str] | None = None) -> None:
        super().__init__()
        self._initial_files = _coerce_file_mapping(files or {})
        if isinstance(directories, (str, PurePosixPath)):
            directories = [str(directories)]
        self._initial_dirs = {_safe_path(d) for d in (directories or ())}
        self._initial_dirs.add("/")
        for path in self._initial_files:
            self._initial_dirs.update(self._parents(path))
        self.files: dict[str, str] = {}
        self.directories: set[str] = set()
        self.reset()

    @staticmethod
    def _parents(path: str) -> set[str]:
        p = PurePosixPath(path)
        result = {"/"}
        current = p.parent
        while str(current) not in ("", "."):
            result.add(_safe_path(str(current)))
            if str(current) == "/":
                break
            current = current.parent
        return result

    def reset(
        self,
        seed: int | None = None,
        initial_state: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        task: Any | None = None,
    ) -> Mapping[str, Any]:
        super().reset(seed)
        payload = _state_mapping(initial_state, state, task)
        raw_files = payload.get("files", self._initial_files)
        if isinstance(raw_files, Mapping):
            self.files = {_safe_path(k): str(v) for k, v in raw_files.items()}
        else:
            # Hand-authored tasks sometimes use [{path, content}] records.
            self.files = {}
            for item in raw_files if isinstance(raw_files, Sequence) and not isinstance(raw_files, (str, bytes)) else ():
                if isinstance(item, Mapping):
                    key = item.get("path", item.get("name"))
                    if key is not None:
                        self.files[_safe_path(str(key))] = str(item.get("content", item.get("text", "")))
        raw_dirs = payload.get("directories", self._initial_dirs)
        self.directories = {_safe_path(d) for d in raw_dirs} if isinstance(raw_dirs, Iterable) and not isinstance(raw_dirs, (str, bytes, Mapping)) else set(self._initial_dirs)
        self.directories.add("/")
        for path in self.files:
            self.directories.update(self._parents(path))
        return self.observe()

    def _entry(self, path: str) -> dict[str, Any]:
        if path in self.files:
            return {"path": path, "type": "file", "size": len(self.files[path].encode("utf-8"))}
        if path in self.directories:
            return {"path": path, "type": "directory"}
        raise KeyError(path)

    def observe(self) -> Mapping[str, Any]:
        return {
            "environment": self.name,
            "step": self.step_count,
            "cwd": "/",
            "root_entries": self.list_files("/", recursive=False),
            "available_tools": list(self.available_tools),
            "done": self.done,
        }

    def list_files(self, path: str = "/", recursive: bool = False) -> Mapping[str, Any]:
        parent = _safe_path(path)
        if parent not in self.directories and parent not in self.files:
            raise ValueError(f"path does not exist: {parent}")
        if parent in self.files:
            return {"path": parent, "entries": [self._entry(parent)]}
        prefix = parent.rstrip("/") + "/" if parent != "/" else "/"
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in sorted((*self.directories, *self.files)):
            if candidate == parent or not candidate.startswith(prefix):
                continue
            remainder = candidate[len(prefix) :]
            if not remainder:
                continue
            if not recursive and "/" in remainder.rstrip("/"):
                first = prefix + remainder.split("/", 1)[0]
                if first not in seen:
                    seen.add(first)
                    # A first segment may be an implicit directory.
                    entries.append(self._entry(first) if first in self.files or first in self.directories else {"path": first, "type": "directory"})
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            entries.append(self._entry(candidate))
        return {"path": parent, "entries": entries}

    def read_file(self, path: str, start_line: int = 1, max_chars: int | None = None) -> Mapping[str, Any]:
        target = _safe_path(path)
        if target not in self.files:
            raise ValueError(f"file does not exist: {target}")
        if start_line < 1:
            raise ValueError("start_line must be >= 1")
        content = self.files[target]
        lines = content.splitlines(keepends=True)
        sliced = "".join(lines[start_line - 1 :])
        if max_chars is not None:
            if max_chars < 0:
                raise ValueError("max_chars must be non-negative")
            sliced = sliced[:max_chars]
        return {"path": target, "content": sliced, "size": len(content.encode("utf-8")), "truncated": sliced != "".join(lines[start_line - 1 :])}

    def write_file(self, path: str, content: str, overwrite: bool = True) -> Mapping[str, Any]:
        target = _safe_path(path)
        if target in self.directories:
            raise ValueError(f"cannot overwrite directory: {target}")
        if target in self.files and not overwrite:
            raise ValueError(f"file already exists: {target}")
        self.directories.update(self._parents(target))
        self.files[target] = str(content)
        return {"path": target, "written": True, "size": len(self.files[target].encode("utf-8"))}

    def move_file(self, source: str, destination: str, overwrite: bool = False) -> Mapping[str, Any]:
        src = _safe_path(source)
        dest = _safe_path(destination)
        if src not in self.files:
            raise ValueError(f"file does not exist: {src}")
        if dest in self.directories:
            dest = _safe_path(dest.rstrip("/") + "/" + PurePosixPath(src).name)
        if dest in self.files and not overwrite:
            raise ValueError(f"destination already exists: {dest}")
        self.directories.update(self._parents(dest))
        self.files[dest] = self.files.pop(src)
        return {"source": src, "destination": dest, "moved": True}

    def delete_file(self, path: str) -> Mapping[str, Any]:
        target = _safe_path(path)
        if target not in self.files:
            raise ValueError(f"file does not exist: {target}")
        del self.files[target]
        return {"path": target, "deleted": True}

    def search_files(self, query: str, path: str = "/", case_sensitive: bool = False) -> Mapping[str, Any]:
        root = _safe_path(path)
        if root not in self.directories and root not in self.files:
            raise ValueError(f"path does not exist: {root}")
        needle = str(query) if case_sensitive else str(query).lower()
        matches = []
        for candidate, content in sorted(self.files.items()):
            if root != "/" and not (candidate == root or candidate.startswith(root.rstrip("/") + "/")):
                continue
            haystack = content if case_sensitive else content.lower()
            filename = PurePosixPath(candidate).name if case_sensitive else PurePosixPath(candidate).name.lower()
            if needle in haystack or needle in filename:
                matches.append({"path": candidate, "matches": haystack.count(needle) if needle else 0})
        return {"query": query, "matches": matches}

    def ground_truth(self) -> Mapping[str, Any]:
        return {
            "environment": self.name,
            "files": copy.deepcopy(self.files),
            "directories": sorted(self.directories),
            "step": self.step_count,
            "done": self.done,
        }


# ---------------------------------------------------------------------------
# Virtual mail


@dataclass
class EmailMessage:
    id: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    timestamp: str
    labels: list[str] = field(default_factory=list)
    unread: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "EmailMessage":
        raw_recipients = value.get("recipients", value.get("to", []))
        raw_labels = value.get("labels", [])
        recipients = [str(x) for x in raw_recipients] if isinstance(raw_recipients, (list, tuple, set)) else ([str(raw_recipients)] if raw_recipients not in (None, "") else [])
        labels = [str(x) for x in raw_labels] if isinstance(raw_labels, (list, tuple, set)) else ([str(raw_labels)] if raw_labels not in (None, "") else [])
        return cls(
            id=str(value.get("id", f"m{index + 1}")),
            sender=str(value.get("sender", value.get("from", "unknown@example.test"))),
            recipients=recipients,
            subject=str(value.get("subject", "")),
            body=str(value.get("body", "")),
            timestamp=str(value.get("timestamp", value.get("date", ""))),
            labels=labels,
            unread=bool(value.get("unread", True)),
        )


class VirtualEmailInbox(BaseEnvironment):
    """An in-memory inbox with search and read-state semantics."""

    name = "virtual_email_inbox"
    available_tools = ("list_mail", "search_mail", "read_mail", "mark_mail_read")

    def __init__(self, messages: Sequence[Mapping[str, Any] | EmailMessage] | None = None) -> None:
        super().__init__()
        self._initial_messages: list[EmailMessage] = []
        if isinstance(messages, Mapping):
            messages = list(messages.values()) if all(isinstance(v, Mapping) for v in messages.values()) else [messages]  # type: ignore[list-item]
        for idx, message in enumerate(messages or ()):
            self._initial_messages.append(message if isinstance(message, EmailMessage) else EmailMessage.from_mapping(message, idx))
        self.messages: dict[str, EmailMessage] = {}
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        initial_state: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        task: Any | None = None,
    ) -> Mapping[str, Any]:
        super().reset(seed)
        payload = _state_mapping(initial_state, state, task)
        raw_messages = payload.get("messages", payload.get("emails", None))
        if raw_messages is None:
            messages = self._initial_messages
        else:
            messages = [
                item if isinstance(item, EmailMessage) else EmailMessage.from_mapping(item, index)
                for index, item in enumerate(raw_messages if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)) else ())
            ]
        self.messages = {m.id: copy.deepcopy(m) for m in messages}
        return self.observe()

    @staticmethod
    def _summary(message: EmailMessage) -> dict[str, Any]:
        return {"id": message.id, "from": message.sender, "to": list(message.recipients), "subject": message.subject, "timestamp": message.timestamp, "labels": list(message.labels), "unread": message.unread}

    def observe(self) -> Mapping[str, Any]:
        recent = sorted(self.messages.values(), key=lambda m: m.timestamp, reverse=True)
        return {"environment": self.name, "step": self.step_count, "messages": [self._summary(m) for m in recent[:10]], "unread_count": sum(m.unread for m in self.messages.values()), "available_tools": list(self.available_tools)}

    def list_mail(self, unread_only: bool = False, label: str | None = None, limit: int = 50) -> Mapping[str, Any]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        messages = [m for m in self.messages.values() if (not unread_only or m.unread) and (label is None or label in m.labels)]
        messages.sort(key=lambda m: m.timestamp, reverse=True)
        return {"messages": [self._summary(m) for m in messages[:limit]], "total": len(messages)}

    def search_mail(self, query: str, unread_only: bool = False, sender: str | None = None, limit: int = 50) -> Mapping[str, Any]:
        q = str(query).lower()
        messages = []
        for message in self.messages.values():
            if unread_only and not message.unread:
                continue
            if sender and sender.lower() not in message.sender.lower():
                continue
            haystack = " ".join((message.sender, " ".join(message.recipients), message.subject, message.body, " ".join(message.labels))).lower()
            if q in haystack:
                messages.append(message)
        messages.sort(key=lambda m: m.timestamp, reverse=True)
        return {"query": query, "messages": [self._summary(m) for m in messages[:limit]], "total": len(messages)}

    def read_mail(self, message_id: str, mark_read: bool = False) -> Mapping[str, Any]:
        key = str(message_id)
        if key not in self.messages:
            raise ValueError(f"message does not exist: {key}")
        message = self.messages[key]
        if mark_read:
            message.unread = False
        return {"id": message.id, "from": message.sender, "to": list(message.recipients), "subject": message.subject, "body": message.body, "timestamp": message.timestamp, "labels": list(message.labels), "unread": message.unread}

    def mark_mail_read(self, message_id: str, unread: bool = False) -> Mapping[str, Any]:
        key = str(message_id)
        if key not in self.messages:
            raise ValueError(f"message does not exist: {key}")
        self.messages[key].unread = bool(unread)
        return {"id": key, "unread": self.messages[key].unread}

    def ground_truth(self) -> Mapping[str, Any]:
        return {"environment": self.name, "messages": [asdict(m) for m in self.messages.values()], "step": self.step_count}


# ---------------------------------------------------------------------------
# Virtual calendar


def _parse_dt(value: Any) -> _dt.datetime:
    if isinstance(value, _dt.datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _dt.datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        # Date-only values are useful for benchmark instructions.
        try:
            return _dt.datetime.combine(_dt.date.fromisoformat(text), _dt.time())
        except ValueError as exc:
            raise ValueError(f"invalid datetime: {value}") from exc


@dataclass
class CalendarEvent:
    id: str
    title: str
    start: str
    end: str
    attendees: list[str] = field(default_factory=list)
    location: str = ""
    description: str = ""
    status: str = "confirmed"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "CalendarEvent":
        start = str(value.get("start", ""))
        end = str(value.get("end", ""))
        if not start or not end:
            raise ValueError("calendar event requires start and end")
        raw_attendees = value.get("attendees", [])
        attendees = [str(x) for x in raw_attendees] if isinstance(raw_attendees, (list, tuple, set)) else ([str(raw_attendees)] if raw_attendees not in (None, "") else [])
        return cls(
            id=str(value.get("id", f"e{index + 1}")),
            title=str(value.get("title", value.get("summary", "Untitled"))),
            start=start,
            end=end,
            attendees=attendees,
            location=str(value.get("location", "")),
            description=str(value.get("description", "")),
            status=str(value.get("status", "confirmed")),
        )


class VirtualCalendar(BaseEnvironment):
    """A deterministic in-memory calendar; all datetimes are local-naive."""

    name = "virtual_calendar"
    available_tools = ("calendar_lookup", "calendar_find_conflicts", "calendar_create", "calendar_update", "calendar_cancel")

    def __init__(self, events: Sequence[Mapping[str, Any] | CalendarEvent] | None = None) -> None:
        super().__init__()
        self._initial_events: list[CalendarEvent] = []
        if isinstance(events, Mapping):
            events = list(events.values()) if all(isinstance(v, Mapping) for v in events.values()) else [events]  # type: ignore[list-item]
        for idx, event in enumerate(events or ()):
            self._initial_events.append(event if isinstance(event, CalendarEvent) else CalendarEvent.from_mapping(event, idx))
        self.events: dict[str, CalendarEvent] = {}
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        initial_state: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        task: Any | None = None,
    ) -> Mapping[str, Any]:
        super().reset(seed)
        payload = _state_mapping(initial_state, state, task)
        raw_events = payload.get("events", None)
        if raw_events is None:
            events = self._initial_events
        else:
            events = [
                item if isinstance(item, CalendarEvent) else CalendarEvent.from_mapping(item, index)
                for index, item in enumerate(raw_events if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)) else ())
            ]
        self.events = {e.id: copy.deepcopy(e) for e in events}
        return self.observe()

    @staticmethod
    def _summary(event: CalendarEvent) -> dict[str, Any]:
        return {"id": event.id, "title": event.title, "start": event.start, "end": event.end, "attendees": list(event.attendees), "location": event.location, "status": event.status}

    def observe(self) -> Mapping[str, Any]:
        events = sorted((e for e in self.events.values() if e.status != "cancelled"), key=lambda e: _parse_dt(e.start))
        return {"environment": self.name, "step": self.step_count, "upcoming": [self._summary(e) for e in events[:10]], "available_tools": list(self.available_tools)}

    def calendar_lookup(self, start: str | None = None, end: str | None = None, date: str | None = None, include_cancelled: bool = False) -> Mapping[str, Any]:
        if date is not None:
            start_dt = _parse_dt(date)
            end_dt = start_dt + _dt.timedelta(days=1)
        else:
            start_dt = _parse_dt(start) if start is not None else None
            end_dt = _parse_dt(end) if end is not None else None
        selected: list[CalendarEvent] = []
        for event in self.events.values():
            if not include_cancelled and event.status == "cancelled":
                continue
            event_start, event_end = _parse_dt(event.start), _parse_dt(event.end)
            if start_dt is not None and event_end <= start_dt:
                continue
            if end_dt is not None and event_start >= end_dt:
                continue
            selected.append(event)
        selected.sort(key=lambda e: _parse_dt(e.start))
        return {"events": [self._summary(e) for e in selected], "total": len(selected)}

    def calendar_find_conflicts(self, event_id: str | None = None, start: str | None = None, end: str | None = None) -> Mapping[str, Any]:
        if event_id is not None:
            if event_id not in self.events:
                raise ValueError(f"event does not exist: {event_id}")
            candidate = self.events[event_id]
            start_dt, end_dt = _parse_dt(candidate.start), _parse_dt(candidate.end)
            ignore = event_id
        elif start is not None and end is not None:
            start_dt, end_dt, ignore = _parse_dt(start), _parse_dt(end), None
        else:
            raise ValueError("provide event_id or start and end")
        if end_dt <= start_dt:
            raise ValueError("event end must be after start")
        conflicts = []
        for event in self.events.values():
            if event.id == ignore or event.status == "cancelled":
                continue
            a, b = _parse_dt(event.start), _parse_dt(event.end)
            if a < end_dt and b > start_dt:
                conflicts.append(self._summary(event))
        return {"conflicts": conflicts, "has_conflict": bool(conflicts)}

    def calendar_create(self, title: str, start: str, end: str, attendees: Sequence[str] | None = None, location: str = "", description: str = "", event_id: str | None = None, allow_conflict: bool = True) -> Mapping[str, Any]:
        start_dt, end_dt = _parse_dt(start), _parse_dt(end)
        if end_dt <= start_dt:
            raise ValueError("event end must be after start")
        key = event_id or f"e{len(self.events) + 1}"
        while key in self.events:
            key = f"e{len(self.events) + 1}"
        conflict = self.calendar_find_conflicts(start=start, end=end)
        if conflict["has_conflict"] and not allow_conflict:
            raise ValueError("event conflicts with existing event")
        event = CalendarEvent(key, str(title), str(start), str(end), [str(a) for a in (attendees or [])], str(location), str(description))
        self.events[key] = event
        return {"event": self._summary(event), "conflicts": conflict["conflicts"]}

    def calendar_update(self, event_id: str, **changes: Any) -> Mapping[str, Any]:
        if event_id not in self.events:
            raise ValueError(f"event does not exist: {event_id}")
        event = self.events[event_id]
        data = asdict(event)
        for key in ("title", "start", "end", "attendees", "location", "description", "status"):
            if key in changes and changes[key] is not None:
                data[key] = changes[key]
        candidate = CalendarEvent.from_mapping(data, 0)
        if _parse_dt(candidate.end) <= _parse_dt(candidate.start):
            raise ValueError("event end must be after start")
        self.events[event_id] = candidate
        return {"event": self._summary(candidate)}

    def calendar_cancel(self, event_id: str) -> Mapping[str, Any]:
        if event_id not in self.events:
            raise ValueError(f"event does not exist: {event_id}")
        self.events[event_id].status = "cancelled"
        return {"id": event_id, "cancelled": True}

    def ground_truth(self) -> Mapping[str, Any]:
        return {"environment": self.name, "events": [asdict(e) for e in self.events.values()], "step": self.step_count}


# ---------------------------------------------------------------------------
# Virtual web pages


@dataclass
class VirtualWebPage:
    url: str
    title: str
    content: str
    links: dict[str, str] = field(default_factory=dict)
    status: int = 200
    hidden: bool = False

    @classmethod
    def from_mapping(cls, key: str, value: Mapping[str, Any]) -> "VirtualWebPage":
        raw_links = value.get("links", {})
        if isinstance(raw_links, Mapping):
            links = {str(k): str(v) for k, v in raw_links.items()}
        elif isinstance(raw_links, Sequence) and not isinstance(raw_links, (str, bytes)):
            links = {str(item.get("label", item.get("title", item.get("url", "link")))): str(item.get("url", "")) for item in raw_links if isinstance(item, Mapping)}
        else:
            links = {}
        return cls(str(value.get("url", key)), str(value.get("title", "")), str(value.get("content", value.get("body", ""))), links, int(value.get("status", 200)), bool(value.get("hidden", False)))


def _canonical_url(url: str) -> str:
    text = str(url).strip()
    if not text:
        raise ValueError("url is required")
    if "://" not in text:
        text = "https://" + text
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("only virtual http(s) URLs are supported")
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}" + (f"?{parsed.query}" if parsed.query else "")


class VirtualWebPages(BaseEnvironment):
    """Allowlisted web-like pages; no network requests are ever made."""

    name = "virtual_web_pages"
    available_tools = ("browser_open", "browser_search")

    def __init__(self, pages: Mapping[str, Mapping[str, Any] | VirtualWebPage] | Sequence[VirtualWebPage] | None = None) -> None:
        super().__init__()
        self._initial_pages: dict[str, VirtualWebPage] = {}
        if isinstance(pages, Mapping):
            for key, value in pages.items():
                page = value if isinstance(value, VirtualWebPage) else VirtualWebPage.from_mapping(key, value)
                self._initial_pages[_canonical_url(page.url)] = page
        else:
            for index, value in enumerate(pages or ()):
                page = value if isinstance(value, VirtualWebPage) else VirtualWebPage.from_mapping(f"https://page-{index}.test/", value)  # type: ignore[arg-type]
                self._initial_pages[_canonical_url(page.url)] = page
        self.pages: dict[str, VirtualWebPage] = {}
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        initial_state: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        task: Any | None = None,
    ) -> Mapping[str, Any]:
        super().reset(seed)
        payload = _state_mapping(initial_state, state, task)
        raw_pages = payload.get("pages", None)
        if isinstance(raw_pages, Mapping):
            pages: dict[str, VirtualWebPage] = {}
            for key, value in raw_pages.items():
                page = value if isinstance(value, VirtualWebPage) else VirtualWebPage.from_mapping(str(key), value)
                pages[_canonical_url(page.url)] = page
            self.pages = copy.deepcopy(pages)
        else:
            self.pages = copy.deepcopy(self._initial_pages)
        return self.observe()

    def observe(self) -> Mapping[str, Any]:
        return {"environment": self.name, "step": self.step_count, "known_urls": sorted(url for url, page in self.pages.items() if not page.hidden), "available_tools": list(self.available_tools)}

    def browser_open(self, url: str) -> Mapping[str, Any]:
        target = _canonical_url(url)
        if target not in self.pages or self.pages[target].hidden:
            raise ValueError(f"virtual page not found: {target}")
        page = self.pages[target]
        return {"url": target, "status": page.status, "title": page.title, "content": page.content, "links": dict(page.links)}

    def browser_search(self, query: str, limit: int = 10) -> Mapping[str, Any]:
        q = str(query).lower()
        matches = []
        for url, page in self.pages.items():
            if page.hidden:
                continue
            haystack = f"{page.title} {page.content} {' '.join(page.links)}".lower()
            if q in haystack:
                matches.append({"url": url, "title": page.title, "snippet": page.content[:240]})
        return {"query": query, "results": matches[: max(0, int(limit))], "total": len(matches)}

    def ground_truth(self) -> Mapping[str, Any]:
        return {"environment": self.name, "pages": {url: asdict(page) for url, page in self.pages.items()}, "step": self.step_count}


# ---------------------------------------------------------------------------
# Simple grid world


_DIRECTIONS = ("N", "E", "S", "W")
_DELTAS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


def _grid_coord(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        value = value.replace(" ", "").split(",")
    return tuple(map(int, value))  # type: ignore[return-value]


@dataclass
class GridObject:
    id: str
    kind: str
    position: tuple[int, int] | None
    portable: bool = False
    state: dict[str, Any] = field(default_factory=dict)


class SimpleGridWorld(BaseEnvironment):
    """A small partially observable grid with basic object interactions."""

    name = "simple_grid_world"
    available_tools = ("grid_move", "grid_turn", "grid_pick_up", "grid_drop", "grid_use", "grid_interact", "grid_wait")

    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        walls: Iterable[tuple[int, int]] | None = None,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (6, 6),
        objects: Sequence[Mapping[str, Any] | GridObject] | None = None,
        observation_radius: int = 1,
        terrain: Mapping[tuple[int, int], str] | None = None,
        hidden_rules: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if width < 1 or height < 1:
            raise ValueError("grid dimensions must be positive")
        self._initial_width, self._initial_height = int(width), int(height)
        self.width, self.height = self._initial_width, self._initial_height
        self._initial_walls = {_grid_coord(cell) for cell in (walls or ())}
        self._initial_start = _grid_coord(start)
        self._initial_goal = _grid_coord(goal)
        self._initial_terrain = {_grid_coord(k): str(v) for k, v in (terrain or {}).items()}
        self._initial_rules = copy.deepcopy(dict(hidden_rules or {}))
        self.observation_radius = max(0, int(observation_radius))
        self._initial_objects: list[GridObject] = []
        if isinstance(objects, Mapping):
            objects = list(objects.values())  # type: ignore[list-item]
        for idx, obj in enumerate(objects or ()):
            if isinstance(obj, GridObject):
                self._initial_objects.append(copy.deepcopy(obj))
            else:
                pos = obj.get("position", obj.get("pos"))
                position = _grid_coord(pos) if pos is not None else None
                self._initial_objects.append(GridObject(str(obj.get("id", f"o{idx + 1}")), str(obj.get("kind", obj.get("type", "object"))), position, bool(obj.get("portable", False)), copy.deepcopy(dict(obj.get("state", {})))))
        self.reset()

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def reset(
        self,
        seed: int | None = None,
        initial_state: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        task: Any | None = None,
    ) -> Mapping[str, Any]:
        super().reset(seed)
        payload = _state_mapping(initial_state, state, task)
        raw_grid = payload.get("grid")
        self.width = int(payload.get("width", self._initial_width))
        self.height = int(payload.get("height", self._initial_height))
        if self.width < 1 or self.height < 1:
            raise ValueError("grid dimensions must be positive")
        walls_value = payload.get("walls", None)
        if walls_value is None and isinstance(raw_grid, Sequence) and not isinstance(raw_grid, (str, bytes)):
            walls_value = [(x, y) for y, row in enumerate(raw_grid) for x, cell in enumerate(str(row)) if cell == "#"]
        self.walls = {_grid_coord(cell) for cell in (walls_value if walls_value is not None else self._initial_walls)}
        terrain_value = payload.get("terrain", None)
        if isinstance(terrain_value, Mapping):
            self.terrain = {_grid_coord(k): str(v) for k, v in terrain_value.items()}
        else:
            self.terrain = copy.deepcopy(self._initial_terrain)
        start_value = payload.get("start", payload.get("position", self._initial_start))
        goal_value = payload.get("goal", self._initial_goal)
        # Grid rows may be the sole source of S/G coordinates.
        if isinstance(raw_grid, Sequence) and not isinstance(raw_grid, (str, bytes)):
            for y, row in enumerate(raw_grid):
                for x, cell in enumerate(str(row)):
                    if cell == "S" and "start" not in payload and "position" not in payload:
                        start_value = (x, y)
                    elif cell == "G" and "goal" not in payload:
                        goal_value = (x, y)
        self._runtime_start = _grid_coord(start_value)
        self.position = tuple(self._runtime_start)
        self.goal = _grid_coord(goal_value)
        self.orientation = str(payload.get("orientation", "E")).upper()
        if self.orientation not in _DIRECTIONS:
            self.orientation = "E"
        raw_objects = payload.get("objects", None)
        object_values = self._initial_objects if raw_objects is None else (raw_objects if isinstance(raw_objects, Sequence) and not isinstance(raw_objects, (str, bytes)) else ())
        self.objects = {}
        for idx, obj in enumerate(object_values):
            if isinstance(obj, GridObject):
                parsed = copy.deepcopy(obj)
            elif isinstance(obj, Mapping):
                pos = obj.get("position", obj.get("pos"))
                parsed = GridObject(str(obj.get("id", f"o{idx + 1}")), str(obj.get("kind", obj.get("type", "object"))), _grid_coord(pos) if pos is not None else None, bool(obj.get("portable", False)), copy.deepcopy(dict(obj.get("state", {}))))
            else:
                continue
            self.objects[parsed.id] = parsed
        self.inventory: list[str] = [str(item) for item in payload.get("inventory", [])] if isinstance(payload.get("inventory", []), Sequence) and not isinstance(payload.get("inventory", []), (str, bytes)) else []
        self._initial_rules_runtime = copy.deepcopy(payload.get("hidden_rules", self._initial_rules))
        self.teleport_history: list[tuple[int, int]] = []
        self._check_goal()
        return self.observe()

    def _object_at(self, pos: tuple[int, int]) -> list[GridObject]:
        return [o for o in self.objects.values() if o.position == pos]

    def _cell_public(self, pos: tuple[int, int]) -> dict[str, Any]:
        objects = self._object_at(pos)
        terrain = self.terrain.get(pos, "floor")
        if pos == self.goal:
            terrain = "goal" if terrain == "floor" else terrain
        return {"x": pos[0], "y": pos[1], "terrain": "wall" if pos in self.walls else terrain, "objects": [{"id": o.id, "kind": o.kind, "portable": o.portable, "state": copy.deepcopy(o.state)} for o in objects]}

    def observe(self) -> Mapping[str, Any]:
        r = self.observation_radius
        cells = []
        for y in range(max(0, self.position[1] - r), min(self.height, self.position[1] + r + 1)):
            for x in range(max(0, self.position[0] - r), min(self.width, self.position[0] + r + 1)):
                cells.append(self._cell_public((x, y)))
        visible_objects = [obj for cell in cells for obj in cell["objects"]]
        return {
            "environment": self.name,
            "step": self.step_count,
            "position": {"x": self.position[0], "y": self.position[1]},
            "orientation": self.orientation,
            "inventory": list(self.inventory),
            "visible_cells": cells,
            "visible_objects": visible_objects,
            "terrain_here": self.terrain.get(self.position, "goal" if self.position == self.goal else "floor"),
            "in_danger": self.terrain.get(self.position) == "danger",
            "observation_radius": r,
            "goal_reached": self.success,
            "done": self.done,
            "available_tools": list(self.available_tools),
        }

    def _check_goal(self) -> None:
        required_item = self._initial_rules_runtime.get("goal_requires_item") if isinstance(self._initial_rules_runtime, Mapping) else None
        if self.position == self.goal and (required_item is None or str(required_item) in self.inventory):
            self.success = True
            self.done = True

    def grid_move(self, direction: str | None = None, steps: int = 1) -> Mapping[str, Any]:
        if self.done:
            raise ValueError("episode is complete")
        count = int(steps)
        if count < 1 or count > 20:
            raise ValueError("steps must be between 1 and 20")
        direction = (direction or self.orientation).upper()
        relative = {"FORWARD": self.orientation, "LEFT": _DIRECTIONS[(_DIRECTIONS.index(self.orientation) - 1) % 4], "RIGHT": _DIRECTIONS[(_DIRECTIONS.index(self.orientation) + 1) % 4], "BACK": _DIRECTIONS[(_DIRECTIONS.index(self.orientation) + 2) % 4]}
        direction = relative.get(direction, direction)
        if direction not in _DELTAS:
            raise ValueError("direction must be N, E, S, W, forward, left, right, or back")
        moved = 0
        blocked_reason = None
        for _ in range(count):
            dx, dy = _DELTAS[direction]
            candidate = (self.position[0] + dx, self.position[1] + dy)
            if not self._in_bounds(candidate):
                blocked_reason = "boundary"
                break
            if candidate in self.walls:
                blocked_reason = "wall"
                break
            doors = [o for o in self._object_at(candidate) if o.kind == "door" and not o.state.get("open", False)]
            if doors:
                required = [str(o.state.get("key_id", "key")) for o in doors]
                if not any(k in self.inventory for k in required):
                    blocked_reason = "locked_door"
                    break
                for door in doors:
                    door.state["open"] = True
            occupants = [
                o for o in self._object_at(candidate)
                if o.kind in ("npc", "movable", "crate", "box") and not o.state.get("passable", False)
            ]
            if occupants:
                blocked_reason = "occupied"
                break
            self.position = candidate
            moved += 1
            if self.terrain.get(candidate) == "danger":
                self.last_error = "entered danger zone"
            for obj in self._object_at(candidate):
                if obj.kind in ("teleport", "portal"):
                    destination = obj.state.get("destination")
                    if destination is not None:
                        dest = _grid_coord(destination)
                        if self._in_bounds(dest):
                            self.teleport_history.append(self.position)
                            self.position = dest
            self._check_goal()
            if self.done:
                break
        return {"position": {"x": self.position[0], "y": self.position[1]}, "direction": direction, "moved": moved, "blocked": blocked_reason, "done": self.done, "success": self.success}

    def grid_turn(self, direction: str | None = None, degrees: int | None = None) -> Mapping[str, Any]:
        if degrees is not None:
            quarter_turns = round(int(degrees) / 90)
            self.orientation = _DIRECTIONS[(_DIRECTIONS.index(self.orientation) + quarter_turns) % 4]
        elif direction:
            value = direction.upper()
            if value in _DIRECTIONS:
                self.orientation = value
            elif value in ("LEFT", "RIGHT", "BACK"):
                delta = {"LEFT": -1, "RIGHT": 1, "BACK": 2}[value]
                self.orientation = _DIRECTIONS[(_DIRECTIONS.index(self.orientation) + delta) % 4]
            else:
                raise ValueError("direction must be cardinal or left/right/back")
        else:
            raise ValueError("direction or degrees is required")
        return {"orientation": self.orientation}

    def grid_pick_up(self, object_id: str | None = None) -> Mapping[str, Any]:
        candidates = [o for o in self._object_at(self.position) if o.portable and o.id not in self.inventory]
        if object_id is not None:
            candidates = [o for o in candidates if o.id == object_id]
        if not candidates:
            raise ValueError("no portable object available here")
        obj = candidates[0]
        self.inventory.append(obj.id)
        obj.position = None
        return {"picked_up": obj.id, "inventory": list(self.inventory)}

    def grid_drop(self, object_id: str | None = None) -> Mapping[str, Any]:
        if not self.inventory:
            raise ValueError("inventory is empty")
        key = object_id or self.inventory[-1]
        if key not in self.inventory or key not in self.objects:
            raise ValueError(f"object not in inventory: {key}")
        self.inventory.remove(key)
        self.objects[key].position = self.position
        return {"dropped": key, "position": {"x": self.position[0], "y": self.position[1]}, "inventory": list(self.inventory)}

    def grid_use(self, object_id: str | None = None, target_id: str | None = None) -> Mapping[str, Any]:
        key = object_id or (self.inventory[-1] if self.inventory else None)
        if key is None or key not in self.inventory:
            raise ValueError("specify an object in inventory")
        obj = self.objects[key]
        target = self.objects.get(target_id) if target_id else None
        if target and target.kind == "door" and key == str(target.state.get("key_id", key)):
            target.state["open"] = True
            return {"used": key, "target": target.id, "opened": True}
        if obj.kind in ("food", "consumable"):
            self.inventory.remove(key)
            del self.objects[key]
            return {"used": key, "consumed": True, "inventory": list(self.inventory)}
        return {"used": key, "target": target_id, "effect": "no_effect"}

    def grid_interact(self, object_id: str | None = None) -> Mapping[str, Any]:
        nearby = self._object_at(self.position)
        if object_id:
            nearby = [o for o in nearby if o.id == object_id]
        # A movable object can be pushed from the cell in front of the agent.
        # This is a deterministic interaction rather than a physics engine.
        if not nearby:
            dx, dy = _DELTAS[self.orientation]
            ahead = (self.position[0] + dx, self.position[1] + dy)
            nearby = [
                o for o in self._object_at(ahead)
                if o.kind in ("movable", "crate", "box") and (object_id is None or o.id == object_id)
            ]
            if nearby:
                obj = nearby[0]
                destination = (ahead[0] + dx, ahead[1] + dy)
                occupants = self._object_at(destination)
                # Portals/teleport pads are floor fixtures and may share a
                # cell with a pushed object; solid actors and other movable
                # objects still block the push.
                blocking = [o for o in occupants if o.kind not in ("teleport", "portal") and not o.state.get("passable", False)]
                if not self._in_bounds(destination) or destination in self.walls or blocking:
                    raise ValueError("movable object is blocked")
                obj.position = destination
                return {
                    "object": obj.id,
                    "kind": obj.kind,
                    "moved_to": {"x": destination[0], "y": destination[1]},
                    "state": copy.deepcopy(obj.state),
                    "done": self.done,
                }
        if not nearby:
            raise ValueError("no interactable object here")
        obj = nearby[0]
        if obj.kind == "goal":
            self._check_goal()
        if obj.kind == "switch":
            obj.state["activated"] = True
        return {"object": obj.id, "kind": obj.kind, "state": copy.deepcopy(obj.state), "done": self.done}

    def grid_wait(self, ticks: int = 1) -> Mapping[str, Any]:
        ticks = int(ticks)
        if ticks < 1 or ticks > 100:
            raise ValueError("ticks must be between 1 and 100")
        return {"waited": ticks, "position": {"x": self.position[0], "y": self.position[1]}}

    def ground_truth(self) -> Mapping[str, Any]:
        return {
            "environment": self.name,
            "width": self.width,
            "height": self.height,
            "walls": sorted([list(w) for w in self.walls]),
            "position": list(self.position),
            "goal": list(self.goal),
            "orientation": self.orientation,
            "terrain": {f"{x},{y}": value for (x, y), value in self.terrain.items()},
            "objects": {key: {"id": obj.id, "kind": obj.kind, "position": list(obj.position) if obj.position is not None else None, "portable": obj.portable, "state": copy.deepcopy(obj.state)} for key, obj in self.objects.items()},
            "inventory": list(self.inventory),
            "hidden_rules": copy.deepcopy(getattr(self, "_initial_rules_runtime", self._initial_rules)),
            "step": self.step_count,
            "done": self.done,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Code sandbox mock


class CodeSandboxMock(BaseEnvironment):
    """A non-executing code environment.

    ``execute_mock_code`` performs static, deterministic simulation only.  It
    never calls ``eval``, ``exec``, a shell, or a host interpreter.  A task can
    provide expected test markers to model a tiny repair workflow safely.
    """

    name = "code_sandbox_mock"
    available_tools = ("sandbox_list_files", "sandbox_read_file", "sandbox_write_file", "execute_mock_code", "run_mock_tests")

    def __init__(self, files: Mapping[str, str] | None = None, tests: Sequence[Mapping[str, Any]] | None = None) -> None:
        super().__init__()
        self._initial_files = _coerce_file_mapping(files or {})
        self._initial_tests = copy.deepcopy(list(tests or ()))
        self.files: dict[str, str] = {}
        self.tests: list[dict[str, Any]] = []
        self.reset()

    def reset(
        self,
        seed: int | None = None,
        initial_state: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        task: Any | None = None,
    ) -> Mapping[str, Any]:
        super().reset(seed)
        payload = _state_mapping(initial_state, state, task)
        raw_files = payload.get("files", self._initial_files)
        self.files = {_safe_path(k): str(v) for k, v in raw_files.items()} if isinstance(raw_files, Mapping) else copy.deepcopy(self._initial_files)
        self.tests = copy.deepcopy(payload.get("tests", self._initial_tests)) if isinstance(payload.get("tests", self._initial_tests), Sequence) and not isinstance(payload.get("tests", self._initial_tests), (str, bytes)) else copy.deepcopy(self._initial_tests)
        return self.observe()

    def observe(self) -> Mapping[str, Any]:
        return {"environment": self.name, "step": self.step_count, "files": sorted(self.files), "test_count": len(self.tests), "available_tools": list(self.available_tools)}

    def sandbox_list_files(self) -> Mapping[str, Any]:
        return {"files": [{"path": p, "size": len(c.encode("utf-8"))} for p, c in sorted(self.files.items())]}

    def sandbox_read_file(self, path: str) -> Mapping[str, Any]:
        target = _safe_path(path)
        if target not in self.files:
            raise ValueError(f"file does not exist: {target}")
        return {"path": target, "content": self.files[target]}

    def sandbox_write_file(self, path: str, content: str) -> Mapping[str, Any]:
        target = _safe_path(path)
        self.files[target] = str(content)
        return {"path": target, "written": True}

    @staticmethod
    def _static_syntax_check(code: str) -> tuple[bool, str | None]:
        # ``ast.parse`` is a parser only; it does not execute user code.
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    def execute_mock_code(self, code: str | None = None, path: str | None = None, language: str = "python") -> Mapping[str, Any]:
        if path is not None:
            target = _safe_path(path)
            if target not in self.files:
                raise ValueError(f"file does not exist: {target}")
            code = self.files[target]
        if code is None:
            raise ValueError("code or path is required")
        source = str(code)
        if len(source) > 200_000:
            return {"status": "failed", "language": language, "stdout": "", "stderr": "source exceeds mock sandbox limit"}
        if language.lower() not in ("python", "py", "javascript", "js", "text"):
            return {"status": "unsupported", "language": language, "stdout": "", "stderr": "unsupported mock language"}
        if language.lower() in ("python", "py"):
            valid, error = self._static_syntax_check(source)
            if not valid:
                return {"status": "failed", "language": language, "stdout": "", "stderr": error}
        if re.search(r"\b(raise|throw|panic)\b", source):
            return {"status": "failed", "language": language, "stdout": "", "stderr": "simulated runtime error"}
        outputs = re.findall(r"(?:print|console\.log)\s*\(\s*(['\"])(.*?)\1\s*\)", source, flags=re.DOTALL)
        stdout = "\n".join(match[1] for match in outputs)
        return {"status": "passed", "language": language, "stdout": stdout, "stderr": "", "executed": False, "note": "mock execution; source was not run"}

    def run_mock_tests(self, path: str | None = None, code: str | None = None) -> Mapping[str, Any]:
        if path is not None:
            source = self.sandbox_read_file(path)["content"]
        else:
            source = str(code) if code is not None else "\n".join(self.files.values())
        run = self.execute_mock_code(code=source)
        if run["status"] != "passed":
            return {"status": "failed", "passed": 0, "failed": len(self.tests) or 1, "tests": [], "execution": run}
        results = []
        for index, test in enumerate(self.tests):
            name = str(test.get("name", f"test_{index + 1}"))
            contains = test.get("contains")
            not_contains = test.get("not_contains")
            required = test.get("required_text")
            ok = True
            reason = "passed"
            if contains is not None and str(contains) not in source:
                ok, reason = False, "required text missing"
            if required is not None and str(required) not in source:
                ok, reason = False, "required text missing"
            if not_contains is not None and str(not_contains) in source:
                ok, reason = False, "forbidden text present"
            results.append({"name": name, "passed": ok, "reason": reason})
        failed = sum(1 for item in results if not item["passed"])
        return {"status": "passed" if failed == 0 else "failed", "passed": len(results) - failed, "failed": failed, "tests": results, "execution": run}

    def ground_truth(self) -> Mapping[str, Any]:
        return {"environment": self.name, "files": copy.deepcopy(self.files), "tests": copy.deepcopy(self.tests), "step": self.step_count}


# ---------------------------------------------------------------------------
# Convenience factories


ENVIRONMENT_TYPES: dict[str, type[BaseEnvironment]] = {
    "virtual_file_system": VirtualFileSystem,
    "virtual_filesystem": VirtualFileSystem,
    "vfs": VirtualFileSystem,
    "virtual_email_inbox": VirtualEmailInbox,
    "virtual_email": VirtualEmailInbox,
    "email": VirtualEmailInbox,
    "virtual_calendar": VirtualCalendar,
    "calendar": VirtualCalendar,
    "virtual_web_pages": VirtualWebPages,
    "virtual_web": VirtualWebPages,
    "web": VirtualWebPages,
    "simple_grid_world": SimpleGridWorld,
    "simple_grid": SimpleGridWorld,
    "grid": SimpleGridWorld,
    "code_sandbox_mock": CodeSandboxMock,
    "code_sandbox": CodeSandboxMock,
    "code": CodeSandboxMock,
}


def make_environment(kind: str, **kwargs: Any) -> BaseEnvironment:
    """Create a safe environment by stable name or short alias."""

    key = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ENVIRONMENT_TYPES:
        raise ValueError(f"unknown environment: {kind}")
    return ENVIRONMENT_TYPES[key](**kwargs)


__all__ = [
    "BaseEnvironment",
    "Environment",
    "EnvironmentProtocol",
    "EmailMessage",
    "CalendarEvent",
    "VirtualFileSystem",
    "VirtualEmailInbox",
    "VirtualCalendar",
    "VirtualWebPage",
    "VirtualWebPages",
    "GridObject",
    "SimpleGridWorld",
    "CodeSandboxMock",
    "ENVIRONMENT_TYPES",
    "make_environment",
]

# Short name used by a few provider adapters and external task authors.
Environment = BaseEnvironment
