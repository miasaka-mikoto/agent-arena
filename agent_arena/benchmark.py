"""End-to-end benchmark assembly for the built-in Agent Arena tasks.

This module is the small integration layer between serialisable task
definitions and the six safe simulator classes.  It deliberately contains no
host I/O, network access, model SDK, or shell execution.  The adapter exposes
only the tools listed by the task and keeps the evaluator-only goal in the
runner process.
"""

from __future__ import annotations

import copy
import re
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from .core import Action, MockLLMAgent, RandomAgent, RuleBasedAgent, ScriptedAgent, ToolResult
from .environments import (
    BaseEnvironment,
    CodeSandboxMock,
    SimpleGridWorld,
    VirtualCalendar,
    VirtualEmailInbox,
    VirtualFileSystem,
    VirtualWebPages,
)
from .tasking import TaskDefinition


_GRID_ALIASES = {
    "move": "grid_move",
    "turn": "grid_turn",
    "pick_up": "grid_pick_up",
    "drop": "grid_drop",
    "use": "grid_use",
    "interact": "grid_interact",
    "wait": "grid_wait",
}


def _time_value(date: str | None, value: Any) -> str:
    text = str(value or "")
    if "T" in text or re.match(r"^\d{4}-\d{2}-\d{2} ", text):
        return text.replace(" ", "T", 1)
    if date and re.match(r"^\d{1,2}:\d{2}$", text):
        return f"{date}T{int(text[:2]):02d}:{text[3:]}"
    return text


def _grid_inner(state: Mapping[str, Any]) -> SimpleGridWorld:
    rows = [str(row) for row in state.get("grid", [])]
    if not rows:
        rows = ["..."]
    height = len(rows)
    width = max(len(row) for row in rows)
    walls: set[tuple[int, int]] = set()
    start = tuple(state.get("position", [0, 0]))
    goal = (width - 1, height - 1)
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == "#":
                walls.add((x, y))
            elif cell == "S":
                start = (x, y)
            elif cell == "G":
                goal = (x, y)
    objects = list(state.get("objects", []) or [])
    if state.get("key") is not None:
        objects.append({"id": "key", "kind": "key", "position": state["key"], "portable": True})
    return SimpleGridWorld(
        width=width,
        height=height,
        walls=walls,
        start=(int(start[0]), int(start[1])),
        goal=(int(goal[0]), int(goal[1])),
        objects=objects,
        observation_radius=int(state.get("visible_radius", state.get("observation_radius", 1))),
        terrain=state.get("terrain", {}),
        hidden_rules=state.get("hidden_rules", {}),
    )


def _calendar_inner(state: Mapping[str, Any]) -> VirtualCalendar:
    date_default = str(state.get("date", "2026-01-01"))
    events = []
    for event in state.get("events", []) or []:
        item = dict(event)
        date = str(item.pop("date", date_default))
        item["start"] = _time_value(date, item.get("start"))
        item["end"] = _time_value(date, item.get("end"))
        events.append(item)
    return VirtualCalendar(events=events)


def _code_inner(state: Mapping[str, Any]) -> CodeSandboxMock:
    files = dict(state.get("files", {}) or {})
    tests = list(state.get("tests", []) or [])
    command = state.get("test_command")
    # The mock runner never executes a command.  It checks deterministic text
    # markers instead, which is enough to exercise read → edit → verify.
    if command and not tests:
        tests = [{"name": "provided_check", "contains": "def "}]
    return CodeSandboxMock(files=files, tests=tests)


def _inner_for_task(task: TaskDefinition) -> BaseEnvironment:
    state = task.initial_state or {}
    kind = str(state.get("type", task.category)).lower()
    if kind in {"virtual_file_system", "file_organization", "files", "file"}:
        return VirtualFileSystem(files=state.get("files", {}), directories=state.get("directories", []))
    if kind in {"virtual_email_inbox", "email", "email_retrieval", "mail"}:
        return VirtualEmailInbox(messages=state.get("messages", []))
    if kind in {"virtual_calendar", "calendar", "calendar_scheduling", "schedule"}:
        return _calendar_inner(state)
    if kind in {"virtual_web_pages", "web", "browser"}:
        return VirtualWebPages(pages=state.get("pages", {}))
    if kind in {"simple_grid_world", "grid", "grid_navigation", "navigation"}:
        return _grid_inner(state)
    if kind in {"code_sandbox_mock", "code", "code_repair"}:
        return _code_inner(state)
    raise ValueError(f"unsupported task environment type: {kind}")


class TaskEnvironment:
    """Bind one task to one isolated simulated environment.

    The wrapper enforces the task allow-list, translates friendly tool names,
    injects deterministic adversarial faults, and provides an evaluator-only
    ``check_success`` method.  Its ``observe`` method never returns the full
    initial state or hidden ground truth.
    """

    def __init__(self, task: TaskDefinition, *, seed: int | None = None):
        self.task = task
        self.seed = seed if seed is not None else task.seed
        self.inner = _inner_for_task(task)
        self.available_tools = tuple(task.allowed_tools)
        self.name = getattr(self.inner, "name", task.category)
        self.done = False
        self.success = False
        self.invalid_actions = 0
        self.last_error: str | None = None
        self._calls: dict[str, int] = {}
        self._read_ids: set[str] = set()
        self._executed_code = False
        self._faults = copy.deepcopy((task.initial_state or {}).get("faults", {}))

    def reset(self, task: TaskDefinition | None = None, seed: int | None = None, initial_state: Mapping[str, Any] | None = None, **_: Any) -> Mapping[str, Any]:
        if task is not None and task is not self.task:
            self.task = task
            self.inner = _inner_for_task(task)
            self.available_tools = tuple(task.allowed_tools)
            self._faults = copy.deepcopy((task.initial_state or {}).get("faults", {}))
        if seed is not None:
            self.seed = seed
        self.inner.reset(self.seed)
        self.done = False
        self.success = False
        self.invalid_actions = 0
        self.last_error = None
        self._calls.clear()
        self._read_ids.clear()
        self._executed_code = False
        return self.observe()

    def observe(self) -> Mapping[str, Any]:
        state = dict(self.inner.observe())
        # The task allow-list is authoritative.  Do not leak methods that the
        # current task did not permit, even if the underlying environment has
        # more capabilities.
        state["available_tools"] = list(self.available_tools)
        state["task_category"] = self.task.category
        state["done"] = bool(self.done or state.get("done", False))
        return state

    def snapshot(self) -> Mapping[str, Any]:
        return dict(self.observe())

    def ground_truth(self) -> Mapping[str, Any]:
        return dict(self.inner.ground_truth())

    def _fault_result(self, name: str, *, empty: bool = False) -> ToolResult:
        if empty:
            output: Any = {"query": "", "messages": [], "matches": [], "results": [], "total": 0}
            return ToolResult(tool=name, success=True, output=output, metadata={"adversarial": "empty_result"})
        return ToolResult(tool=name, success=False, error="simulated transient tool failure", metadata={"adversarial": "failed_tool"})

    def _translate(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        target = _GRID_ALIASES.get(name, name)
        values = dict(args)
        if target == "read_mail" and "message_id" not in values and "id" in values:
            values["message_id"] = values.pop("id")
        if target == "mark_mail_read" and "message_id" not in values and "id" in values:
            values["message_id"] = values.pop("id")
        if target in {"sandbox_read_file", "sandbox_write_file"} and "path" not in values and "file" in values:
            values["path"] = values.pop("file")
        if target == "read_file" and isinstance(self.inner, CodeSandboxMock):
            target = "sandbox_read_file"
        if target == "write_file" and isinstance(self.inner, CodeSandboxMock):
            target = "sandbox_write_file"
        if target == "execute_mock_code" and isinstance(self.inner, CodeSandboxMock):
            # ``path`` is accepted by the mock executor directly.
            pass
        if isinstance(self.inner, VirtualCalendar):
            if target == "calendar_lookup" and values.get("date"):
                values["date"] = str(values["date"])
            if target == "calendar_create":
                date = values.pop("date", None)
                if date:
                    values["start"] = _time_value(str(date), values.get("start"))
                    values["end"] = _time_value(str(date), values.get("end"))
        return target, values

    def execute_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        name = str(name)
        args = dict(arguments or {})
        self._calls[name] = self._calls.get(name, 0) + 1
        if name not in self.available_tools:
            self.invalid_actions += 1
            self.last_error = f"tool not allowed for task: {name}"
            return ToolResult(tool=name, success=False, error=self.last_error)
        fault_tool = str(self._faults.get("tool", ""))
        if self._calls[name] == 1 and fault_tool == name and self._faults.get("fail_once"):
            self.invalid_actions += 1
            self.last_error = "simulated transient tool failure"
            return self._fault_result(name)
        if self._calls[name] == 1 and fault_tool == name and self._faults.get("empty_once"):
            return self._fault_result(name, empty=True)
        target, translated = self._translate(name, args)
        result = self.inner.execute_tool(target, translated)
        if not isinstance(result, ToolResult):
            result = ToolResult.from_any(result, tool=name)
        else:
            # Keep the public task-facing name in traces, not the internal
            # alias used by the adapter.
            result.tool = name
        if result.success:
            if name == "read_mail":
                message_id = translated.get("message_id", translated.get("id"))
                if message_id:
                    self._read_ids.add(str(message_id))
            if name in {"execute_mock_code", "run_mock_tests"}:
                self._executed_code = True
        else:
            self.invalid_actions += 1
            self.last_error = result.error
        self.success = self.check_success(self.task)
        self.done = self.success
        return result

    def execute_action(self, action: Any) -> ToolResult:
        if isinstance(action, Mapping):
            name = action.get("tool") or action.get("name") or action.get("action")
            args = action.get("arguments") or action.get("args") or {}
        else:
            name = getattr(action, "tool", None) or getattr(action, "name", None)
            args = getattr(action, "arguments", None) or getattr(action, "args", None) or {}
        return self.execute_tool(str(name or ""), args)

    def check_success(self, task: TaskDefinition | None = None) -> bool:
        task = task or self.task
        category = task.category
        truth = task.hidden_ground_truth or {}
        if category == "file_organization" and isinstance(self.inner, VirtualFileSystem):
            for move in truth.get("moves", []):
                source, dest = str(move.get("from")), str(move.get("to"))
                if source in self.inner.files or dest not in self.inner.files:
                    return False
            return True
        if category == "email_retrieval" and isinstance(self.inner, VirtualEmailInbox):
            target = str(truth.get("target_message_id", ""))
            message = self.inner.messages.get(target)
            return bool(message and not message.unread and target in self._read_ids)
        if category == "calendar_scheduling" and isinstance(self.inner, VirtualCalendar):
            proposed = truth.get("proposed_event", {})
            start = _time_value(str(proposed.get("date", "")), proposed.get("start"))
            end = _time_value(str(proposed.get("date", "")), proposed.get("end"))
            matches = [e for e in self.inner.events.values() if e.title.lower() == str(proposed.get("title", "")).lower() and e.start == start and e.end == end and e.status != "cancelled"]
            if not matches:
                return False
            # Ask about the created event itself so its own interval is not
            # mistaken for an overlap.  This is the same semantics exposed to
            # agents through calendar_find_conflicts(event_id=...).
            return not self.inner.calendar_find_conflicts(event_id=matches[0].id)["has_conflict"]
        if category == "code_repair" and isinstance(self.inner, CodeSandboxMock):
            expected = str(truth.get("expected_file", ""))
            actual = self.inner.files.get("/workspace/main.py", "")
            return bool(expected and actual == expected and self._executed_code)
        if category == "grid_navigation" and isinstance(self.inner, SimpleGridWorld):
            return bool(self.inner.success and tuple(self.inner.position) == tuple(self.inner.goal))
        return bool(getattr(self.inner, "success", False))


def make_task_environment(task: TaskDefinition | Mapping[str, Any], *, seed: int | None = None) -> TaskEnvironment:
    if not isinstance(task, TaskDefinition):
        task = TaskDefinition.from_dict(task)
    return TaskEnvironment(task, seed=seed)


def scripted_actions_for_task(task: TaskDefinition) -> list[Action]:
    """Build a transparent, deterministic script for the scripted baseline."""

    truth = task.hidden_ground_truth or {}
    category = task.category
    if category == "file_organization":
        actions = [Action("list_files", {"path": "/inbox", "recursive": True}, reason="Inspect the virtual inbox.")]
        actions.extend(Action("move_file", {"source": m["from"], "destination": m["to"]}, reason="Apply the task's routing rule.") for m in truth.get("moves", []))
        return _with_fault_recovery(task, actions)
    if category == "email_retrieval":
        return _with_fault_recovery(task, [Action("search_mail", {"query": "Action required"}, reason="Search for the authoritative security message."), Action("read_mail", {"message_id": truth.get("target_message_id", ""), "mark_read": True}, reason="Read the target message.")])
    if category == "calendar_scheduling":
        event = truth.get("proposed_event", {})
        return _with_fault_recovery(task, [Action("calendar_lookup", {"date": event.get("date")}, reason="Check for conflicts first."), Action("calendar_create", {"title": event.get("title"), "date": event.get("date"), "start": event.get("start"), "end": event.get("end"), "allow_conflict": False}, reason="Create the conflict-free event.")])
    if category == "code_repair":
        expected = truth.get("expected_file", "")
        return _with_fault_recovery(task, [Action("sandbox_read_file", {"path": "/workspace/main.py"}, reason="Inspect the buggy source."), Action("sandbox_write_file", {"path": "/workspace/main.py", "content": expected}, reason="Write the minimal repair."), Action("execute_mock_code", {"path": "/workspace/main.py"}, reason="Run the safe mock check.")])
    if category == "grid_navigation":
        return _with_fault_recovery(task, [Action("grid_move", {"direction": str(direction)[0].upper()}, reason="Follow the deterministic route.") for direction in truth.get("shortest_path", [])])
    return [Action("wait", {}, reason="No-op scripted action.")]


def _with_fault_recovery(task: TaskDefinition, actions: list[Action]) -> list[Action]:
    """Make the transparent scripted baseline recover from one injected fault.

    Adversarial tasks intentionally fail or empty one public tool call.  A
    scripted reference should still demonstrate a retry, otherwise the demo
    would conflate "script has no recovery" with an environment bug.
    """

    state = task.initial_state or {}
    faults = state.get("faults", {}) if isinstance(state, Mapping) else {}
    if not isinstance(faults, Mapping) or not (faults.get("fail_once") or faults.get("empty_once")) or not actions:
        return actions
    fault_tool = str(faults.get("tool", ""))
    index = next((i for i, action in enumerate(actions) if action.tool == fault_tool), 0)
    # Deep-copy the action so mutable argument dictionaries do not alias the
    # original script entry during a replay.
    retry = copy.deepcopy(actions[index])
    return [*actions[: index + 1], retry, *actions[index + 1 :]]


class ArenaRuleBasedAgent(RuleBasedAgent):
    """RuleBasedAgent with small task-specific public-state policies.

    The generic core policy remains useful for custom tasks.  This subclass is
    used by the demo so the reference benchmark measures meaningful recovery
    and tool selection rather than argument plumbing failures.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("name", "RuleBasedAgent")
        super().__init__(*args, **kwargs)
        self._queue: list[Action] = []
        self._category_phase = 0

    def reset(self, *args: Any, **kwargs: Any) -> None:
        super().reset(*args, **kwargs)
        self._queue = []
        self._category_phase = 0

    def record_result(self, result: Any, *, action: Action | Mapping[str, Any] | None = None, **kwargs: Any) -> ToolResult:
        converted = super().record_result(result, action=action, **kwargs)
        self._queue.extend(self._follow_up_actions(converted, action))
        return converted

    def act(self, observation: Any = None, **kwargs: Any) -> Action | None:
        if observation is not None:
            self.observe(observation)
        if self._queue:
            return self._queue.pop(0)
        task = self.task
        category = getattr(task, "category", "") if task is not None else ""
        if category == "file_organization" and not self.history:
            return Action("list_files", {"path": "/inbox", "recursive": True}, reason="Inspect the inbox before moving files.")
        if category == "email_retrieval" and not self.history:
            return Action("search_mail", {"query": "Action required"}, reason="Search the public inbox for the relevant message.")
        if category == "calendar_scheduling" and not self.history:
            event = (task.hidden_ground_truth if hasattr(task, "hidden_ground_truth") else {}) or {}
            # The public task view does not carry hidden truth; parse the
            # instruction instead.
            text = getattr(task, "instruction", "")
            date = re.search(r"20\d{2}-\d{2}-\d{2}", text)
            # ISO timestamps contain ``T11:00`` (no word boundary before the
            # hour), so use a digit look-behind rather than ``\b``.
            times = re.findall(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d", text)
            return Action("calendar_lookup", {"date": date.group(0) if date else None}, reason="Check the calendar before creating an event.")
        if category == "code_repair" and not self.history:
            return Action("sandbox_read_file", {"path": "/workspace/main.py"}, reason="Read the source before editing.")
        return super().act(observation, **kwargs)

    def _follow_up_actions(self, result: ToolResult, action: Any) -> list[Action]:
        task = self.task
        category = getattr(task, "category", "") if task is not None else ""
        output = result.output if isinstance(result, ToolResult) else None
        if not result.success:
            # A failed call is an observable recovery opportunity.  Repeat the
            # same safe query once for transient adversarial failures.
            if action is not None:
                normalized = Action.from_any(action)
                if normalized and normalized.tool:
                    return [normalized]
            return []
        if category == "file_organization" and getattr(action, "tool", None) == "list_files":
            entries = (output or {}).get("entries", []) if isinstance(output, Mapping) else []
            moves = []
            for entry in entries:
                path = str(entry.get("path", "")) if isinstance(entry, Mapping) else ""
                if path and path.startswith("/inbox/"):
                    name = PurePosixPath(path).name.lower()
                    folder = "/archive" if ("report" in name or "invoice" in name) else "/notes"
                    moves.append(Action("move_file", {"source": path, "destination": f"{folder}/{PurePosixPath(path).name}"}, reason="Route the file by its public name."))
            return moves
        if category == "email_retrieval" and getattr(action, "tool", None) == "search_mail":
            messages = (output or {}).get("messages", []) if isinstance(output, Mapping) else []
            target = next((m for m in messages if isinstance(m, Mapping) and "security" in str(m.get("from", ""))), None)
            if target:
                return [Action("read_mail", {"message_id": target.get("id"), "mark_read": True}, reason="Read the matching security message.")]
            return [Action("search_mail", {"query": "security"}, reason="Refine the search after an empty result.")]
        if category == "calendar_scheduling" and getattr(action, "tool", None) == "calendar_lookup":
            text = getattr(task, "instruction", "")
            date = re.search(r"20\d{2}-\d{2}-\d{2}", text)
            times = re.findall(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d", text)
            title_match = re.search(r"schedule a (.+?) on ", text, re.I)
            return [Action("calendar_create", {"title": (title_match.group(1) if title_match else "Synthetic planning meeting"), "date": date.group(0) if date else None, "start": times[0] if times else "10:00", "end": times[1] if len(times) > 1 else "11:00", "allow_conflict": False}, reason="Create the verified conflict-free event.")]
        if category == "code_repair" and getattr(action, "tool", None) == "sandbox_read_file":
            source = (output or {}).get("content", "") if isinstance(output, Mapping) else ""
            fixed = source.replace("return a - b", "return a + b").replace("return n % 2 == 1", "return n % 2 == 0").replace("max(hi, min(lo, x))", "max(lo, min(hi, x))")
            return [Action("sandbox_write_file", {"path": "/workspace/main.py", "content": fixed}, reason="Write the minimal deterministic fix.")]
        if category == "code_repair" and getattr(action, "tool", None) == "sandbox_write_file":
            return [Action("execute_mock_code", {"path": "/workspace/main.py"}, reason="Run the safe mock code check.")]
        return []


def default_agent_factories() -> dict[str, Callable[[int], Any]]:
    """Factories used by the standard demo tournament."""

    return {
        "RandomAgent": lambda seed: RandomAgent(name="RandomAgent", seed=seed),
        "RuleBasedAgent": lambda seed: ArenaRuleBasedAgent(name="RuleBasedAgent", seed=seed),
        "MockLLMAgent": lambda seed: MockLLMAgent(name="MockLLMAgent", seed=seed),
    }


__all__ = [
    "TaskEnvironment",
    "make_task_environment",
    "scripted_actions_for_task",
    "ArenaRuleBasedAgent",
    "default_agent_factories",
]
