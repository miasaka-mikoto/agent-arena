"""Bridge generated task states to the safe in-memory environments.

This module is intentionally a tiny adapter: task generation stays pure and
the environments remain reusable for hand-authored scenarios.  It never
touches host files, network services, or real accounts.
"""

from __future__ import annotations

from typing import Any, Mapping

from .tasking import TaskDefinition


def environment_spec(task: TaskDefinition | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(environment_kind, constructor_kwargs)`` for a task."""

    state = task.initial_state if isinstance(task, TaskDefinition) else dict(task.get("initial_state", {}))
    kind = str(state.get("type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if kind == "virtual_file_system":
        return kind, {"files": state.get("files", {}), "directories": state.get("directories", [])}
    if kind == "virtual_email_inbox":
        return kind, {"messages": state.get("messages", [])}
    if kind == "virtual_calendar":
        return kind, {"events": state.get("events", [])}
    if kind == "virtual_web_pages":
        return kind, {"pages": state.get("pages", {})}
    if kind == "code_sandbox_mock":
        return kind, {"files": state.get("files", {}), "tests": state.get("tests", [])}
    if kind == "simple_grid_world":
        walls = [tuple(cell) for cell in state.get("walls", [])]
        raw_terrain = state.get("terrain", {})
        terrain = {}
        if isinstance(raw_terrain, Mapping):
            for key, value in raw_terrain.items():
                if isinstance(key, str) and "," in key:
                    coords = tuple(int(part.strip()) for part in key.split(",", 1))
                else:
                    coords = tuple(map(int, key)) if isinstance(key, (list, tuple)) else None
                if coords is not None and len(coords) == 2:
                    terrain[coords] = value
        return kind, {
            "width": int(state.get("width", 1)),
            "height": int(state.get("height", 1)),
            "walls": walls,
            "start": tuple(state.get("start", state.get("position", (0, 0)))),
            "goal": tuple(state.get("goal", (0, 0))),
            "objects": state.get("objects", []),
            "observation_radius": int(state.get("observation_radius", state.get("visible_radius", 1))),
            "terrain": terrain,
            "hidden_rules": state.get("hidden_rules", {}),
        }
    # Allow callers to pass an environment alias understood by make_environment
    # while still failing loudly for malformed task state.
    if kind:
        return kind, {}
    raise ValueError("task initial_state must contain an environment 'type'")


def make_environment_for_task(task: TaskDefinition | Mapping[str, Any]) -> Any:
    """Instantiate the corresponding safe environment for a task."""

    from .environments import make_environment

    kind, kwargs = environment_spec(task)
    return make_environment(kind, **kwargs)


__all__ = ["environment_spec", "make_environment_for_task"]
