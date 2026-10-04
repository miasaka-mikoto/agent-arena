"""Contract tests for the offline Arena environments and tool boundary."""

from __future__ import annotations

from agent_arena.environments import (
    CodeSandboxMock,
    SimpleGridWorld,
    VirtualCalendar,
    VirtualEmailInbox,
    VirtualFileSystem,
    VirtualWebPages,
    make_environment,
)
from agent_arena.tools import EnvironmentBundle, build_default_tool_registry
from agent_arena.task_catalog import build_task
from agent_arena.task_environment import make_environment_for_task


def test_virtual_file_system_is_in_memory_and_blocks_traversal() -> None:
    env = VirtualFileSystem({"/docs/a.txt": "alpha", "/notes.txt": "beta"})
    assert env.execute_tool("list_files", {"path": "/"}).success
    result = env.execute_tool("read_file", {"path": "/docs/../notes.txt"})
    assert not result.success
    assert "traversal" in (result.error or "")
    env.execute_tool("move_file", {"source": "/notes.txt", "destination": "/docs"})
    assert env.execute_tool("read_file", {"file_path": "/docs/a.txt"}).success
    assert env.files == {"/docs/a.txt": "alpha", "/docs/notes.txt": "beta"}
    assert env.ground_truth()["files"]["/docs/notes.txt"] == "beta"


def test_virtual_email_search_does_not_expose_body_in_summary() -> None:
    env = VirtualEmailInbox(
        [
            {
                "id": "m1",
                "from": "alice@example.test",
                "subject": "Meeting",
                "body": "bring the blue folder",
                "timestamp": "2026-01-02T09:00:00",
            }
        ]
    )
    observation = env.observe()
    assert "body" not in observation["messages"][0]
    hit = env.execute_tool("search_mail", {"query": "blue folder"})
    assert hit.success and hit.output["messages"][0]["id"] == "m1"
    message = env.execute_tool("read_mail", {"message_id": "m1", "mark_read": True})
    assert message.output["body"] == "bring the blue folder"
    assert env.observe()["unread_count"] == 0


def test_virtual_calendar_lookup_and_conflict_detection() -> None:
    env = VirtualCalendar(
        [
            {"id": "e1", "title": "Focus", "start": "2026-02-01T10:00", "end": "2026-02-01T11:00"},
            {"id": "e2", "title": "Lunch", "start": "2026-02-01T12:00", "end": "2026-02-01T13:00"},
        ]
    )
    lookup = env.execute_tool("calendar_lookup", {"date": "2026-02-01"})
    assert lookup.output["total"] == 2
    conflict = env.execute_tool("calendar_find_conflicts", {"start": "2026-02-01T10:30", "end": "2026-02-01T10:45"})
    assert conflict.output["has_conflict"] is True
    created = env.execute_tool(
        "calendar_create",
        {"title": "Free slot", "start": "2026-02-01T11:00", "end": "2026-02-01T12:00", "allow_conflict": False},
    )
    assert created.success and created.output["event"]["title"] == "Free slot"


def test_virtual_web_pages_are_allowlisted_and_never_fetch_network() -> None:
    env = VirtualWebPages({"https://intranet.test/start": {"title": "Start", "content": "Welcome", "links": {"next": "https://intranet.test/next"}}})
    opened = env.execute_tool("browser_open", {"url": "intranet.test/start"})
    assert opened.success and opened.output["title"] == "Start"
    missing = env.execute_tool("browser_open", {"url": "https://example.com"})
    assert not missing.success


def test_grid_world_partial_observation_and_key_door() -> None:
    env = SimpleGridWorld(
        width=4,
        height=2,
        start=(0, 0),
        goal=(3, 0),
        observation_radius=1,
        objects=[
            {"id": "key1", "kind": "key", "position": (0, 0), "portable": True},
            {"id": "door1", "kind": "door", "position": (2, 0), "state": {"key_id": "key1"}},
        ],
    )
    assert all(abs(c["x"]) <= 1 for c in env.observe()["visible_cells"])
    assert env.execute_tool("grid_pick_up", {"object_id": "key1"}).success
    moved = env.execute_tool("grid_move", {"direction": "E", "steps": 3})
    assert moved.success and moved.output["done"]
    assert env.observe()["position"] == {"x": 3, "y": 0}


def test_grid_world_supports_danger_teleport_movable_and_hidden_goal_rule() -> None:
    env = SimpleGridWorld(
        width=5,
        height=1,
        start=(0, 0),
        goal=(4, 0),
        terrain={(1, 0): "danger"},
        hidden_rules={"goal_requires_item": "food"},
        objects=[
            {"id": "food", "kind": "food", "position": (0, 0), "portable": True},
            {"id": "crate", "kind": "movable", "position": (2, 0)},
            {"id": "portal", "kind": "teleport", "position": (4, 0), "state": {"destination": (1, 0)}},
            {"id": "npc", "kind": "npc", "position": (0, 0), "state": {"passable": True}},
        ],
    )
    assert env.observe()["in_danger"] is False
    assert env.execute_tool("grid_pick_up", {"object_id": "food"}).success
    env.execute_tool("grid_move", {"direction": "E"})
    assert env.observe()["in_danger"] is True
    # The crate is directly ahead from the danger cell.
    assert env.execute_tool("grid_interact", {"object_id": "crate"}).success
    assert env.objects["crate"].position == (3, 0)


def test_code_sandbox_does_not_execute_source() -> None:
    env = CodeSandboxMock({"/main.py": 'print("hello")\n'})
    result = env.execute_tool("execute_mock_code", {"path": "/main.py"})
    assert result.success
    assert result.output["stdout"] == "hello"
    assert result.output["executed"] is False
    malicious = env.execute_tool("execute_mock_code", {"code": '__import__("os").system("touch /tmp/bad")'})
    assert malicious.success  # parser-only mock; source is never evaluated


def test_registry_routes_through_environment_and_calculator_is_bounded() -> None:
    env = VirtualFileSystem({"/a.txt": "a"})
    registry = build_default_tool_registry(env)
    result = registry.invoke("list_files", {"path": "/"})
    assert result.success
    assert env.step_count == 1
    assert registry.invoke("calculator", {"expression": "2 * (3 + 4)"}).output["value"] == 14
    unsafe = registry.invoke("calculator", {"expression": "__import__('os').system('echo bad')"})
    assert not unsafe.success


def test_environment_bundle_resets_children_and_routes_unique_tools() -> None:
    files = VirtualFileSystem({"/a": "x"})
    mail = VirtualEmailInbox([{"id": "m1", "subject": "x", "body": "y", "timestamp": "2026-01-01"}])
    bundle = EnvironmentBundle({"files": files, "mail": mail})
    registry = build_default_tool_registry(bundle, aliases=False)
    assert registry.invoke("list_files", {"path": "/"}).success
    assert registry.invoke("search_mail", {"query": "x"}).success
    assert files.step_count == 1 and mail.step_count == 1
    files.execute_tool("write_file", {"path": "/new", "content": "z"})
    bundle.reset(seed=4)
    assert "/new" not in files.files


def test_factory_aliases_are_stable() -> None:
    assert isinstance(make_environment("vfs"), VirtualFileSystem)
    assert isinstance(make_environment("grid"), SimpleGridWorld)


def test_task_initial_state_can_be_loaded_on_reset() -> None:
    # TaskRunner passes initial_state when a reset signature accepts it.  This
    # keeps one reusable environment safe for generated tasks without leaking
    # hidden ground truth.
    for category in ("file_organization", "email_retrieval", "calendar_scheduling", "code_repair", "grid_navigation"):
        task = build_task(category, seed=17, difficulty="easy")
        env = make_environment_for_task(task)
        env.reset(initial_state=task.initial_state, seed=17)
        assert env.observe()["environment"]
