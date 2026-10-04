"""Contract tests for Agent Arena's portable task models.

These tests intentionally exercise only public behaviour.  They are useful as
an early compatibility check for runners/environments that consume task JSON,
and do not couple the test suite to a particular implementation module beyond
the package's documented ``tasking`` module.
"""

from __future__ import annotations

import json

import pytest

from agent_arena.tasking import (
    TaskDefinition,
    TaskResult,
    TaskStatus,
    tasks_from_json,
    tasks_to_json,
)


def test_task_definition_defaults_and_aliases() -> None:
    task = TaskDefinition(task_id="t-1", instruction="wait")

    assert task.id == "t-1"
    assert task.max_steps == 20
    assert task.timeout == 30.0
    assert task.initial_observation == {}


def test_task_definition_deduplicates_tools_and_tags_preserving_order() -> None:
    task = TaskDefinition(
        task_id="files-1",
        instruction="organise files",
        allowed_tools=["list_files", "read_file", "list_files"],
        tags=["files", "easy", "files"],
    )

    assert task.allowed_tools == ["list_files", "read_file"]
    assert task.tags == ["files", "easy"]
    assert task.category == "files"


def test_task_definition_state_is_defensively_copied() -> None:
    initial = {"files": [{"name": "a.txt"}]}
    truth = {"answer": {"move": "a.txt"}}
    task = TaskDefinition(
        task_id="copy-1",
        instruction="copy",
        initial_state=initial,
        hidden_ground_truth=truth,
    )

    observation = task.initial_observation
    serialised = task.to_dict()
    observation["files"][0]["name"] = "mutated.txt"
    serialised["hidden_ground_truth"]["answer"]["move"] = "other.txt"

    assert task.initial_state["files"][0]["name"] == "a.txt"
    assert task.hidden_ground_truth["answer"]["move"] == "a.txt"


def test_task_definition_round_trip_json_and_clone_are_independent() -> None:
    task = TaskDefinition(
        task_id="mail-1",
        instruction="find mail",
        initial_state={"inbox": [{"id": 1}]},
        allowed_tools=["search_mail"],
        hidden_ground_truth={"mail_id": 1},
        success_conditions={"required": [1]},
        timeout=4.5,
        max_steps=8,
        difficulty="medium",
        tags=["email"],
        category="email",
        seed=42,
        metadata={"source": "test"},
    )

    payload = task.to_json(indent=None)
    restored = TaskDefinition.from_dict(json.loads(payload))
    clone = task.clone(task_id="mail-1-copy", difficulty="hard")

    assert restored.to_dict() == task.to_dict()
    assert clone.task_id == "mail-1-copy"
    assert clone.difficulty == "hard"
    assert task.difficulty == "medium"
    clone.initial_state["inbox"][0]["id"] = 99
    assert task.initial_state["inbox"][0]["id"] == 1


def test_dataset_helpers_require_a_json_list() -> None:
    tasks = [
        TaskDefinition(task_id="a", instruction="one"),
        TaskDefinition(task_id="b", instruction="two"),
    ]
    restored = tasks_from_json(tasks_to_json(tasks))

    assert [task.task_id for task in restored] == ["a", "b"]
    with pytest.raises(ValueError, match="must contain a list"):
        tasks_from_json("{}")


def test_task_definition_validates_limits() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        TaskDefinition(task_id="bad", instruction="x", max_steps=0)
    with pytest.raises(ValueError, match="timeout"):
        TaskDefinition(task_id="bad", instruction="x", timeout=0)
    with pytest.raises(ValueError, match="task_id"):
        TaskDefinition(task_id="", instruction="x")
    assert TaskDefinition(task_id="unlimited", instruction="x", timeout=None).timeout is None


def test_task_result_serialises_enum_status() -> None:
    result = TaskResult(task_id="t", status=TaskStatus.SUCCESS, score=1.0, steps=3)

    assert result.to_dict() == {
        "task_id": "t",
        "status": "success",
        "score": 1.0,
        "steps": 3,
        "error": None,
        "metrics": {},
    }
