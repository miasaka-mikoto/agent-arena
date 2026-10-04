"""Contract tests for the offline AgentProvider implementations.

The tests keep the provider boundary deliberately small: agents see an
Observation, return an Action, and receive a ToolResult.  No real external
provider or environment is involved.
"""

from __future__ import annotations

from agent_arena.core import (
    Action,
    AgentProvider,
    AgentResult,
    MockLLMAgent,
    Observation,
    Plan,
    RandomAgent,
    RuleBasedAgent,
    ScriptedAgent,
    TaskDefinition,
    ToolCall,
    ToolResult,
)
from agent_arena.models import TaskDefinition as FacadeTaskDefinition
from agent_arena.tasking import TaskDefinition as CanonicalTaskDefinition


def _observation(*tools: str, state: dict | None = None, step: int = 0) -> Observation:
    return Observation(
        state=state or {}, available_tools=list(tools), step=step
    )


def test_task_definition_facades_share_one_canonical_type() -> None:
    # Runner/catalogue data and provider reset data must be interchangeable;
    # having two subtly different dataclasses here would make isinstance checks
    # and JSON round-trips fail at integration boundaries.
    from agent_arena.core import TaskDefinition as CoreTaskDefinition

    assert CoreTaskDefinition is CanonicalTaskDefinition
    assert FacadeTaskDefinition is CanonicalTaskDefinition


def test_action_and_tool_result_accept_common_adapter_shapes() -> None:
    action = Action.from_any({"name": "read_file", "args": {"path": "a.txt"}})
    call = ToolCall("list_files")
    result = ToolResult.from_any({"ok": False, "result": None, "error": "missing"}, tool="read_file")

    assert action is not None
    assert action.tool == "read_file"
    assert action.arguments == {"path": "a.txt"}
    assert call.to_action().tool == "list_files"
    assert result.tool == "read_file"
    assert result.success is False
    assert result.error == "missing"


def test_observation_round_trip_does_not_have_a_hidden_state_channel() -> None:
    observation = _observation("list_files", state={"visible": 1})
    payload = observation.to_dict()

    assert payload["state"] == {"visible": 1}
    assert "hidden_ground_truth" not in payload
    restored = Observation.from_any(payload)
    assert restored.available_tools == ["list_files"]
    assert restored.state == {"visible": 1}


def test_agent_provider_lifecycle_records_public_result_only() -> None:
    provider = AgentProvider(name="fixture", seed=7)
    task = TaskDefinition(task_id="task-1", instruction="inspect", allowed_tools=["list_files"])
    provider.reset(task=task, seed=11)
    obs = provider.observe(_observation("list_files"))
    plan = provider.plan()
    result = provider.record_result(
        {"success": True, "output": ["a.txt"]}, action=Action("list_files")
    )
    finished = provider.finish()

    assert obs.available_tools == ["list_files"]
    assert isinstance(plan, Plan)
    assert result.ok
    assert provider.last_action is not None
    assert finished.steps == 1
    assert provider.finished is True


def test_random_agent_is_seeded_and_uses_only_public_tools() -> None:
    obs = _observation("read_file", "list_files", state={"files": [{"path": "a.txt"}]})
    left = RandomAgent(seed=123)
    right = RandomAgent(seed=123)
    left.reset()
    right.reset()

    first = left.act(obs)
    second = right.act(obs)

    assert first is not None and second is not None
    assert first.to_dict() == second.to_dict()
    assert first.tool in {"read_file", "list_files"}


def test_rule_based_agent_prioritises_informative_tools_and_recovers() -> None:
    agent = RuleBasedAgent(seed=1)
    agent.reset()
    obs = _observation("read_file", "list_files", "calculator")
    plan = agent.plan(observation=obs)
    first = agent.act()

    assert [action.tool for action in plan.actions] == ["list_files", "read_file", "calculator"]
    assert first is not None and first.tool == "list_files"

    agent.record_result(ToolResult(tool="list_files", success=False, error="temporary"), action=first)
    recovery = agent.act(obs)
    assert recovery is not None
    assert recovery.tool in {"read_file", "calculator"}


def test_scripted_agent_replays_actions_in_order_and_stops_at_end() -> None:
    agent = ScriptedAgent(
        [
            {"tool": "list_files"},
            Action("read_file", {"path": "a.txt"}),
        ]
    )
    agent.reset()
    obs = _observation("list_files", "read_file")

    first = agent.act(obs)
    second = agent.act(obs)
    third = agent.act(obs)

    assert first is not None and first.tool == "list_files"
    assert second is not None and second.tool == "read_file"
    assert third is None


def test_mock_llm_uses_responses_then_rule_fallback_and_marks_estimates() -> None:
    agent = MockLLMAgent(responses=[{"tool": "list_files"}], seed=0)
    agent.reset()
    obs = _observation("list_files", "calculator")

    explicit = agent.act(obs)
    fallback = agent.act(obs)
    result = agent.finish(AgentResult(success=True))

    assert explicit is not None and explicit.tool == "list_files"
    assert fallback is not None and fallback.tool in {"list_files", "calculator"}
    usage = result.metrics["token_usage_estimate"]
    assert usage["estimated"] is True
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
