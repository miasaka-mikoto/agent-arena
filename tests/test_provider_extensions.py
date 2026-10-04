"""Small safety contracts for deferred provider adapters."""

from __future__ import annotations

import pytest

from agent_arena.core import Action, OpenAIProvider


def test_finish_action_mapping_is_normalized_without_network_access() -> None:
    action = Action.from_any({"finish": True, "reason": "done"})
    assert action is not None
    assert action.finish is True
    assert action.action_type == "finish"


def test_external_provider_stubs_are_disabled_by_default() -> None:
    provider = OpenAIProvider()
    provider.reset()
    with pytest.raises(RuntimeError, match="disabled in offline mode"):
        provider.act({"available_tools": []})
