from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_arena.benchmark import default_agent_factories
from agent_arena.cli import main
from agent_arena.replay import Replay, ReplayViewer
from agent_arena.task_catalog import generate_demo_tasks
from agent_arena.tournament import TournamentRunner


def test_tournament_cartesian_product_and_replay() -> None:
    tasks = generate_demo_tasks(5, seed=11, adversarial_rate=0.0)
    result = TournamentRunner(default_agent_factories()).run(tasks, seeds=(0, 1))
    assert len(result.runs) == 3 * 5 * 2
    assert {run.agent_id for run in result.runs} == {"RandomAgent", "RuleBasedAgent", "MockLLMAgent"}
    assert all(run.metrics.get("synthetic") is True for run in result.runs)
    replay = Replay.from_run(result.runs[0])
    html = ReplayViewer.render(replay)
    assert "Timeline" in html
    assert "private" not in html.lower() or "private reasoning" not in html.lower()


def test_cli_demo_writes_complete_artifact_set(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    assert main(["demo", "--tasks", "5", "--seeds", "0", "--output", str(output)]) == 0
    for name in ("tasks.json", "results.json", "results.sqlite", "report.html", "replay.html", "summary.json", "manifest.json"):
        assert (output / name).exists(), name
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 15
    assert payload["metadata"]["synthetic"] is True
    with sqlite3.connect(output / "results.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 15
    assert not list(output.glob(".results.sqlite.*.tmp"))
