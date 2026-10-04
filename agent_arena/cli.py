"""Command line interface for the Agent Arena demo and local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .benchmark import default_agent_factories, make_task_environment
from .core import ScriptedAgent
from .demo_dataset import generate_demo_tasks, load_dataset, save_dataset
from .evaluator import Evaluator
from .replay import Replay, ReplayStore, ReplayViewer
from .reporting import aggregate_metrics, write_report
from .storage import read_json, write_json, write_sqlite
from .task_catalog import build_task
from .tournament import TournamentRunner
from .benchmark import scripted_actions_for_task


def _parse_seeds(values: list[str] | None) -> list[int]:
    if not values:
        return [0]
    output: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                output.append(int(token))
    return output or [0]


def _runs_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        runs = payload.get("runs", payload.get("results", []))
        if isinstance(runs, list):
            return runs
        if payload.get("trace") or payload.get("steps"):
            return [payload]
    if isinstance(payload, list):
        return payload
    return []


def cmd_demo(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = generate_demo_tasks(count=args.tasks, seed=args.dataset_seed, adversarial_rate=args.adversarial_rate)
    dataset_path = output / "tasks.json"
    save_dataset(tasks, dataset_path)
    seeds = _parse_seeds(args.seeds)
    tournament = TournamentRunner(default_agent_factories(), evaluator=Evaluator()).run(tasks, seeds=seeds)
    results_path = output / "results.json"
    write_json(results_path, tournament.to_dict())
    sqlite_path = output / "results.sqlite"
    write_sqlite(sqlite_path, tournament.runs)
    replay_dir = output / "replays"
    report_path = output / "report.html"
    write_report(tournament.runs, report_path, title="Agent Arena Demo Tournament", replay_dir=replay_dir)
    # Keep a single obvious entry point in addition to the per-run pages.
    if tournament.runs:
        ReplayViewer.write(Replay.from_run(tournament.runs[0]), output / "replay.html", title="Agent Arena Replay")
    summary = tournament.summary()
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {
        "project": "Agent Arena",
        "synthetic": True,
        "task_count": len(tasks),
        "agent_count": len(tournament.agents),
        "seeds": seeds,
        "files": [p.name for p in sorted(output.iterdir()) if p.is_file()],
    })
    print(json.dumps({
        "output": str(output),
        "tasks": len(tasks),
        "agents": tournament.agents,
        "seeds": seeds,
        "runs": len(tournament.runs),
        "success_rate": summary["overall"]["success_rate"],
        "report": str(report_path),
        "synthetic": True,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_generate_tasks(args: argparse.Namespace) -> int:
    from .task_catalog import generate_tasks

    tasks = generate_tasks(args.count, seed=args.seed, adversarial_rate=args.adversarial_rate)
    save_dataset(tasks, args.output)
    print(json.dumps({"output": str(args.output), "count": len(tasks)}, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    task = build_task(args.category, seed=args.seed, difficulty=args.difficulty)
    env = make_task_environment(task, seed=args.seed)
    if args.agent == "scripted":
        agent = ScriptedAgent(scripted_actions_for_task(task), name="ScriptedAgent", seed=args.seed)
    else:
        factory = default_agent_factories().get(args.agent)
        if factory is None:
            raise SystemExit(f"unknown agent: {args.agent}")
        agent = factory(args.seed)
    from .runner import TaskRunner

    result = TaskRunner(environment=env, evaluator=Evaluator()).run(task, agent, seed=args.seed, agent_id=args.agent)
    payload = result.to_dict()
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


def cmd_replay(args: argparse.Namespace) -> int:
    payload = read_json(args.input)
    runs = _runs_from_payload(payload)
    selected: Any = None
    if args.run_id:
        for run in runs:
            if str(run.get("run_id", run.get("id", ""))) == str(args.run_id):
                selected = run
                break
        if selected is None:
            raise SystemExit(f"run id not found: {args.run_id}")
    elif runs:
        selected = runs[0]
    else:
        selected = payload
    replay = Replay.from_dict(selected) if isinstance(selected, dict) and "events" in selected else Replay.from_run(selected)
    output = Path(args.output)
    ReplayViewer.write(replay, output)
    print(json.dumps({"output": str(output), "run_id": replay.run_id, "steps": replay.step_count}, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    runs = _runs_from_payload(read_json(args.input))
    output = Path(args.output)
    write_report(runs, output, title=args.title, replay_dir=(output.parent / "replays" if args.with_replays else None))
    summary = aggregate_metrics(runs)
    print(json.dumps({"output": str(output), "runs": len(runs), "success_rate": summary["overall"]["success_rate"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    payload = read_json(args.input)
    runs = _runs_from_payload(payload)
    summary = aggregate_metrics(runs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-arena", description="Agent Arena / AI Agent 竞技场")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the standard 3-agent benchmark")
    demo.add_argument("--tasks", type=int, default=50)
    demo.add_argument("--dataset-seed", type=int, default=20261004)
    demo.add_argument("--seeds", nargs="*", default=["0", "1", "2"], help="one or more seeds, e.g. --seeds 0 1 2")
    demo.add_argument("--adversarial-rate", type=float, default=0.30)
    demo.add_argument("--output", default="artifacts/demo")
    demo.set_defaults(func=cmd_demo)

    gen = sub.add_parser("generate-tasks", help="create a deterministic task dataset")
    gen.add_argument("--count", type=int, default=50)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--adversarial-rate", type=float, default=0.0)
    gen.add_argument("--output", required=True)
    gen.set_defaults(func=cmd_generate_tasks)

    run = sub.add_parser("run", help="run one task against one offline agent")
    run.add_argument("--category", default="file_organization")
    run.add_argument("--difficulty", default="easy", choices=("easy", "medium", "hard"))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--agent", default="RuleBasedAgent", choices=("RandomAgent", "RuleBasedAgent", "MockLLMAgent", "scripted"))
    run.add_argument("--output")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="render a run trace as standalone HTML")
    replay.add_argument("--input", required=True)
    replay.add_argument("--run-id")
    replay.add_argument("--output", required=True)
    replay.set_defaults(func=cmd_replay)

    report = sub.add_parser("report", help="render a dashboard from JSON results")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--title", default="Agent Arena Report")
    report.add_argument("--with-replays", action="store_true")
    report.set_defaults(func=cmd_report)

    inspect = sub.add_parser("inspect", help="print aggregate metrics as JSON")
    inspect.add_argument("--input", required=True)
    inspect.set_defaults(func=cmd_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = ["build_parser", "main"]
