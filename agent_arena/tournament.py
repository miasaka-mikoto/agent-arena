"""Deterministic Agent × Task × Seed tournament execution."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from .benchmark import default_agent_factories, make_task_environment
from .evaluator import Evaluator
from .runner import TaskRunner
from .storage import jsonable, write_json, write_sqlite
from .tasking import TaskDefinition


def _call_factory(factory: Any, *, seed: int, task: TaskDefinition) -> Any:
    """Instantiate a provider factory with the richest supported signature."""

    if not callable(factory):
        # A provider object is accepted for one-off experiments.  Prefer a
        # shallow/deep copy so episode state cannot leak across runs.
        import copy

        try:
            return copy.deepcopy(factory)
        except Exception:
            return factory
    try:
        signature = inspect.signature(factory)
        params = list(signature.parameters.values())
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if any(p.kind == p.VAR_POSITIONAL for p in params) or len(positional) >= 2:
            return factory(seed, task)
        if len(positional) == 1:
            return factory(seed)
        return factory()
    except (TypeError, ValueError):
        for args in ((seed, task), (seed,), ()):
            try:
                return factory(*args)
            except TypeError:
                continue
        return factory()


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120]


@dataclass
class TournamentResult:
    """A complete tournament with serialisable aggregate metadata."""

    runs: list[Any] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    task_count: int = 0
    seeds: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def results(self) -> list[Any]:
        return self.runs

    def summary(self) -> dict[str, Any]:
        from .reporting import build_dashboard_data

        return build_dashboard_data(self.runs)

    def to_dict(self, *, include_summary: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "agents": list(self.agents),
            "task_count": self.task_count,
            "seeds": list(self.seeds),
            "metadata": jsonable(self.metadata),
            "runs": [jsonable(run) for run in self.runs],
        }
        if include_summary:
            payload["summary"] = self.summary()
        return payload

    def save_json(self, path: str) -> str:
        write_json(path, self.to_dict())
        return str(path)

    def save_sqlite(self, path: str) -> str:
        write_sqlite(path, self.runs)
        return str(path)


class TournamentRunner:
    """Run a reproducible Cartesian product of agents, tasks and seeds."""

    def __init__(
        self,
        agent_factories: Mapping[str, Any] | None = None,
        *,
        environment_factory: Callable[..., Any] = make_task_environment,
        evaluator: Evaluator | None = None,
        continue_on_error: bool = True,
    ) -> None:
        self.agent_factories = dict(agent_factories or default_agent_factories())
        self.environment_factory = environment_factory
        self.evaluator = evaluator or Evaluator()
        self.continue_on_error = bool(continue_on_error)

    def _environment(self, task: TaskDefinition, seed: int) -> Any:
        factory = self.environment_factory
        try:
            return factory(task=task, seed=seed)
        except TypeError:
            try:
                return factory(task, seed)
            except TypeError:
                return factory(task)

    def run(
        self,
        tasks: Iterable[TaskDefinition | Mapping[str, Any]],
        *,
        seeds: Sequence[int] = (0,),
        max_tasks: int | None = None,
    ) -> TournamentResult:
        task_list: list[TaskDefinition] = []
        for raw in tasks:
            task = raw if isinstance(raw, TaskDefinition) else TaskDefinition.from_dict(raw)
            task_list.append(task)
            if max_tasks is not None and len(task_list) >= max_tasks:
                break
        seed_list = [int(seed) for seed in seeds]
        if not seed_list:
            seed_list = [0]
        output: list[Any] = []
        for agent_name, factory in self.agent_factories.items():
            for task in task_list:
                for base_seed in seed_list:
                    run_seed = int(base_seed) if task.seed is None else int(base_seed) + int(task.seed)
                    agent = _call_factory(factory, seed=run_seed, task=task)
                    env = self._environment(task, run_seed)
                    run_id = f"{_safe_id(agent_name)}__{_safe_id(task.task_id)}__s{run_seed}"
                    runner = TaskRunner(environment=env, evaluator=self.evaluator,
                                        run_id_factory=lambda rid=run_id: rid)
                    try:
                        result = runner.run(task, agent, seed=run_seed, agent_id=str(agent_name), run_id=run_id)
                    except Exception:
                        if not self.continue_on_error:
                            raise
                        # Keep a failed run visible in the tournament instead
                        # of silently dropping an agent/task cell.
                        from .trace import AgentTrace, RunResult

                        trace = AgentTrace(run_id=run_id, task_id=task.task_id, agent_id=str(agent_name), seed=run_seed)
                        trace.finish("error", error="runner exception")
                        result = RunResult(run_id=run_id, task_id=task.task_id, agent_id=str(agent_name), seed=run_seed,
                                           status="error", success=False, trace=trace, error="runner exception")
                    result.metrics.setdefault("category", task.category)
                    result.metrics.setdefault("difficulty", task.difficulty)
                    result.metrics.setdefault("synthetic", True)
                    result.metrics.setdefault("agent_type", getattr(agent, "provider_type", agent.__class__.__name__))
                    result.metrics.setdefault("task_adversarial", bool(task.metadata.get("is_adversarial")))
                    output.append(result)
        return TournamentResult(runs=output, agents=[str(name) for name in self.agent_factories],
                                task_count=len(task_list), seeds=seed_list,
                                metadata={"synthetic": True, "agent_count": len(self.agent_factories),
                                          "run_count": len(output)})


def run_tournament(tasks: Iterable[TaskDefinition | Mapping[str, Any]], *, agent_factories: Mapping[str, Any] | None = None,
                   seeds: Sequence[int] = (0,), **kwargs: Any) -> TournamentResult:
    return TournamentRunner(agent_factories, **kwargs).run(tasks, seeds=seeds)


__all__ = ["TournamentResult", "TournamentRunner", "run_tournament"]

