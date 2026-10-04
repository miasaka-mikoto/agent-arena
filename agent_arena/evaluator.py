"""Deterministic task evaluation and failure classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .trace import AgentTrace, RunResult, TraceStep, json_safe


class FailureTaxonomy(str, Enum):
    PLANNING_FAILURE = "planning_failure"
    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENT = "wrong_argument"
    LOST_CONTEXT = "lost_context"
    LOOP = "loop"
    HALLUCINATED_OBJECT = "hallucinated_object"
    PREMATURE_FINISH = "premature_finish"
    TIMEOUT = "timeout"
    TOOL_ERROR_RECOVERY_FAILURE = "tool_error_recovery_failure"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _task_value(task: Any, key: str, default: Any = None) -> Any:
    return _get(task, key, default)


def _path_get(value: Any, path: str, default: Any = None) -> Any:
    if path in ("", ".", None):
        return value
    current = value
    for part in str(path).replace("[", ".").replace("]", "").split("."):
        if part == "":
            continue
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, TypeError):
                return default
        else:
            current = getattr(current, part, default)
            if current is default:
                return default
    return current


def _normalise_tool(step: Any) -> Optional[str]:
    tool = _get(step, "tool")
    if tool:
        return str(tool)
    action = _get(step, "action")
    if isinstance(action, Mapping):
        value = action.get("tool", action.get("name", action.get("tool_name")))
        return str(value) if value else None
    return str(getattr(action, "tool", "")) or None


def _fingerprint(step: Any) -> str:
    tool = _normalise_tool(step) or ""
    args = _get(step, "arguments", {}) or {}
    action_type = _get(step, "action_type", "") or ""
    return repr((tool, action_type, args))


def _step_error(step: Any) -> Optional[str]:
    error = _get(step, "error")
    if error not in (None, "", False):
        return str(error)
    result = _get(step, "result")
    if isinstance(result, Mapping):
        if result.get("error") not in (None, "", False):
            return str(result["error"])
        if result.get("ok") is False or result.get("success") is False:
            return str(result.get("message", "tool failure"))
    else:
        raw = getattr(result, "error", None)
        if raw not in (None, "", False):
            return str(raw)
    return None


def _safe_equal(left: Any, right: Any) -> bool:
    # JSON-like values are compared structurally, while allowing numeric 1 and
    # 1.0 to match as users generally expect in task definitions.
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return dict(left) == dict(right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    return left == right


def _mapping_contains(value: Any, expected: Any) -> bool:
    """Return whether ``value`` structurally contains an expected partial map."""

    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping):
            return False
        return all(key in value and _mapping_contains(value[key], wanted) for key, wanted in expected.items())
    if isinstance(expected, (list, tuple)):
        if not isinstance(value, (list, tuple)) or len(value) < len(expected):
            return False
        return all(_mapping_contains(actual, wanted) for actual, wanted in zip(value, expected))
    return _safe_equal(value, expected)


def _environment_state(environment: Any) -> Any:
    """Evaluator-only state accessor; never passed to an agent."""

    if environment is None:
        return None
    for name in ("ground_truth", "get_ground_truth", "evaluation_state", "snapshot"):
        fn = getattr(environment, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                continue
    return None


def _condition_satisfied(condition: Any, *, final_observation: Any, final_state: Any,
                         trace: AgentTrace, task: Any, environment: Any = None,
                         environment_state: Any = None) -> bool:
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    if callable(condition):
        try:
            return bool(condition(final_state, final_observation, trace))
        except TypeError:
            try:
                return bool(condition(final_state))
            except Exception:
                return False
        except Exception:
            return False
    if isinstance(condition, (list, tuple)):
        return all(_condition_satisfied(c, final_observation=final_observation, final_state=final_state,
                                        trace=trace, task=task, environment=environment,
                                        environment_state=environment_state) for c in condition)
    if not isinstance(condition, Mapping):
        return False

    if "all" in condition or "checks" in condition:
        values = condition.get("all", condition.get("checks", []))
        return all(_condition_satisfied(c, final_observation=final_observation, final_state=final_state,
                                        trace=trace, task=task, environment=environment,
                                        environment_state=environment_state) for c in values)
    if "any" in condition:
        return any(_condition_satisfied(c, final_observation=final_observation, final_state=final_state,
                                        trace=trace, task=task, environment=environment,
                                        environment_state=environment_state) for c in condition["any"])
    if "not" in condition:
        return not _condition_satisfied(condition["not"], final_observation=final_observation,
                                        final_state=final_state, trace=trace, task=task, environment=environment,
                                        environment_state=environment_state)

    op = str(condition.get("op", condition.get("type", condition.get("condition", "equals")))).lower()
    if op in {"tool_called", "tool_used"}:
        expected = str(condition.get("tool", condition.get("name", "")))
        return any(_normalise_tool(step) == expected for step in trace.steps)
    if op in {"required_tools", "tools"}:
        required = {str(x) for x in condition.get("tools", condition.get("required", []))}
        seen = {_normalise_tool(step) for step in trace.steps}
        return required.issubset(seen)
    if op in {"forbidden_tool", "forbidden_tools"}:
        forbidden = {str(x) for x in condition.get("tools", condition.get("tool", []))}
        if isinstance(condition.get("tool"), str):
            forbidden = {str(condition["tool"])}
        return not any(_normalise_tool(step) in forbidden for step in trace.steps)
    if op in {"min_steps", "max_steps", "steps"}:
        actual = len(trace.steps)
        if op == "min_steps":
            return actual >= int(condition.get("value", condition.get("min", 0)))
        if op == "max_steps":
            return actual <= int(condition.get("value", condition.get("max", 0)))
        return actual == int(condition.get("value", 0))
    if op in {"goal_reached", "environment_success", "success"} and environment is not None:
        for name in ("check_success", "is_success"):
            fn = getattr(environment, name, None)
            if callable(fn):
                try:
                    return bool(fn(task))
                except TypeError:
                    try:
                        return bool(fn())
                    except Exception:
                        pass
                except Exception:
                    pass

    # State predicates default to final_state, then final_observation.  A task
    # may set ``source`` explicitly to observation/state.
    source_name = str(condition.get("source", "state")).lower()
    if source_name in {"environment", "env", "ground_truth", "evaluation"}:
        source = environment_state if environment_state is not None else _environment_state(environment)
    else:
        source = final_observation if source_name in {"observation", "obs"} else final_state
    if source is None:
        source = final_observation
    path = condition.get("path", condition.get("field", condition.get("key", "")))
    actual = _path_get(source, str(path), None)
    if "expected" in condition:
        expected = condition["expected"]
    elif "value" in condition:
        expected = condition["value"]
    elif "equals" in condition:
        expected = condition["equals"]
    else:
        expected = None
    if op in {"contains", "includes"}:
        try:
            return expected in actual
        except Exception:
            return False
    if op in {"contains_item", "contains_mapping", "has_item", "has_mapping"}:
        expected_item = condition.get("expected", condition.get("value", condition.get("match", {})))
        if isinstance(actual, Mapping):
            # A mapping itself can be the target item; a list-valued field is
            # the usual case for email/calendar/objects.
            return _mapping_contains(actual, expected_item)
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            return False
        return any(_mapping_contains(item, expected_item) for item in actual)
    if op in {"absent", "not_exists", "missing"}:
        return actual is None
    if op in {"not_contains", "excludes"}:
        try:
            return expected not in actual
        except Exception:
            return True
    if op in {"exists", "present"}:
        return actual is not None
    if op in {"truthy", "true"}:
        return bool(actual)
    if op in {"falsy", "false"}:
        return not bool(actual)
    if op in {"greater", "gt"}:
        try:
            return actual > expected
        except Exception:
            return False
    if op in {"less", "lt"}:
        try:
            return actual < expected
        except Exception:
            return False
    # A named, environment-specific condition (for example
    # ``all_moves_applied``) can be evaluated by the simulator itself.  This
    # remains safe: evaluator-only code may see state, but no state is sent
    # back through the agent observation.
    if environment is not None and op not in {"equals", "equal", "state_equals", "path_equals"}:
        for name in ("check_success", "evaluate_condition", "is_success"):
            fn = getattr(environment, name, None)
            if callable(fn):
                try:
                    return bool(fn(task, condition))
                except TypeError:
                    try:
                        return bool(fn(task))
                    except TypeError:
                        try:
                            return bool(fn())
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
    return _safe_equal(actual, expected)


def classify_failures(task: Any, trace: AgentTrace, *, success: Optional[bool] = None,
                      environment: Any = None) -> List[FailureTaxonomy]:
    """Classify observable failures without inspecting private reasoning."""

    failures: List[FailureTaxonomy] = []
    allowed = {str(x) for x in (_task_value(task, "allowed_tools", []) or [])}
    seen_fingerprints: Dict[str, int] = {}
    had_error = False
    recovered_after_error = False

    if trace.status in {"timeout", "timed_out", "max_steps"}:
        failures.append(FailureTaxonomy.TIMEOUT)
    for index, step in enumerate(trace.steps):
        tool = _normalise_tool(step)
        error = _step_error(step)
        fp = _fingerprint(step)
        seen_fingerprints[fp] = seen_fingerprints.get(fp, 0) + 1
        # Repeating a successful movement (for example four eastward steps on
        # a straight grid corridor) is not a loop.  Only flag repetition when
        # the run is unsuccessful or the repeated call itself errored.
        if seen_fingerprints[fp] >= 3 and (success is not True or error) and FailureTaxonomy.LOOP not in failures:
            failures.append(FailureTaxonomy.LOOP)
        if tool and allowed and tool not in allowed and FailureTaxonomy.WRONG_TOOL not in failures:
            failures.append(FailureTaxonomy.WRONG_TOOL)
        if not tool and not _get(step, "action_type") and error is None and FailureTaxonomy.PLANNING_FAILURE not in failures:
            failures.append(FailureTaxonomy.PLANNING_FAILURE)
        if error:
            lowered = error.lower()
            had_error = True
            if any(token in lowered for token in ("argument", "parameter", "invalid value", "expected ", "malformed")):
                if FailureTaxonomy.WRONG_ARGUMENT not in failures:
                    failures.append(FailureTaxonomy.WRONG_ARGUMENT)
            if any(token in lowered for token in ("not found", "does not exist", "unknown file", "unknown object", "no such")):
                if FailureTaxonomy.HALLUCINATED_OBJECT not in failures:
                    failures.append(FailureTaxonomy.HALLUCINATED_OBJECT)
            if index + 1 < len(trace.steps) and _step_error(trace.steps[index + 1]) is None:
                recovered_after_error = True
            elif index + 1 >= len(trace.steps):
                if FailureTaxonomy.TOOL_ERROR_RECOVERY_FAILURE not in failures:
                    failures.append(FailureTaxonomy.TOOL_ERROR_RECOVERY_FAILURE)
            if any(token in lowered for token in ("context", "remember", "lost", "missing required")):
                if FailureTaxonomy.LOST_CONTEXT not in failures:
                    failures.append(FailureTaxonomy.LOST_CONTEXT)

    if success is False and trace.status in {"finished", "complete", "completed"}:
        # An explicit finish with unsatisfied conditions is distinguishable
        # from a timeout.
        if not trace.steps or _get(trace.steps[-1], "action_type", "") in {"finish", "done"}:
            failures.append(FailureTaxonomy.PREMATURE_FINISH)
    if had_error and not recovered_after_error and FailureTaxonomy.TOOL_ERROR_RECOVERY_FAILURE not in failures:
        failures.append(FailureTaxonomy.TOOL_ERROR_RECOVERY_FAILURE)
    return failures


@dataclass
class EvaluationResult:
    task_id: str
    run_id: str
    agent_id: str
    success: bool
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    failure_details: List[str] = field(default_factory=list)

    def apply_to(self, result: RunResult) -> None:
        result.success = self.success
        result.score = self.score
        result.metrics.update(self.metrics)
        result.failures = list(dict.fromkeys([*result.failures, *self.failures]))

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "task_id": self.task_id, "run_id": self.run_id, "agent_id": self.agent_id,
            "success": self.success, "score": self.score, "metrics": self.metrics,
            "failures": self.failures, "failure_details": self.failure_details,
        })


class Evaluator:
    """Evaluate a run using task conditions and observable execution data."""

    def __init__(self, *, cost_per_token: float = 0.0, loop_threshold: int = 3):
        self.cost_per_token = float(cost_per_token)
        self.loop_threshold = max(2, int(loop_threshold))

    def _environment_success(self, task: Any, environment: Any) -> Optional[bool]:
        if environment is None:
            return None
        for name in ("check_success", "is_success"):
            fn = getattr(environment, name, None)
            if callable(fn):
                for args in ((task,), ()):
                    try:
                        value = fn(*args)
                        if value is not None:
                            return bool(value)
                        break
                    except TypeError:
                        continue
                    except Exception:
                        break
        value = getattr(environment, "success", None)
        if isinstance(value, bool):
            return value
        return None

    def _token_usage(self, trace: AgentTrace, run: Any) -> int:
        total = 0
        for step in trace.steps:
            value = _get(step, "tokens", None)
            if value is None:
                result = _get(step, "result", {})
                if isinstance(result, Mapping):
                    value = result.get("token_usage", result.get("tokens", 0))
            try:
                total += int(value or 0)
            except (TypeError, ValueError):
                pass
        metadata = trace.metadata or {}
        try:
            total = max(total, int(metadata.get("token_usage", metadata.get("tokens", 0)) or 0))
        except (TypeError, ValueError):
            pass
        agent_metrics = metadata.get("agent_metrics", {})
        if isinstance(agent_metrics, Mapping):
            estimate = agent_metrics.get("token_usage_estimate", agent_metrics.get("token_usage", 0))
            if isinstance(estimate, Mapping):
                estimate = estimate.get("total_tokens", estimate.get("total", 0))
            try:
                total = max(total, int(estimate or 0))
            except (TypeError, ValueError):
                pass
        return total

    def evaluate(self, task: Any, run: RunResult | AgentTrace | Mapping[str, Any], *, environment: Any = None) -> EvaluationResult:
        if isinstance(run, AgentTrace):
            trace = run
            run_id = trace.run_id
            agent_id = trace.agent_id
            initial_success = None
            final_observation = trace.final_observation
            final_state = trace.final_state
        elif isinstance(run, RunResult):
            trace = run.trace
            run_id = run.run_id
            agent_id = run.agent_id
            initial_success = run.success
            final_observation = run.final_observation if run.final_observation is not None else trace.final_observation
            final_state = run.final_state if run.final_state is not None else trace.final_state
        else:
            raw_trace = run.get("trace", run)
            trace = raw_trace if isinstance(raw_trace, AgentTrace) else AgentTrace.from_dict(raw_trace)
            run_id = str(run.get("run_id", trace.run_id))
            agent_id = str(run.get("agent_id", trace.agent_id))
            initial_success = run.get("success")
            final_observation = run.get("final_observation", trace.final_observation)
            final_state = run.get("final_state", trace.final_state)

        conditions = _task_value(task, "success_conditions", {})
        env_success = self._environment_success(task, environment)
        # A task-aware simulator can validate private state against the hidden
        # goal without leaking it to the agent.  Prefer that authoritative
        # result for built-in benchmark environments; generic environments use
        # the portable declarative conditions below.
        if env_success is not None:
            success = env_success
        elif conditions:
            success = _condition_satisfied(conditions, final_observation=final_observation,
                                            final_state=final_state, trace=trace,
                                            task=task, environment=environment)
        elif initial_success is not None:
            success = bool(initial_success)
        else:
            # A task without an explicit predicate is considered successful
            # only after a clean, explicit finish; this prevents an empty run
            # from being reported as a pass.
            success = trace.status in {"finished", "complete", "completed"} and bool(trace.steps) and not any(_step_error(s) for s in trace.steps)

        failures_enum = classify_failures(task, trace, success=success, environment=environment)
        failures = [f.value for f in failures_enum]
        errors = [_step_error(s) for s in trace.steps]
        error_indices = [i for i, e in enumerate(errors) if e]
        invalid = 0
        allowed = {str(x) for x in (_task_value(task, "allowed_tools", []) or [])}
        for step in trace.steps:
            tool = _normalise_tool(step)
            if _step_error(step) or (tool and allowed and tool not in allowed) or not _get(step, "valid", True):
                invalid += 1
        retries = 0
        for previous, current in zip(trace.steps, trace.steps[1:]):
            if _step_error(previous) or _fingerprint(previous) == _fingerprint(current):
                retries += 1
        successful_tool_calls = sum(1 for s in trace.steps if _normalise_tool(s) and not _step_error(s))
        tool_calls = sum(1 for s in trace.steps if _normalise_tool(s))
        recovered = 0
        for i in error_indices:
            if any(_step_error(s) is None for s in trace.steps[i + 1:]):
                recovered += 1
        recovery = 1.0 if not error_indices else recovered / len(error_indices)
        token_usage = self._token_usage(trace, run)
        metadata = _task_value(task, "metadata", {}) or {}
        cost_rate = float(metadata.get("cost_per_token", self.cost_per_token) or 0.0) if isinstance(metadata, Mapping) else self.cost_per_token
        estimated_cost = token_usage * cost_rate
        latency = trace.total_latency
        tool_efficiency = successful_tool_calls / tool_calls if tool_calls else (1.0 if success else 0.0)
        # Primary score is success; efficiency/recovery are exposed separately
        # so a leaderboard cannot silently turn a failed task into a pass.
        score = 1.0 if success else 0.0
        metrics = {
            "success": bool(success),
            "success_rate": 1.0 if success else 0.0,
            "steps": len(trace.steps),
            "invalid_actions": invalid,
            "retries": retries,
            "recovery_ability": round(recovery, 6),
            "tool_efficiency": round(tool_efficiency, 6),
            "latency": round(latency, 6),
            "latency_ms": round(latency * 1000.0, 3),
            "token_usage": token_usage,
            "estimated_cost": estimated_cost,
            "failure_count": len(failures),
            "tool_calls": tool_calls,
            "successful_tool_calls": successful_tool_calls,
        }
        return EvaluationResult(task_id=str(_task_value(task, "task_id", _task_value(task, "id", trace.task_id))),
                                run_id=run_id, agent_id=agent_id, success=bool(success), score=score,
                                metrics=metrics, failures=failures,
                                failure_details=[f.label for f in failures_enum])


def evaluate_run(task: Any, run: RunResult | AgentTrace, *, environment: Any = None, **kwargs: Any) -> EvaluationResult:
    return Evaluator(**kwargs).evaluate(task, run, environment=environment)


__all__ = ["FailureTaxonomy", "EvaluationResult", "Evaluator", "classify_failures", "evaluate_run"]
