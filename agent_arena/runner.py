"""Task execution loop for Agent Arena.

The runner intentionally talks to agents and environments through a small
duck-typed adapter.  This lets the built-in mock agents use the same runner as
future HTTP/model providers, while keeping all environments simulated and
side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import itertools
import time
import uuid
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .tasking import TaskDefinition
from .trace import AgentTrace, RunResult, TraceStep, json_safe


def _task_value(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(name, default)
    return getattr(task, name, default)


def _public_task(task: Any) -> Any:
    """Return an agent-safe task view with hidden ground truth removed."""

    public = getattr(task, "public_dict", None)
    if callable(public):
        try:
            return public()
        except Exception:
            pass
    if isinstance(task, Mapping):
        value = dict(task)
        value.pop("hidden_ground_truth", None)
        value.pop("ground_truth", None)
        value.pop("initial_state", None)
        value.pop("success_conditions", None)
        return value
    # A custom object may not offer a public_dict method.  Build a small
    # mapping from documented task fields rather than handing the object (and
    # potentially its private state) to an agent.
    fields = ("task_id", "instruction", "allowed_tools", "timeout", "max_steps", "difficulty", "tags", "metadata")
    result = {name: getattr(task, name) for name in fields if hasattr(task, name)}
    return result or task


def _method(obj: Any, name: str) -> Optional[Callable[..., Any]]:
    candidate = getattr(obj, name, None)
    return candidate if callable(candidate) else None


def _call_flexible(fn: Callable[..., Any], *, kwargs: Optional[Mapping[str, Any]] = None,
                   positional: Sequence[Tuple[Any, ...]] = ((),), default: Any = None) -> Any:
    """Call a provider method while tolerating small signature differences.

    We first filter named arguments according to the signature.  For opaque
    callables (some plugin proxies do not expose a signature), a short list of
    conservative positional fallbacks is attempted.  Exceptions raised by the
    actual provider are re-raised after all compatible forms fail.
    """

    kwargs = dict(kwargs or {})
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        filtered = kwargs if accepts_var_kw else {k: v for k, v in kwargs.items() if k in params}
        required_missing = [
            p for p in params.values()
            if p.name != "self"
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
            and p.name not in filtered
        ]
        # If all required positional parameters are represented, named calling
        # avoids guessing whether a provider expects an Observation or a Plan.
        if not required_missing:
            try:
                return fn(**filtered)
            except TypeError:
                # Fall through only for a signature mismatch; provider errors
                # are retried below and ultimately surfaced.
                pass
    except (TypeError, ValueError):
        pass

    last_error: Optional[Exception] = None
    for args in positional:
        try:
            return fn(*args)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        if default is not None:
            return default
        raise last_error
    return default


def _extract_tool_result(result: Any) -> Tuple[Any, Optional[str], bool, Optional[bool]]:
    """Return ``(public_result, error, is_error, done)`` for any tool result."""

    public = result
    error: Optional[str] = None
    done: Optional[bool] = None
    if isinstance(result, Mapping):
        # Keep the complete result for replay; environments are responsible for
        # not returning hidden ground truth through this public channel.
        if result.get("error") not in (None, "", False):
            error = str(result.get("error"))
        elif result.get("ok") is False or result.get("success") is False:
            error = str(result.get("message", "tool returned failure"))
        if "done" in result:
            done = bool(result.get("done"))
        elif "terminated" in result:
            done = bool(result.get("terminated"))
    else:
        raw_error = getattr(result, "error", None)
        if raw_error not in (None, "", False):
            error = str(raw_error)
        elif getattr(result, "ok", True) is False or getattr(result, "success", True) is False:
            error = str(getattr(result, "message", "tool returned failure"))
        if hasattr(result, "done"):
            done = bool(getattr(result, "done"))
        elif hasattr(result, "terminated"):
            done = bool(getattr(result, "terminated"))
    return public, error, error is not None, done


@dataclass
class NormalizedAction:
    raw: Any
    tool: Optional[str] = None
    arguments: Dict[str, Any] = None  # type: ignore[assignment]
    action_type: Optional[str] = None
    public_reason_summary: Optional[str] = None
    finish: bool = False

    def __post_init__(self) -> None:
        if self.arguments is None:
            self.arguments = {}


def normalize_action(action: Any) -> NormalizedAction:
    """Normalise Action dataclasses, dictionaries and simple strings."""

    if action is None:
        return NormalizedAction(raw=action, finish=True, action_type="finish")

    # Providers may return (action, public_reason_summary).
    reason: Optional[str] = None
    if isinstance(action, tuple) and len(action) == 2:
        action, reason = action

    tool: Optional[str] = None
    args: Dict[str, Any] = {}
    action_type: Optional[str] = None
    done = False

    if isinstance(action, str):
        action_type = action
        done = action.lower() in {"finish", "done", "stop", "terminate", "end"}
    elif isinstance(action, Mapping):
        tool_value = action.get("tool", action.get("name", action.get("tool_name")))
        if tool_value is not None:
            tool = str(tool_value)
        action_type = action.get("action_type", action.get("type", action.get("kind")))
        raw_args = action.get("arguments", action.get("args", action.get("parameters", {})))
        if isinstance(raw_args, Mapping):
            args = dict(raw_args)
        elif raw_args is not None:
            args = {"value": raw_args}
        reason = reason or action.get("public_reason_summary", action.get("reason", action.get("explanation")))
        done = bool(action.get("finish", action.get("done", action.get("terminate", False))))
    else:
        tool_value = getattr(action, "tool", getattr(action, "name", getattr(action, "tool_name", None)))
        if tool_value is not None:
            tool = str(tool_value)
        action_type = getattr(action, "action_type", getattr(action, "type", getattr(action, "kind", None)))
        raw_args = getattr(action, "arguments", getattr(action, "args", getattr(action, "parameters", {})))
        if isinstance(raw_args, Mapping):
            args = dict(raw_args)
        elif raw_args is not None:
            args = {"value": raw_args}
        reason = reason or getattr(action, "public_reason_summary", getattr(action, "reason", None))
        done = bool(getattr(action, "finish", getattr(action, "done", False)))

    if action_type is not None and str(action_type).lower() in {"finish", "done", "stop", "terminate", "end"}:
        done = True
    if tool is not None and tool.lower() in {"finish", "done", "stop", "terminate", "end"}:
        done = True
    if reason is not None:
        reason = str(reason)[:2000]
    return NormalizedAction(raw=action, tool=tool, arguments=args, action_type=str(action_type) if action_type is not None else None,
                            public_reason_summary=reason, finish=done)


class TaskRunner:
    """Execute one :class:`TaskDefinition` against an agent/environment pair."""

    def __init__(self, environment: Any = None, *, environment_factory: Optional[Callable[..., Any]] = None,
                 evaluator: Any = None, clock: Callable[[], float] = time.monotonic,
                 run_id_factory: Optional[Callable[[], str]] = None):
        self.environment = environment
        self.environment_factory = environment_factory
        self.evaluator = evaluator
        self.clock = clock
        self.run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)

    def _make_environment(self, task: Any, seed: Optional[int], supplied: Any = None) -> Any:
        env = supplied if supplied is not None else self.environment
        if env is None and self.environment_factory is not None:
            factory = self.environment_factory
            env = _call_flexible(factory, kwargs={"task": task, "seed": seed},
                                 positional=((task, seed), (task,), (seed,), ()))
        if env is None:
            # Generated tasks carry an explicit simulated environment type in
            # ``initial_state``.  Constructing it here keeps the convenient
            # ``TaskRunner().run(task, agent)`` path fully offline while still
            # allowing callers to inject custom environments/factories.
            try:
                from .task_environment import make_environment_for_task

                env = make_environment_for_task(task)
            except Exception as exc:
                raise ValueError("TaskRunner requires an environment or environment_factory") from exc
        # A reusable environment can provide clone/copy; prefer it so each
        # tournament run starts from an isolated state.
        if supplied is None and self.environment_factory is None:
            for name in ("clone", "copy"):
                fn = _method(env, name)
                if fn:
                    try:
                        candidate = _call_flexible(fn, kwargs={"task": task, "seed": seed}, positional=((task, seed), ()))
                        if candidate is not None and candidate is not env:
                            env = candidate
                            break
                    except Exception:
                        pass
        return env

    def _reset_environment(self, env: Any, task: Any, seed: Optional[int]) -> Any:
        # Optional task loading hook is useful for environments with a richer
        # reset API; it receives only public task fields.
        load = _method(env, "load_task")
        if load:
            try:
                _call_flexible(load, kwargs={"task": task, "seed": seed}, positional=((task, seed), (task,), ()))
            except Exception:
                # A reset call below remains authoritative.
                pass
        reset = _method(env, "reset")
        if not reset:
            return self._observe(env)
        initial_state = _task_value(task, "initial_state", {})
        kwargs = {"task": task, "seed": seed, "initial_state": initial_state, "state": initial_state}
        variants = ((initial_state, seed), (initial_state,), (seed,), ())
        try:
            value = _call_flexible(reset, kwargs=kwargs, positional=variants)
        except TypeError:
            value = reset()
        # Some environments expose set_state separately and reset only seeds.
        set_state = _method(env, "set_state")
        if set_state and initial_state:
            try:
                _call_flexible(set_state, kwargs={"state": initial_state}, positional=((initial_state,),))
            except Exception:
                pass
        return value if value is not None else self._observe(env)

    def _observe(self, env: Any) -> Any:
        observe = _method(env, "observe")
        if observe:
            return _call_flexible(observe, positional=((),), default={})
        for name in ("get_observation", "observation"):
            candidate = getattr(env, name, None)
            if callable(candidate):
                return _call_flexible(candidate, positional=((),), default={})
            if candidate is not None:
                return candidate
        return {}

    def _public_snapshot(self, env: Any) -> Any:
        for name in ("public_state", "snapshot_public", "serialize_public_state", "snapshot"):
            fn = _method(env, name)
            if fn:
                try:
                    return fn()
                except Exception:
                    continue
        return None

    def _plan(self, agent: Any, task: Any, observation: Any) -> Any:
        plan = _method(agent, "plan")
        if not plan:
            return None
        public_task = _public_task(task)
        return _call_flexible(plan, kwargs={"task": public_task, "observation": observation},
                              positional=((public_task, observation), (observation,), (public_task,), ()), default=None)

    def _act(self, agent: Any, plan: Any, observation: Any) -> Any:
        act = _method(agent, "act")
        if not act:
            # A scripted provider may return its action as its plan.
            return plan
        # Prefer semantically named kwargs.  If a provider accepts one
        # positional argument, pass the plan when present, otherwise the
        # observation.
        kwargs = {"plan": plan, "observation": observation, "action_plan": plan}
        positional = ((plan, observation), (plan,), (observation,), ()) if plan is not None else ((observation,), ())
        return _call_flexible(act, kwargs=kwargs, positional=positional, default=None)

    def _notify_observation(self, agent: Any, observation: Any) -> Any:
        observe = _method(agent, "observe")
        if not observe:
            return None
        return _call_flexible(observe, kwargs={"observation": observation}, positional=((observation,), ()), default=None)

    def _execute(self, env: Any, action: NormalizedAction) -> Any:
        if action.tool:
            for name in ("execute_tool", "invoke_tool", "call_tool", "tool"):
                fn = _method(env, name)
                if fn:
                    return _call_flexible(fn, kwargs={"name": action.tool, "tool": action.tool, "arguments": action.arguments, "args": action.arguments},
                                          positional=((action.tool, action.arguments), (action.tool,), (action.raw,)))
        for name in ("step", "execute_action", "execute", "act", "apply_action"):
            fn = _method(env, name)
            if fn:
                return _call_flexible(fn, kwargs={"action": action.raw}, positional=((action.raw,), ()))
        raise AttributeError("environment has no tool/action execution method")

    def _env_done(self, env: Any) -> Optional[bool]:
        for name in ("is_done", "done", "terminated"):
            value = getattr(env, name, None)
            if callable(value):
                try:
                    return bool(value())
                except Exception:
                    continue
            if value is not None:
                return bool(value)
        return None

    def _agent_reset(self, agent: Any, task: Any, seed: Optional[int]) -> None:
        reset = _method(agent, "reset")
        if reset:
            public_task = _public_task(task)
            _call_flexible(reset, kwargs={"task": public_task, "seed": seed}, positional=((public_task, seed), (public_task,), (seed,), ()))

    def _agent_finish(self, agent: Any, payload: Any) -> Any:
        finish = _method(agent, "finish")
        if not finish:
            return None
        try:
            return _call_flexible(finish, kwargs={"result": payload, "observation": payload}, positional=((payload,), ()), default=None)
        except Exception:
            return None

    def run(self, task: Any, agent: Any, *, environment: Any = None, seed: Optional[int] = None,
            agent_id: Optional[str] = None, run_id: Optional[str] = None,
            timeout: Optional[float] = None, max_steps: Optional[int] = None) -> RunResult:
        task_id = str(_task_value(task, "task_id", _task_value(task, "id", "task")))
        if seed is None:
            seed = _task_value(task, "seed", None)
        agent_id = agent_id or str(getattr(agent, "agent_id", getattr(agent, "name", agent.__class__.__name__)))
        run_id = run_id or self.run_id_factory()
        raw_timeout = timeout if timeout is not None else _task_value(task, "timeout", 30.0)
        # ``None`` means use a conservative default rather than failing before
        # the first action (core TaskDefinition permits an unlimited timeout).
        timeout_limit = 30.0 if raw_timeout is None else float(raw_timeout)
        if timeout_limit <= 0:
            timeout_limit = 30.0
        step_limit = int(max_steps if max_steps is not None else _task_value(task, "max_steps", 20))
        if step_limit < 1:
            step_limit = 1

        trace = AgentTrace(run_id=run_id, task_id=task_id, agent_id=agent_id, seed=seed)
        started = self.clock()
        env: Any = None
        observation: Any = None
        status = "error"
        success: Optional[bool] = None
        run_error: Optional[str] = None
        finish_result: Any = None
        try:
            env = self._make_environment(task, seed, supplied=environment)
            reset_value = self._reset_environment(env, task, seed)
            observation = reset_value if reset_value is not None else self._observe(env)
            self._agent_reset(agent, task, seed)
            self._notify_observation(agent, observation)

            for step_number in range(1, step_limit + 1):
                elapsed = self.clock() - started
                if elapsed >= timeout_limit:
                    status = "timeout"
                    trace.metadata["timeout_reached"] = True
                    break
                plan = self._plan(agent, task, observation)
                raw_action = self._act(agent, plan, observation)
                action = normalize_action(raw_action)
                if action.finish:
                    status = "finished"
                    trace.append(TraceStep(step=step_number, observation=observation, action=raw_action,
                                           tool=action.tool, arguments=action.arguments,
                                           public_reason_summary=action.public_reason_summary,
                                           action_type=action.action_type, latency=0.0,
                                           timestamp=None, state_snapshot=self._public_snapshot(env)))
                    break

                action_started = self.clock()
                result: Any = None
                error: Optional[str] = None
                is_error = False
                try:
                    result = self._execute(env, action)
                    result, error, is_error, result_done = _extract_tool_result(result)
                except Exception as exc:  # tool failures are part of the benchmark
                    error = f"{type(exc).__name__}: {exc}"
                    is_error = True
                    result_done = False
                latency = max(0.0, self.clock() - action_started)
                # Feed the public result back through the provider's optional
                # result hook.  Built-in RuleBasedAgent uses this to choose a
                # recovery action; custom providers can ignore the hook.
                record_result = _method(agent, "record_result") or _method(agent, "on_result") or _method(agent, "observe_result")
                if record_result:
                    try:
                        record_result(result, action=action.raw)
                    except TypeError:
                        try:
                            record_result(result)
                        except Exception:
                            pass
                    except Exception:
                        pass
                # A failed tool still produces an observation, allowing agents
                # to demonstrate recovery rather than aborting the run.
                try:
                    next_observation = self._observe(env)
                except Exception as exc:
                    next_observation = {"error": f"observation failed: {exc}"}
                    if error is None:
                        error = str(exc)
                    is_error = True
                trace.append(TraceStep(step=step_number, observation=observation, action=raw_action,
                                       tool=action.tool, arguments=action.arguments, result=result,
                                       public_reason_summary=action.public_reason_summary,
                                       error=error, latency=latency, timestamp=None,
                                       state_snapshot=self._public_snapshot(env), valid=not is_error,
                                       action_type=action.action_type))
                observation = next_observation
                self._notify_observation(agent, observation)
                done = result_done if result_done is not None else self._env_done(env)
                if done:
                    status = "finished"
                    break
            else:
                status = "timeout"
                trace.metadata["max_steps_reached"] = True

            if status == "error":
                status = "finished"
        except Exception as exc:
            run_error = f"{type(exc).__name__}: {exc}"
            status = "error"
        finally:
            try:
                finish_payload = {"status": status, "observation": observation, "error": run_error}
                finish_result = self._agent_finish(agent, finish_payload)
            except Exception:
                pass
            trace.final_observation = observation
            trace.final_state = self._public_snapshot(env) if env is not None else None
            trace.metadata.update({
                "timeout": timeout_limit,
                "max_steps": step_limit,
                "elapsed": max(0.0, self.clock() - started),
            })
            # Preserve provider-level public accounting (for example the
            # offline MockLLMAgent token estimate) without recording private
            # reasoning or prompt contents.
            if finish_result is not None:
                metrics = getattr(finish_result, "metrics", None)
                if isinstance(finish_result, Mapping):
                    metrics = finish_result.get("metrics", metrics)
                if isinstance(metrics, Mapping):
                    trace.metadata["agent_metrics"] = dict(metrics)
                    token_estimate = metrics.get("token_usage_estimate", metrics.get("token_usage"))
                    if isinstance(token_estimate, Mapping):
                        trace.metadata["token_usage"] = token_estimate.get("total_tokens", token_estimate.get("total", 0))
                    elif token_estimate is not None:
                        trace.metadata["token_usage"] = token_estimate
            trace.finish(status, final_observation=observation, final_state=trace.final_state, error=run_error)

        result = RunResult(run_id=run_id, task_id=task_id, agent_id=agent_id, seed=seed,
                           status=status, success=success, trace=trace,
                           final_observation=observation, final_state=trace.final_state,
                           error=run_error)
        if self.evaluator is not None:
            try:
                evaluated = self.evaluator.evaluate(task, result, environment=env)
                if hasattr(evaluated, "apply_to"):
                    evaluated.apply_to(result)
                elif isinstance(evaluated, Mapping):
                    result.success = evaluated.get("success", result.success)
                    result.score = float(evaluated.get("score", result.score))
                    result.metrics.update(dict(evaluated.get("metrics", {})))
                    result.failures.extend(str(v) for v in evaluated.get("failures", []))
                else:
                    result.success = getattr(evaluated, "success", result.success)
                    result.score = float(getattr(evaluated, "score", result.score))
                    result.metrics.update(dict(getattr(evaluated, "metrics", {}) or {}))
                    result.failures.extend(str(v) for v in getattr(evaluated, "failures", []) or [])
            except Exception as exc:
                result.metrics["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        return result


def run_task(task: Any, agent: Any, environment: Any = None, **kwargs: Any) -> RunResult:
    """Convenience wrapper around :class:`TaskRunner`."""

    return TaskRunner(environment=environment, environment_factory=kwargs.pop("environment_factory", None),
                      evaluator=kwargs.pop("evaluator", None)).run(task, agent, **kwargs)


__all__ = ["TaskRunner", "RunResult", "NormalizedAction", "normalize_action", "run_task"]
