"""Tool registration and execution for Agent Arena.

Tools are the only intended bridge between an agent and an environment.  The
registry below is intentionally small and explicit: it has no reflection over
the host machine, no network client, and no shell execution.  Environment
instances are bound at construction time, which makes a benchmark trace easy
to audit and replay.
"""

from __future__ import annotations

import ast
import copy
import inspect
import math
import operator
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from .environments import BaseEnvironment, LocalToolResult, _result


@dataclass(frozen=True)
class ToolSpec:
    """A serialisable description of one callable tool."""

    name: str
    description: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    category: str = "environment"
    mutating: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "category": self.category,
            "mutating": self.mutating,
        }

    @property
    def schema(self) -> Mapping[str, Any]:
        """Input-schema alias used by provider adapters."""

        return self.arguments

    @property
    def input_schema(self) -> Mapping[str, Any]:
        return self.arguments


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[..., Any]
    environment: Any = None


class ToolRegistry:
    """Explicit allowlist of tools exposed to an agent.

    A registry may be bound to one environment or to a collection of
    environments.  ``invoke`` always returns a ToolResult-like object and
    catches handler exceptions, so one malformed action cannot crash a run.
    """

    def __init__(self, environment: Any | None = None, environments: Mapping[str, Any] | None = None) -> None:
        self.environment = environment
        self._tools: dict[str, RegisteredTool] = {}
        self._owners: dict[str, Any] = {}
        if environments:
            for env in environments.values():
                self.add_environment(env)
        elif environment is not None:
            self.add_environment(environment)

    def register(
        self,
        name: str | ToolSpec,
        handler: Callable[..., Any] | None = None,
        *,
        description: str = "",
        arguments: Mapping[str, Any] | None = None,
        category: str = "environment",
        mutating: bool = False,
        environment: Any | None = None,
        replace: bool = False,
    ) -> ToolSpec:
        if isinstance(name, ToolSpec):
            spec = name
            actual_name = spec.name
            if handler is None:
                raise ValueError("handler is required when registering a ToolSpec")
        else:
            actual_name = str(name).strip()
            if not actual_name:
                raise ValueError("tool name cannot be empty")
            spec = ToolSpec(actual_name, description or actual_name.replace("_", " ").title(), arguments or {}, category, mutating)
        if handler is None or not callable(handler):
            raise ValueError(f"handler is required for tool {actual_name}")
        if actual_name in self._tools and not replace:
            raise ValueError(f"tool already registered: {actual_name}")
        owner = environment if environment is not None else self.environment
        self._tools[actual_name] = RegisteredTool(spec, handler, owner)
        if owner is not None:
            self._owners[actual_name] = owner
        return spec

    def unregister(self, name: str) -> None:
        self._tools.pop(str(name), None)
        self._owners.pop(str(name), None)

    def add_environment(self, environment: Any, *, include_calculator: bool = False) -> list[ToolSpec]:
        """Register an environment's allowlisted methods.

        ``available_tools`` is authoritative; arbitrary methods on an
        environment are never exposed implicitly.
        """

        added: list[ToolSpec] = []
        names = tuple(getattr(environment, "available_tools", ()))
        for name in names:
            handler = getattr(environment, name, None)
            # Route through execute_tool when available.  This preserves the
            # environment's step/error counters and keeps the registry from
            # accidentally bypassing its safety boundary (EnvironmentBundle
            # intentionally exposes names but not one method per name).
            if callable(getattr(environment, "execute_tool", None)):
                tool_name = str(name)

                def invoke_bound(_tool_name: str = tool_name, **kwargs: Any) -> Any:
                    return environment.execute_tool(_tool_name, kwargs)

                handler = invoke_bound
            if not callable(handler):
                continue
            if name in self._tools:
                # A composite benchmark may intentionally have two copies of
                # an environment.  Names are global, so the caller should use
                # a namespaced custom name in that case.
                continue
            mutating = name not in {"list_files", "read_file", "search_files", "list_mail", "search_mail", "read_mail", "calendar_lookup", "calendar_find_conflicts", "browser_open", "browser_search", "sandbox_list_files", "sandbox_read_file"}
            spec = ToolSpec(name, name.replace("_", " ").title(), {}, getattr(environment, "name", "environment"), mutating)
            self.register(spec, handler, environment=environment)
            added.append(spec)
        if include_calculator and "calculator" not in self._tools:
            added.append(self.register_calculator())
        return added

    def register_calculator(self) -> ToolSpec:
        if "calculator" in self._tools:
            return self._tools["calculator"].spec
        return self.register(
            ToolSpec(
                "calculator",
                "Evaluate a bounded arithmetic expression without executing code",
                {"expression": {"type": "string", "required": True}, "variables": {"type": "object", "required": False}},
                "utility",
                False,
            ),
            _calculator,
        )

    def add_alias(self, alias: str, target: str, *, description: str | None = None, replace: bool = False) -> ToolSpec:
        if target not in self._tools:
            raise KeyError(target)
        if alias in self._tools and not replace:
            raise ValueError(f"tool already registered: {alias}")
        target_tool = self._tools[target]

        def call_alias(**kwargs: Any) -> Any:
            return target_tool.handler(**kwargs)

        spec = ToolSpec(alias, description or f"Alias for {target}", target_tool.spec.arguments, target_tool.spec.category, target_tool.spec.mutating)
        self.register(spec, call_alias, environment=target_tool.environment, replace=replace)
        return spec

    def list_tools(self) -> list[dict[str, Any]]:
        return [self._tools[name].spec.as_dict() for name in sorted(self._tools)]

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in sorted(self._tools)]

    def has(self, name: str) -> bool:
        return str(name) in self._tools

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        args: Mapping[str, Any] | None = None,
    ) -> Any:
        tool_name = str(name)
        call_args = dict(arguments if arguments is not None else (args or {}))
        started = time.perf_counter()
        registered = self._tools.get(tool_name)
        if registered is None:
            return _result(tool_name, False, error=f"Unknown tool: {tool_name}", started=started, metadata={"registry": True})
        try:
            output = registered.handler(**call_args)
            if hasattr(output, "success") and hasattr(output, "tool"):
                return output
            return _result(tool_name, True, output=output, started=started, metadata={"registry": True, "environment": getattr(registered.environment, "name", None)})
        except TypeError as exc:
            return _result(tool_name, False, error=f"Invalid arguments: {exc}", started=started, metadata={"registry": True})
        except Exception as exc:
            return _result(tool_name, False, error=f"Tool error: {exc}", started=started, metadata={"registry": True})

    # Common names used by different runner revisions.
    execute = invoke
    call = invoke
    execute_tool = invoke

    def get(self, name: str, default: RegisteredTool | None = None) -> RegisteredTool | None:
        return self._tools.get(str(name), default)

    def invoke_action(self, action: Any) -> Any:
        if isinstance(action, str):
            return self.invoke(action, {})
        if isinstance(action, Mapping):
            name = action.get("tool") or action.get("name") or action.get("action")
            arguments = action.get("arguments") or action.get("args") or {}
        else:
            name = getattr(action, "tool", None) or getattr(action, "name", None)
            arguments = getattr(action, "arguments", None) or getattr(action, "args", None) or {}
        if not name:
            return _result("", False, error="Action did not specify a tool", metadata={"registry": True})
        return self.invoke(str(name), arguments)

    execute_action = invoke_action

    def add_standard_aliases(self) -> None:
        """Add short aliases used in natural-language task templates."""

        aliases = {
            "move": "grid_move",
            "turn": "grid_turn",
            "pick_up": "grid_pick_up",
            "drop": "grid_drop",
            "use": "grid_use",
            "interact": "grid_interact",
            "wait": "grid_wait",
            "open_url": "browser_open",
        }
        for alias, target in aliases.items():
            if target in self._tools and alias not in self._tools:
                self.add_alias(alias, target)


class EnvironmentBundle(BaseEnvironment):
    """A safe composite environment containing several virtual worlds."""

    name = "arena_bundle"

    def __init__(self, environments: Mapping[str, Any] | Sequence[Any]) -> None:
        # Do not call BaseEnvironment.__init__ indirectly through a child.
        super().__init__()
        if isinstance(environments, Mapping):
            self.environments = dict(environments)
        else:
            self.environments = {}
            for env in environments:
                key = getattr(env, "name", env.__class__.__name__.lower())
                self.environments[key] = env
        self._tool_owners: dict[str, Any] = {}
        for env in self.environments.values():
            for name in getattr(env, "available_tools", ()):
                self._tool_owners.setdefault(name, env)
        self.available_tools = tuple(sorted(self._tool_owners))

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        super().reset(seed)
        for env in self.environments.values():
            reset = getattr(env, "reset", None)
            if callable(reset):
                try:
                    reset(seed=seed)
                except TypeError:
                    reset(seed)
        return self.observe()

    def observe(self) -> Mapping[str, Any]:
        return {"environment": self.name, "step": self.step_count, "environments": {key: dict(env.observe()) for key, env in self.environments.items()}, "available_tools": list(self.available_tools)}

    def execute_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        owner = self._tool_owners.get(str(name))
        if owner is None:
            self.step_count += 1
            self.invalid_actions += 1
            return _result(str(name), False, error=f"Unknown tool: {name}", metadata={"environment": self.name})
        self.step_count += 1
        result = owner.execute_tool(str(name), arguments or {})
        if not getattr(result, "success", False):
            self.invalid_actions += 1
        return result

    def execute_action(self, action: Any) -> Any:
        if isinstance(action, Mapping):
            name = action.get("tool") or action.get("name") or action.get("action")
            args = action.get("arguments") or action.get("args") or {}
        else:
            name = getattr(action, "tool", None) or getattr(action, "name", None) or action
            args = getattr(action, "arguments", None) or getattr(action, "args", None) or {}
        return self.execute_tool(str(name), args)

    def ground_truth(self) -> Mapping[str, Any]:
        return {"environment": self.name, "environments": {key: dict(env.ground_truth()) for key, env in self.environments.items()}, "step": self.step_count}


# ---------------------------------------------------------------------------
# Safe calculator


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST, variables: Mapping[str, float], depth: int = 0) -> float:
    if depth > 32:
        raise ValueError("expression is too deeply nested")
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left, variables, depth + 1), _safe_eval(node.right, variables, depth + 1)
        if abs(float(left)) > 1e100 or abs(float(right)) > 1e100:
            raise ValueError("numeric value is too large")
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent is too large")
        value = _BIN_OPS[type(node.op)](left, right)
        if not math.isfinite(float(value)) or abs(float(value)) > 1e100:
            raise ValueError("result is not finite")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand, variables, depth + 1))
    raise ValueError("only numeric arithmetic is allowed")


def _calculator(expression: str, variables: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    text = str(expression).strip()
    if not text or len(text) > 500:
        raise ValueError("expression must contain 1-500 characters")
    names: dict[str, float] = {}
    for key, value in (variables or {}).items():
        if not isinstance(key, str) or not key.isidentifier() or not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("variables must map identifiers to numbers")
        names[key] = float(value)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("expression is too complex")
    value = _safe_eval(tree, names)
    return {"expression": text, "value": value}


def build_default_tool_registry(
    environment: Any | None = None,
    *,
    environments: Mapping[str, Any] | Sequence[Any] | None = None,
    include_calculator: bool = True,
    aliases: bool = True,
) -> ToolRegistry:
    """Build an allowlisted registry for one or more simulated environments."""

    if environments is not None:
        registry = ToolRegistry()
        iterable = environments.values() if isinstance(environments, Mapping) else environments
        for env in iterable:
            registry.add_environment(env)
    else:
        registry = ToolRegistry(environment)
        if environment is not None and not registry._tools:
            registry.add_environment(environment)
    if include_calculator:
        registry.register_calculator()
    if aliases:
        registry.add_standard_aliases()
    return registry


__all__ = [
    "ToolSpec",
    "RegisteredTool",
    "ToolRegistry",
    "EnvironmentBundle",
    "build_default_tool_registry",
]
