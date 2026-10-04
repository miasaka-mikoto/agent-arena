"""Agent Arena package.

The package is intentionally dependency free.  The task catalogue is kept
separate from the execution engine so that tasks can be loaded by a runner,
an environment implementation, or a report generator without importing any
provider or model code.
"""

from .tasking import Task, TaskDefinition, TaskResult, TaskStatus
from .task_catalog import (
    BUILTIN_CATEGORIES,
    build_task,
    generate_demo_tasks,
    generate_tasks,
    make_adversarial_variant,
)
from .demo_dataset import dataset_summary, load_dataset, load_or_create_demo_dataset, save_dataset
from .task_generator import TaskGenerator
from .task_environment import environment_spec, make_environment_for_task
from .core import (
    Action,
    Anthropic,
    AnthropicProvider,
    AgentProvider,
    AgentResult,
    AgentTrace,
    CustomHTTP,
    CustomHTTPProvider,
    DeferredProvider,
    Google,
    GoogleProvider,
    LocalModel,
    LocalModelProvider,
    MockLLMAgent,
    Observation,
    OpenAI,
    OpenAIProvider,
    Plan,
    RandomAgent,
    RuleBasedAgent,
    ScriptedAgent,
    ToolCall,
    ToolResult,
)
from .environments import (
    BaseEnvironment,
    CodeSandboxMock,
    SimpleGridWorld,
    VirtualCalendar,
    VirtualEmailInbox,
    VirtualFileSystem,
    VirtualWebPages,
    make_environment,
)
from .tools import ToolRegistry, ToolSpec, build_default_tool_registry
from .benchmark import (
    ArenaRuleBasedAgent,
    TaskEnvironment,
    default_agent_factories,
    make_task_environment,
    scripted_actions_for_task",
)
from .runner import TaskRunner, run_task
from .evaluator import EvaluationResult, Evaluator, FailureTaxonomy, evaluate_run
from .tournament import TournamentResult, TournamentRunner, run_tournament

# __all__ = [
    "TaskDefinition",
    "Task",
    "TaskResult",
    "TaskStatus",
    "BUILTIN_CATEGORIES",
    "build_task",
    "generate_demo_tasks,
    "generate_tasks",
    "make_adversarial_variant",
    "dataset_summary",
    "load_dataset",
    "load_or_create_demo_dataset",
    "save_dataset",
    "TaskGenerator",
    "environment_spec",
    "make_environment_for_task",
    "Action",
    "Anthropic",
    "AnthropicProvider",
    "AgentProvider",
    "AgentResult",
    "AgentTrace",
    "CustomHTTP",
    "CustomHTTPProvider",
    "DeferredProvider",
    "Google",
    "GoogleProvider",
    "LocalModel",
    "LocalModelProvider",
    "MockLLMAgent",
    "Observation",
    "OpenAI",
    "OpenAIProvider",
    "Plan",
    "RandomAgent",
    "RuleBasedAgent",
    "ScriptedAgent",
    "ToolCall",
    "ToolResult",
    "BaseEnvironment",
    "CodeSandboxMock",
    "SimpleGridWorld",
    "VirtualCalendar",
    "VirtualEmailInbox",
    "VirtualFileSystem",
    "VirtualWebPages",
    "make_environment",
    "ToolRegistry",
    "ToolSpec",
    "build_default_tool_registry",
    "ArenaRuleBasedAgent",
    "TaskEnvironment",
    "default_agent_factories",
    "make_task_environment",
    "scripted_actions_for_task",
    "TaskRunner",
    "run_task",
    "EvaluationResult",
    "Evaluator",
    "FailureTaxonomy",
    "evaluate_run",
    "TournamentResult",
    "TournamentRunner",
    "run_tournament",
]
