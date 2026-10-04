"""Small end-to-end smoke benchmark over the safe simulators."""

from agent_arena.benchmark import TaskEnvironment, scripted_actions_for_task
from agent_arena.core import ScriptedAgent
from agent_arena.evaluator import Evaluator
from agent_arena.runner import TaskRunner
from agent_arena.task_catalog import generate_demo_tasks


def test_scripted_demo_completes_all_categories_and_adversarial_variants() -> None:
    tasks = generate_demo_tasks(count=50, seed=20261004, adversarial_rate=0.30)
    runner = TaskRunner(evaluator=Evaluator())
    outcomes = []
    for task in tasks:
        env = TaskEnvironment(task)
        agent = ScriptedAgent(scripted_actions_for_task(task), name="ScriptedAgent")
        result = runner.run(task, agent, environment=env, seed=task.seed)
        outcomes.append(result.success)
    assert len(outcomes) == 50
    # A transparent scripted baseline intentionally does not retry every
    # injected empty/failing tool; adversarial cases should therefore remain
    # visible as failures rather than being silently marked as passes.
    assert sum(bool(value) for value in outcomes) >= 45
