import json
import tempfile
import unittest
from pathlib import Path

from agent_arena.demo_dataset import dataset_summary, load_dataset, save_dataset
from agent_arena.task_environment import environment_spec, make_environment_for_task
from agent_arena.benchmark import make_task_environment, scripted_actions_for_task
from agent_arena.core import ScriptedAgent
from agent_arena.evaluator import Evaluator
from agent_arena.runner import TaskRunner
from agent_arena.task_catalog import (
    ADVERSARIAL_VARIANTS,
    BUILTIN_CATEGORIES,
    build_task,
    generate_demo_tasks,
    generate_tasks,
    make_adversarial_variant,
)
from agent_arena.tasking import TaskDefinition, tasks_from_json, tasks_to_json


class TaskCatalogTests(unittest.TestCase):
    def test_each_builtin_category_has_expected_contract(self):
        for index, category in enumerate(BUILTIN_CATEGORIES):
            task = build_task(category, seed=index, difficulty="medium")
            self.assertEqual(task.category, category)
            self.assertTrue(task.task_id)
            self.assertTrue(task.instruction)
            self.assertTrue(task.initial_state)
            self.assertTrue(task.allowed_tools)
            self.assertTrue(task.hidden_ground_truth)
            self.assertTrue(task.success_conditions)
            self.assertGreater(task.max_steps, 0)
            # Hidden fields are separate objects; mutating public state must
            # not alter evaluator-only data.
            before = json.dumps(task.hidden_ground_truth, sort_keys=True)
            task.initial_state.setdefault("_test", True)
            self.assertEqual(before, json.dumps(task.hidden_ground_truth, sort_keys=True))

    def test_generation_is_deterministic_and_balanced(self):
        first = generate_demo_tasks(50, seed=1234, adversarial_rate=0.0)
        second = generate_demo_tasks(50, seed=1234, adversarial_rate=0.0)
        self.assertEqual([t.to_dict() for t in first], [t.to_dict() for t in second])
        summary = dataset_summary(first)
        self.assertEqual(summary["count"], 50)
        self.assertEqual(set(summary["categories"]), set(BUILTIN_CATEGORIES))
        self.assertEqual(set(summary["categories"].values()), {10})

    def test_adversarial_variants_preserve_goal(self):
        base = build_task("email", seed=42, difficulty="easy")
        goal = base.hidden_ground_truth
        for variant in ADVERSARIAL_VARIANTS:
            altered = make_adversarial_variant(base, variant)
            self.assertEqual(altered.hidden_ground_truth, goal)
            self.assertIn("adversarial", altered.tags)
            self.assertIn(variant, altered.metadata["adversarial"])
            self.assertNotEqual(altered.task_id, base.task_id)

    def test_json_round_trip(self):
        tasks = generate_tasks(7, seed=99)
        loaded = tasks_from_json(tasks_to_json(tasks))
        self.assertEqual([t.to_dict() for t in tasks], [t.to_dict() for t in loaded])

    def test_dataset_file_round_trip(self):
        tasks = generate_demo_tasks(5, seed=5, adversarial_rate=0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.json"
            save_dataset(tasks, path)
            loaded = load_dataset(path)
            self.assertEqual([t.task_id for t in tasks], [t.task_id for t in loaded])

    def test_task_definition_accepts_legacy_id(self):
        task = TaskDefinition.from_dict({"id": "legacy", "instruction": "noop"})
        self.assertEqual(task.task_id, "legacy")

    def test_generated_tasks_map_to_safe_in_memory_environments(self):
        for task in generate_demo_tasks(5, seed=77, adversarial_rate=0):
            kind, kwargs = environment_spec(task)
            self.assertEqual(kind, task.initial_state["type"])
            environment = make_environment_for_task(task)
            self.assertIn("available_tools", environment.observe())
            # Constructor/reset must consume the public task state without
            # touching any host resource.
            environment.reset(initial_state=task.initial_state)

    def test_hard_grid_and_calendar_instances_are_always_solvable(self):
        for seed in range(100):
            grid = build_task("grid", seed=seed, difficulty="hard")
            self.assertTrue(grid.hidden_ground_truth["shortest_path"], seed)
            calendar = build_task("calendar", seed=seed, difficulty="hard")
            proposed = calendar.hidden_ground_truth["proposed_event"]
            self.assertLess(proposed["start"], proposed["end"])

    def test_file_tasks_never_duplicate_a_source_path(self):
        for seed in range(200):
            for difficulty in ("easy", "medium", "hard"):
                task = build_task("files", seed=seed, difficulty=difficulty)
                sources = [move["from"] for move in task.hidden_ground_truth["moves"]]
                self.assertEqual(len(sources), len(set(sources)), (seed, difficulty, sources))
                self.assertEqual(len(sources), len(task.initial_state["files"]))

    def test_reference_script_completes_one_task_per_category(self):
        for task in generate_demo_tasks(5, seed=88, adversarial_rate=0):
            env = make_task_environment(task)
            agent = ScriptedAgent(scripted_actions_for_task(task), name="catalog-script")
            run = TaskRunner(evaluator=Evaluator()).run(task, agent, environment=env, seed=88)
            self.assertTrue(run.success, (task.category, run.failures, run.error))


if __name__ == "__main__":
    unittest.main()
