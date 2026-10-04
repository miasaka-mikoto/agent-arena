# Agent Arena Demo Tournament

This directory is a reproducible offline benchmark result.

- Dataset: 50 tasks, five categories, dataset seed `20261004`
- Agents: `RandomAgent`, `RuleBasedAgent`, `MockLLMAgent`
- Seeds: `0`, `1`, `2` (450 runs total)
- Adversarial task rate: `0.30`
- All provider results are synthetic/offline baselines; no paid API or real account was used.

## Files

- `tasks.json`: task definitions (hidden ground truth is not exposed to agents)
- `results.json`: complete public traces and evaluation results
- `results.sqlite`: local queryable result database
- `summary.json`: aggregate metrics, leaderboard, categories, and failure breakdown
- `report.html`: dashboard
- `replay.html`: standalone replay of the first run
- `replays/`: one standalone replay page per run

Regenerate with:

```bash
python -m agent_arena demo --tasks 50 --dataset-seed 20261004 --seeds 0 1 2 --adversarial-rate 0.30 --output artifacts/demo
```
