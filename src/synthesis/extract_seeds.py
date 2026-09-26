"""Phase 0 (optional): build seed trajectories from the stock tau2 tasks.

Each sampled task's gold actions become a trajectory marked ``is_seed_data``.
Seeds are shown to the explorer as examples; they are never summarized.
Pre-built seeds ship in ``synthesis/assets/<domain>/seed_trajectories.json``.

Example:
    python -m synthesis.extract_seeds --domain airline --num-tasks 20 \\
        --output synthesis/assets/airline/seed_trajectories.json
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

from synthesis.common import DOMAINS, TAU2_DOMAINS_DIR, load_json, save_json


def task_to_seed(task: dict, index: int) -> dict:
    now = datetime.now().isoformat()
    instructions = task["user_scenario"]["instructions"]
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    steps = []
    for i, action in enumerate(actions, 1):
        call_id = f"seed_{task['id']}_{action.get('action_id', i)}"
        steps.append({
            "step_number": i,
            "timestamp": now,
            "tool_call": {"id": call_id, "name": action["name"],
                          "arguments": action.get("arguments", {}), "requestor": "assistant"},
            "tool_response": {"role": "tool", "content": "Response not available in seed data",
                              "id": call_id, "requestor": "assistant", "error": False},
            "llm_reasoning": None,
            "available_tools": [],
            "execution_status": "success",
            "error_message": None,
        })
    tool_usage: dict[str, int] = {}
    for step in steps:
        tool_usage[step["tool_call"]["name"]] = tool_usage.get(step["tool_call"]["name"], 0) + 1
    return {
        "trajectory_id": f"traj_{index:04d}",
        "domain_name": instructions.get("domain"),
        "exploration_strategy": "seed_data",
        "max_steps": len(actions),
        "steps": steps,
        "start_time": now,
        "end_time": now,
        "termination_reason": "max_steps_reached",
        "total_cost": 0.0,
        "goal_description": instructions.get("reason_for_call"),
        "user_info": {"source": "task_extraction", "known_info": instructions.get("known_info")},
        "statistics": {"total_steps": len(steps), "success_rate": 1.0, "tool_usage": tool_usage},
        "data_source": "seed",
        "is_seed_data": True,
        "source_task_id": task["id"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=DOMAINS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-tasks", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    tasks = load_json(TAU2_DOMAINS_DIR / args.domain / "tasks.json")
    sampled = random.sample(tasks, min(args.num_tasks, len(tasks)))
    seeds = [task_to_seed(task, i) for i, task in enumerate(sampled)]
    save_json(seeds, args.output)
    print(f"Wrote {len(seeds)} seed trajectories to {args.output}")


if __name__ == "__main__":
    main()
