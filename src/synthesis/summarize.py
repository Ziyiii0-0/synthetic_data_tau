"""Phase 2: summarize exploration trajectories into tau2 tasks.

For each trajectory an LLM writes the user side of the task (reason for call,
known / unknown info, ...) given the observed tool calls. The trajectory's
successful tool calls become the gold ``evaluation_criteria.actions``.

Trajectory selection:
  * seed trajectories are skipped;
  * only trajectories with a state-changing call (WRITE tool, ``calculate`` or
    ``transfer_to_human_agents``) are used;
  * ``infeasible`` additionally requires a ``transfer_to_human_agents`` call;
  * steps from the first transfer onwards, and repeated ``find_user_id_by_*``
    lookups, are dropped.

The output is rewritten after every task; re-running resumes.

Example:
    python -m synthesis.summarize --trajectories outputs/airline/trajectories.json \\
        --task-type general --output outputs/airline/tasks_general.json
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger

from synthesis.common import (
    DEFAULT_MODEL,
    TASK_TYPES,
    base_domain,
    get_write_actions,
    llm_json,
    load_json,
    save_json,
)
from synthesis.prompts.summarization import SUMMARIZATION_PROMPTS, TRAJECTORY_HEADER

TRANSFER = "transfer_to_human_agents"


def clean_steps(trajectory: dict) -> Optional[dict]:
    """Cut at the first transfer and drop repeated ``find_user_id_by_*`` lookups."""
    steps = trajectory.get("steps", [])
    starts_with_lookup = bool(steps) and steps[0]["tool_call"]["name"].startswith("find_user_id_by")
    cleaned = []
    for i, step in enumerate(steps):
        name = step["tool_call"]["name"]
        if name == TRANSFER:
            break
        if i > 0 and starts_with_lookup and name.startswith("find_user_id_by"):
            continue
        cleaned.append(step)
    return {**trajectory, "steps": cleaned} if cleaned else None


def select_trajectories(trajectories: list[dict], domain: str, task_type: str) -> list[dict]:
    required_tools = get_write_actions(domain, include_generic=True)
    selected = []
    for trajectory in trajectories:
        names = {step["tool_call"]["name"] for step in trajectory.get("steps", [])}
        if not names & required_tools:
            continue
        if task_type == "infeasible" and TRANSFER not in names:
            continue
        cleaned = clean_steps(trajectory)
        if cleaned is not None:
            selected.append(cleaned)
    return selected


def format_user_info(user: dict, domain: str) -> str:
    name = user.get("name") or {}
    full_name = f"{name.get('first_name', '')} {name.get('last_name', '')}".strip()
    if domain == "airline":
        fields = [("Name", full_name), ("User ID", user.get("user_id")), ("Email", user.get("email"))]
    else:
        zipcode = (user.get("address") or {}).get("zip")
        fields = [("Name", full_name), ("Email", user.get("email")), ("Zipcode", zipcode)]
    return "\n".join(f"- {label}: {value or 'N/A'}" for label, value in fields)


def build_prompt(trajectory: dict, domain: str, task_type: str) -> str:
    steps = "\n".join(
        f"Step {i}: Called {step['tool_call']['name']} with arguments "
        f"{json.dumps(step['tool_call'].get('arguments', {}))}\n"
        f"  Response: {json.dumps(step.get('tool_response', {}))}"
        for i, step in enumerate(trajectory["steps"], 1)
    )
    header = TRAJECTORY_HEADER.format(
        domain=domain,
        user_info=format_user_info(trajectory.get("user_info") or {}, domain),
        steps=steps,
    )
    return header + SUMMARIZATION_PROMPTS[(domain, task_type)]


def build_task(trajectory: dict, scenario: dict, domain: str, write_actions: set[str]) -> dict:
    """Assemble a tau2 task from the LLM's user scenario and the trajectory."""
    instructions = {
        "task_instructions": scenario.get("task_instructions") or ".",
        "domain": domain,
        "reason_for_call": scenario.get("reason_for_call", ""),
        "known_info": scenario.get("known_info", ""),
        "unknown_info": scenario.get("unknown_info", ""),
    }
    if "infeasible_reason" in scenario:
        instructions["infeasible_reason"] = scenario["infeasible_reason"]
    forbidden = [
        a for a in scenario.get("actions_should_not_taken") or [] if a.get("tool_name") in write_actions
    ]
    if forbidden:
        instructions["actions_should_not_taken"] = forbidden
    if "actions_should_be_taken" in scenario:
        instructions["actions_should_be_taken"] = scenario["actions_should_be_taken"]

    actions = [
        {
            "action_id": f"0_{i}",
            "name": step["tool_call"]["name"],
            "arguments": step["tool_call"].get("arguments", {}),
            "info": None,
        }
        for i, step in enumerate(trajectory["steps"])
        if step.get("execution_status") == "success"
    ]
    return {
        "id": trajectory["trajectory_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "description": {"purpose": None, "relevant_policies": None, "notes": None},
        "user_scenario": {"persona": None, "instructions": instructions},
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "communicate_info": scenario.get("direct_communication_info") or [],
            "nl_assertions": scenario.get("nl_assertions") or None,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", type=Path, required=True, help="Output of synthesis.explore.")
    p.add_argument("--task-type", choices=TASK_TYPES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-base", default=None, help="OpenAI-compatible endpoint for a local model.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=None, help="Truncate trajectories to this many steps.")
    p.add_argument("--limit", type=int, default=None, help="Only summarize the first N selected trajectories.")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    trajectories = [t for t in load_json(args.trajectories) if not t.get("is_seed_data")]
    if not trajectories:
        raise SystemExit(f"No non-seed trajectories in {args.trajectories}")
    domain = base_domain(trajectories[0]["domain_name"])
    if args.max_steps:
        trajectories = [{**t, "steps": t["steps"][: args.max_steps]} for t in trajectories]
    selected = select_trajectories(trajectories, domain, args.task_type)[: args.limit]
    logger.info(f"{len(selected)}/{len(trajectories)} {domain} trajectories selected for '{args.task_type}'")

    tasks = [] if args.no_resume or not args.output.exists() else load_json(args.output)
    done = {t["trajectory_id"] for t in tasks}
    todo = [t for t in selected if t["trajectory_id"] not in done]
    logger.info(f"{len(done)} already summarized, {len(todo)} to go")

    write_actions = get_write_actions(domain)
    lock = threading.Lock()

    def summarize(trajectory: dict) -> dict:
        prompt = build_prompt(trajectory, domain, args.task_type)
        scenario = llm_json(prompt, args.model, args.temperature, args.api_base)
        return build_task(trajectory, scenario, domain, write_actions)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(summarize, t): t["trajectory_id"] for t in todo}
        for future in as_completed(futures):
            traj_id = futures[future]
            try:
                task = future.result()
            except Exception as e:
                failed += 1
                logger.error(f"{traj_id}: {e}")
                continue
            with lock:
                tasks.append(task)
                save_json(tasks, args.output)
            logger.info(f"[{len(tasks)}] {traj_id}")

    logger.info(f"Done: {len(tasks)} tasks in {args.output} ({failed} failed)")


if __name__ == "__main__":
    main()
