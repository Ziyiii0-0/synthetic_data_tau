"""Phase 3: score every (task, exploration trajectory) pair with an LLM judge.

The judge rates realism, necessity and correctness (plus infeasibility
reasonability for infeasible tasks, and a pass/fail "does the intent change"
gate for changing tasks). ``overall_score`` is the mean of the scored criteria
and is what ``synthesis.filter_tasks`` thresholds.

The report is rewritten after every task; re-running resumes.

Example:
    python -m synthesis.validate --tasks outputs/airline/tasks_general.json \\
        --trajectories outputs/airline/trajectories.json \\
        --output outputs/airline/validity_general.json
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger

from synthesis.common import DEFAULT_MODEL, TASK_TYPES, llm_json, load_json, load_policy, save_json
from synthesis.prompts.validity import AIRLINE_TIME_BLOCK, JUDGE_PROMPTS, SCORED_CRITERIA


def detect_task_type(task: dict) -> str:
    """``changing`` tasks look like ``general`` ones, so pass --task-type for them."""
    instructions = task["user_scenario"]["instructions"]
    return "infeasible" if instructions.get("infeasible_reason") else "general"


def format_trajectory(trajectory: dict) -> str:
    blocks = []
    for i, step in enumerate(trajectory.get("steps", []), 1):
        tool_call = step.get("tool_call", {})
        content = (step.get("tool_response") or {}).get("content", "")
        try:
            response = json.dumps(json.loads(content) if content else {}, indent=4)
        except (json.JSONDecodeError, TypeError):
            response = str(content)
        blocks.append(
            f"\nStep {i}:"
            f"\n  Action: {tool_call.get('name', 'unknown')}"
            f"\n  Arguments: {json.dumps(tool_call.get('arguments', {}), indent=4)}"
            f"\n  Response: {response[:500]}..."
            f"\n  Status: {step.get('execution_status', 'unknown')}"
        )
    return "\n".join(blocks)


def format_actions(actions: Optional[list[dict]]) -> str:
    lines = []
    for action in actions or []:
        line = f"  - {action.get('tool_name', 'unknown')}"
        if action.get("arguments"):
            line += f" with arguments: {json.dumps(action['arguments'])}"
        lines.append(line + "\n")
    return "".join(lines)


def build_judge_prompt(task: dict, trajectory: dict, task_type: str) -> str:
    instructions = task["user_scenario"]["instructions"]
    domain = instructions.get("domain", "N/A")
    fields = {
        "domain": domain,
        "reason_for_call": instructions.get("reason_for_call", "N/A"),
        "known_info": instructions.get("known_info", "N/A"),
        "unknown_info": instructions.get("unknown_info", "N/A"),
        "task_instructions": instructions.get("task_instructions", "N/A"),
        "trajectory_text": format_trajectory(trajectory),
        "communication_text": (task.get("evaluation_criteria") or {}).get("communicate_info", []),
    }
    if task_type == "infeasible":
        fields.update(
            domain_policy=load_policy(domain),
            infeasible_reason=instructions.get("infeasible_reason", "N/A"),
            forbidden_actions_text=format_actions(instructions.get("actions_should_not_taken")),
            required_actions_text=format_actions(instructions.get("actions_should_be_taken")),
        )
    else:
        fields["domain_time_block"] = AIRLINE_TIME_BLOCK if "airline" in domain.lower() else ""
    return JUDGE_PROMPTS[task_type].format(**fields)


def overall_score(evaluations: dict, task_type: str) -> float:
    if task_type == "changing":
        relevance = evaluations.get("changing_scenario_relevance", {}).get("score", True)
        if relevance in (False, "False", "false"):
            return 0.0
    scores = [evaluations.get(c, {}).get("score", 0) for c in SCORED_CRITERIA[task_type]]
    return round(sum(scores) / len(scores), 2)


def judge(task: dict, trajectory: dict, task_type: str, model: str, api_base: Optional[str]) -> dict:
    prompt = build_judge_prompt(task, trajectory, task_type)
    result = llm_json(prompt, model, api_base=api_base)
    evaluations = result.get("evaluations", {})
    return {
        "task_id": task.get("id") or task["trajectory_id"],
        "trajectory_id": trajectory["trajectory_id"],
        "scenario_type": task_type,
        "overall_score": overall_score(evaluations, task_type),
        "evaluations": evaluations,
        "suggestions": result.get("suggestions", []),
    }


def print_summary(reports: list[dict]) -> None:
    if not reports:
        return
    by_criterion = defaultdict(list)
    for report in reports:
        for criterion, evaluation in report["evaluations"].items():
            if isinstance(evaluation.get("score"), (int, float)) and not isinstance(evaluation["score"], bool):
                by_criterion[criterion].append(evaluation["score"])
    mean = sum(r["overall_score"] for r in reports) / len(reports)
    logger.info(f"{len(reports)} tasks judged, mean overall score {mean:.2f}/10")
    for criterion, scores in sorted(by_criterion.items()):
        logger.info(f"  {criterion}: {sum(scores) / len(scores):.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", type=Path, required=True, help="Output of synthesis.summarize.")
    p.add_argument("--trajectories", type=Path, required=True, help="Trajectories the tasks came from.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task-type", choices=TASK_TYPES, default=None,
                   help="Judge prompt to use (default: infeasible if the task has an infeasible_reason, else general).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-base", default=None)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    trajectories = {t["trajectory_id"]: t for t in load_json(args.trajectories)}
    pairs = []
    for task in load_json(args.tasks):
        trajectory = trajectories.get(task.get("trajectory_id"))
        if trajectory is None:
            logger.warning(f"No trajectory for task {task.get('id')}")
            continue
        pairs.append((task, trajectory))

    reports = [] if args.no_resume or not args.output.exists() else load_json(args.output)
    done = {r["task_id"] for r in reports}
    todo = [(t, tr) for t, tr in pairs if (t.get("id") or t["trajectory_id"]) not in done]
    logger.info(f"{len(pairs)} task/trajectory pairs, {len(done)} already judged, {len(todo)} to go")

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(judge, task, traj, args.task_type or detect_task_type(task), args.model, args.api_base):
            task.get("id") or task["trajectory_id"]
            for task, traj in todo
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                report = future.result()
            except Exception as e:
                logger.error(f"{task_id}: {e}")
                continue
            with lock:
                reports.append(report)
                save_json(reports, args.output)
            logger.info(f"[{len(reports)}] {task_id}: {report['overall_score']:.2f}/10")

    print_summary(reports)


if __name__ == "__main__":
    main()
