"""Phase 3b: keep tasks whose validity score clears the threshold(s).

Kept tasks are also normalized for tau2 evaluation:
  * general / changing: minus signs are stripped from ``communicate_info``
    (agents state "a refund of 10.2", not "-10.2");
  * infeasible: ``actions_should_be_taken`` is reduced to WRITE actions plus
    ``transfer_to_human_agents``, and the gold ``actions`` / ``communicate_info``
    are dropped, since the agent is graded on what it must (not) do.

Example:
    python -m synthesis.filter_tasks --tasks outputs/airline/tasks_general.json \\
        --report outputs/airline/validity_general.json \\
        --output outputs/airline/tasks_general_filtered.json --threshold 7
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Optional

from loguru import logger

from synthesis.common import get_write_actions, load_json, save_json


def passes(report: dict, threshold: float, min_necessity: Optional[float], min_infeasibility: Optional[float]) -> bool:
    if report.get("overall_score", 0) < threshold:
        return False
    evaluations = report.get("evaluations", {})
    if min_necessity is not None and evaluations.get("necessity", {}).get("score", 0) < min_necessity:
        return False
    if min_infeasibility is not None:
        score = evaluations.get("infeasibility_reasonability", {}).get("score", 0)
        if score < min_infeasibility:
            return False
    return True


def normalize(task: dict, task_type: str) -> dict:
    task = deepcopy(task)
    task.setdefault("id", task["trajectory_id"])
    criteria = task.get("evaluation_criteria") or {}
    if task_type == "infeasible":
        instructions = task["user_scenario"]["instructions"]
        if "actions_should_be_taken" in instructions:
            writes = get_write_actions(instructions["domain"])
            required = [a for a in instructions["actions_should_be_taken"] if a.get("tool_name") in writes]
            instructions["actions_should_be_taken"] = required + [
                {"tool_name": "transfer_to_human_agents", "arguments": {}}
            ]
        criteria.pop("actions", None)
        criteria.pop("communicate_info", None)
    else:
        criteria["communicate_info"] = [
            info.replace("-", "") if isinstance(info, str) else info
            for info in criteria.get("communicate_info") or []
        ]
    return task


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True, help="Output of synthesis.validate.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=7.0, help="Minimum overall_score.")
    p.add_argument("--min-necessity", type=float, default=None)
    p.add_argument("--min-infeasibility", type=float, default=None,
                   help="Minimum infeasibility_reasonability score (infeasible tasks).")
    args = p.parse_args()

    reports = {r["task_id"]: r for r in load_json(args.report)}
    tasks = load_json(args.tasks)
    kept = []
    for task in tasks:
        report = reports.get(task.get("id") or task["trajectory_id"])
        if report and passes(report, args.threshold, args.min_necessity, args.min_infeasibility):
            kept.append(normalize(task, report["scenario_type"]))
    save_json(kept, args.output)
    logger.info(f"Kept {len(kept)}/{len(tasks)} tasks ({len(reports)} judged) -> {args.output}")


if __name__ == "__main__":
    main()
