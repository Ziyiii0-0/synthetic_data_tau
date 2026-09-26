"""Phase 1: generate exploratory tool-call trajectories.

For each trajectory we sample a user from the domain database, a subset of
tools from the API graph, an example trajectory (seed or previously generated)
and a step budget, then let the explorer agent act in the environment.

The output file is rewritten after every trajectory; re-running with the same
``--output`` resumes until ``--num-trajectories`` exist.

Example:
    python -m synthesis.explore --domain airline --num-trajectories 200 \\
        --output outputs/airline/trajectories.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from loguru import logger

from synthesis.common import AUTH_TOOLS, DEFAULT_MODEL, DOMAINS, asset_path
from synthesis.exploration.agent import run_exploration
from synthesis.exploration.tool_sampler import APIGraphSampler
from synthesis.exploration.trajectory_db import TrajectoryDB


def sample_step_budget(max_steps: int) -> int:
    """~N(0.6 * max, max / 5), clipped to [2, max_steps]."""
    if max_steps <= 2:
        return max_steps
    steps = int(random.gauss(max_steps * 0.6, max(max_steps / 5, 1)))
    return max(2, min(steps, max_steps))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=DOMAINS, required=True)
    p.add_argument("--db", choices=["generated", "original"], default="generated",
                   help="Explore the generated database (<domain>_explore) or the stock tau2 one.")
    p.add_argument("--output", type=Path, required=True, help="Trajectory JSON file (resumed if it exists).")
    p.add_argument("--num-trajectories", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=12, help="Upper bound of the sampled step budget.")
    p.add_argument("--tool-subset-size", type=int, default=12,
                   help="Tools offered per trajectory; 0 offers every tool.")
    p.add_argument("--seed-file", type=Path, default=None,
                   help="Seed trajectories (default: synthesis/assets/<domain>/seed_trajectories.json).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--llm-args", type=json.loads, default={"temperature": 0.0},
                   help='JSON dict passed to the LLM call, e.g. \'{"temperature": 0.7}\'.')
    p.add_argument("--seed", type=int, default=15, help="Random seed for all sampling.")
    args = p.parse_args()

    random.seed(args.seed)
    domain_name = f"{args.domain}_explore" if args.db == "generated" else args.domain
    seed_file = args.seed_file or asset_path(args.domain, "seed_trajectories.json")
    db = TrajectoryDB(domain_name, seed_files=[seed_file])
    if args.output.exists():
        db.load(args.output)
        logger.info(f"Resuming: {len(db.trajectories)} trajectories already in {args.output}")
    sampler = APIGraphSampler(args.domain) if args.tool_subset_size > 0 else None

    consecutive_failures = 0
    while len(db.trajectories) < args.num_trajectories:
        tools = (
            sampler.sample_coherent_subset(args.tool_subset_size, AUTH_TOOLS[args.domain])
            if sampler
            else None
        )
        trajectory = run_exploration(
            domain_name,
            llm=args.model,
            llm_args=args.llm_args,
            max_steps=sample_step_budget(args.max_steps),
            user_info=db.sample_user(),
            example_trajectory=db.sample_example(),
            available_tools=tools,
        )
        if not trajectory.steps:
            consecutive_failures += 1
            logger.warning(f"Discarding empty trajectory ({trajectory.termination_reason})")
            if consecutive_failures >= 5:
                raise RuntimeError("5 empty trajectories in a row; check the model / API key.")
            continue
        consecutive_failures = 0
        trajectory.trajectory_id = db.next_trajectory_id()
        db.add(trajectory.to_dict())
        db.save(args.output)
        logger.info(
            f"[{len(db.trajectories)}/{args.num_trajectories}] {trajectory.trajectory_id}: "
            f"{len(trajectory.steps)} steps, success rate {trajectory.get_success_rate():.0%}"
        )
    logger.info(f"Done: {len(db.trajectories)} trajectories in {args.output}")


if __name__ == "__main__":
    main()
