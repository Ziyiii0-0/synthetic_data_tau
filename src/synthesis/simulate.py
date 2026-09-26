"""Phase 4a: run agent + user-simulator conversations on the filtered tasks.

This is a thin wrapper around ``tau2.run.run_domain`` that points the
``<domain>_explore`` task set at ``--tasks`` and runs it against the generated
(default) or stock database. Successful conversations are
turned into SFT data by ``synthesis.extract_sft``.

Example:
    python -m synthesis.simulate --domain airline \\
        --tasks outputs/airline/tasks_general_filtered.json \\
        --output outputs/airline/simulations_general.json --num-trials 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from synthesis.common import DEFAULT_MODEL, DOMAINS
from synthesis.domains import TASKS_FILE_ENV
from tau2.data_model.simulation import RunConfig
from tau2.run import run_domain


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=DOMAINS, required=True)
    p.add_argument("--db", choices=["generated", "original"], default="generated",
                   help="Must match the database the tasks were explored on.")
    p.add_argument("--tasks", type=Path, required=True, help="Output of synthesis.filter_tasks.")
    p.add_argument("--output", type=Path, required=True, help="tau2 results JSON to write.")
    p.add_argument("--agent-llm", default=DEFAULT_MODEL)
    p.add_argument("--user-llm", default=DEFAULT_MODEL)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--no-thinking", action="store_true",
                   help="Disable extended thinking for Claude/Qwen models (on by default).")
    args = p.parse_args()

    os.environ[TASKS_FILE_ENV] = str(args.tasks.resolve())
    llm_args = {"temperature": 1.0, "llm_enable_thinking": not args.no_thinking}
    config = RunConfig(
        domain=f"{args.domain}_explore" if args.db == "generated" else args.domain,
        task_set_name=f"{args.domain}_explore",
        agent="llm_agent",
        llm_agent=args.agent_llm,
        llm_args_agent=dict(llm_args),
        user="user_simulator",
        llm_user=args.user_llm,
        llm_args_user=dict(llm_args),
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        num_tasks=args.num_tasks,
        # run_domain saves to DATA_DIR / "simulations" / f"{save_to}.json"; an
        # absolute path replaces that prefix.
        save_to=str(args.output.resolve().with_suffix("")),
    )
    run_domain(config)


if __name__ == "__main__":
    main()
