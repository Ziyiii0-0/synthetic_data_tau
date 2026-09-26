#!/usr/bin/env bash
# End-to-end data generation for one domain:
#   explore -> summarize -> validate -> filter   (for each task type)
#   [optional] simulate -> extract SFT
#
# Usage:
#   scripts/run_pipeline.sh <airline|retail> [output_dir]
#
# Environment overrides:
#   SYNTH_MODEL        LiteLLM model for every LLM call (see src/synthesis/common.py)
#   NUM_TRAJECTORIES   trajectories to explore            (default 200)
#   MAX_STEPS          explorer step budget upper bound   (default 12)
#   TASK_TYPES         space-separated task types         (default "general changing infeasible")
#   THRESHOLD          minimum validity score to keep     (default 7.0)
#   MAX_WORKERS        parallel LLM calls for phases 2-3  (default 4)
#   RUN_SFT=1          also run simulations and extract SFT data
#   NUM_TRIALS         simulation trials per task          (default 1)
set -euo pipefail

DOMAIN="${1:?usage: $0 <airline|retail> [output_dir]}"
OUT="${2:-outputs/${DOMAIN}}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-200}"
MAX_STEPS="${MAX_STEPS:-12}"
TASK_TYPES="${TASK_TYPES:-general changing infeasible}"
THRESHOLD="${THRESHOLD:-7.0}"
MAX_WORKERS="${MAX_WORKERS:-4}"
NUM_TRIALS="${NUM_TRIALS:-1}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT"
TRAJ="$OUT/trajectories.json"

echo "== Phase 1: exploration ($NUM_TRAJECTORIES trajectories) -> $TRAJ"
"$PYTHON" -m synthesis.explore --domain "$DOMAIN" --output "$TRAJ" \
    --num-trajectories "$NUM_TRAJECTORIES" --max-steps "$MAX_STEPS"

for TYPE in $TASK_TYPES; do
    echo "== Phase 2: summarize '$TYPE' tasks"
    "$PYTHON" -m synthesis.summarize --trajectories "$TRAJ" --task-type "$TYPE" \
        --output "$OUT/tasks_${TYPE}.json" --max-workers "$MAX_WORKERS"

    echo "== Phase 3: validity check '$TYPE' tasks"
    "$PYTHON" -m synthesis.validate --tasks "$OUT/tasks_${TYPE}.json" --trajectories "$TRAJ" \
        --task-type "$TYPE" --output "$OUT/validity_${TYPE}.json" --max-workers "$MAX_WORKERS"
    "$PYTHON" -m synthesis.filter_tasks --tasks "$OUT/tasks_${TYPE}.json" \
        --report "$OUT/validity_${TYPE}.json" --output "$OUT/tasks_${TYPE}_filtered.json" \
        --threshold "$THRESHOLD"

    if [[ "${RUN_SFT:-0}" == "1" ]]; then
        echo "== Phase 4: simulate + extract SFT for '$TYPE' tasks"
        "$PYTHON" -m synthesis.simulate --domain "$DOMAIN" --tasks "$OUT/tasks_${TYPE}_filtered.json" \
            --output "$OUT/simulations_${TYPE}.json" --num-trials "$NUM_TRIALS"
        "$PYTHON" -m synthesis.extract_sft --simulations "$OUT/simulations_${TYPE}.json" \
            --output "$OUT/sft_${TYPE}.jsonl"
    fi
done

echo "== Done. Outputs in $OUT"
