# Synthetic τ²: exploration-based task and SFT data generation for τ²-bench

This repository generates **databases, tasks and SFT data** for the
[τ²-bench](https://github.com/sierra-research/tau2-bench) **airline** and **retail**
customer-service domains.

Instead of writing tasks by hand, an LLM *explorer* acts inside the real tool
environment and records a trajectory. A second LLM then turns each trajectory
into a user-side task description, and an LLM *judge* keeps only valid tasks.
The kept tasks can be run through τ² (agent + user simulator) to collect
successful conversations as SFT data.

```
             ┌──────────────────────┐
 phase 0     │ generated database   │  larger airline / retail DBs  (shipped in src/synthesis/assets)
             └──────────┬───────────┘
                        ▼
 phase 1     explore      ── LLM explorer calls tools on a sampled user   → trajectories.json
                        ▼
 phase 2     summarize    ── trajectory → user scenario + gold actions   → tasks_<type>.json
                        ▼
 phase 3     validate     ── LLM judge scores realism/necessity/...      → validity_<type>.json
             filter_tasks ── keep tasks with score ≥ threshold           → tasks_<type>_filtered.json
                        ▼
 phase 4     simulate     ── τ² agent × user simulator on kept tasks     → simulations_<type>.json
 (optional)  extract_sft  ── successful conversations → chat SFT JSONL   → sft_<type>.jsonl
```

Three task types are generated from the same trajectories:

| type         | the user…                                                           | graded on                                  |
|--------------|---------------------------------------------------------------------|--------------------------------------------|
| `general`    | has one uniquely actionable intent that reproduces the trajectory   | final DB state + communicated numbers      |
| `changing`   | changes their mind or handles several issues in sequence            | final DB state + communicated numbers      |
| `infeasible` | asks for something policy or tools forbid (may include a feasible part) | forbidden actions not taken, required actions taken |

## Repository layout

```
src/synthesis/            the data generation pipeline
  explore.py              phase 1  – exploratory trajectories
  summarize.py            phase 2  – trajectories → tasks
  validate.py             phase 3  – LLM-judge validity check
  filter_tasks.py         phase 3b – threshold + normalize tasks
  simulate.py             phase 4a – run τ² conversations on tasks
  extract_sft.py          phase 4b – conversations → SFT JSONL
  extract_seeds.py        phase 0  – seed trajectories from stock τ² tasks
  generate_db/            phase 0  – airline.py / retail.py database generators
  prompts/                exploration.py, summarization.py, validity.py
  exploration/            explorer agent, tool sampler, trajectory model
  domains.py              registers airline_explore / retail_explore in τ²
  common.py               paths, default model, LLM JSON helper
  assets/<domain>/        db_generated.json, seed_trajectories.json, api_graph.json
src/tau2/                 τ²-bench framework (vendored; airline, retail, mock domains)
data/tau2/domains/        stock τ² policies, databases and tasks
examples/<domain>/        a few records from every stage of a real run
scripts/run_pipeline.sh   end-to-end driver
tests/                    τ² tests + offline pipeline tests (tests/test_synthesis.py)
```

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/Ziyiii0-0/synthetic_data_tau.git && cd synthetic_data_tau
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs both the `synthesis` pipeline and the `tau2` CLI.

### Models and API keys

All LLM calls go through [LiteLLM](https://docs.litellm.ai/docs/providers), so
any provider works. Copy `.env.example` to `.env` and fill in the keys you need.

The default model for every stage is `anthropic/claude-sonnet-4-5-20250929`
(the model the shipped data was generated with). Override it globally with
`SYNTH_MODEL`, or per stage with `--model` (`--agent-llm` / `--user-llm` for
`simulate`):

```bash
export SYNTH_MODEL=openai/gpt-4.1                                   # OpenAI
export SYNTH_MODEL=bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0  # AWS Bedrock
python -m synthesis.summarize ... --model hosted_vllm/Qwen/Qwen3-32B --api-base http://localhost:8000/v1
```

Claude and Qwen models run with extended thinking enabled by τ²'s `generate()`
(see `src/tau2/utils/llm_utils.py`).

## Quick start

```bash
# Generate 200 airline trajectories and all three task types, with validity filtering
NUM_TRAJECTORIES=200 scripts/run_pipeline.sh airline outputs/airline

# Also run conversations on the kept tasks and extract SFT data
RUN_SFT=1 NUM_TRIALS=2 scripts/run_pipeline.sh retail outputs/retail
```

Every stage writes its output after each item and **resumes** if re-run with
the same output path, so interrupted runs can simply be restarted.

## The pipeline, stage by stage

### Phase 0 – databases and seeds (pre-built, optional to regenerate)

The pipeline explores **generated databases** that are larger than the stock
τ² ones, so trajectories (and tasks) don't collide with the benchmark:

| domain  | stock τ² DB                            | generated DB (`assets/<domain>/db_generated.json`) |
|---------|----------------------------------------|----------------------------------------------------|
| airline | 300 flights, 500 users, 2000 reservations | 400 flights, 600 users, 2500 reservations         |
| retail  | 50 products, 500 users, 1000 orders    | 190 products, 490 users, 2000 orders               |

They are registered in τ² as `airline_explore` / `retail_explore` (stock tools
and policy, generated DB). Pass `--db original` to `explore`/`simulate` to use
the stock databases instead.

To regenerate:

```bash
python -m synthesis.generate_db.airline --output outputs/airline/db_generated.json   # --no-llm for offline names
python -m synthesis.generate_db.retail --phase 1   # LLM proposes product names → assets/retail/product_names.json
python -m synthesis.generate_db.retail --phase 2 --output outputs/retail/db_generated.json
cp outputs/airline/db_generated.json src/synthesis/assets/airline/db_generated.json   # to use it
```

Seed trajectories (the gold actions of 20 stock τ² tasks) are shown to the
explorer as examples:

```bash
python -m synthesis.extract_seeds --domain airline --output src/synthesis/assets/airline/seed_trajectories.json
```

### Phase 1 – exploration

```bash
python -m synthesis.explore --domain airline --num-trajectories 200 --max-steps 12 \
    --output outputs/airline/trajectories.json
```

For every trajectory the explorer gets:

* a random **user** from the database (full profile in the system prompt);
* a random **tool subset** (`--tool-subset-size`, default 12) sampled from the
  domain's API dependency graph (`assets/<domain>/api_graph.json`) so that it
  covers an authentication tool, whole workflows and utilities;
* a random **example trajectory** (seeds plus everything generated so far);
* a **step budget** drawn from N(0.6·max, max/5), clipped to [2, `--max-steps`].

It then invents a plausible customer scenario and executes it one tool call at
a time, citing the policy rule before every write (`prompts/exploration.py`).
Near the end of the budget it is nudged to make a state-changing call.
Trajectory IDs continue after the seeds (`traj_0020`, `traj_0021`, …).

### Phase 2 – summarize trajectories into tasks

```bash
python -m synthesis.summarize --trajectories outputs/airline/trajectories.json \
    --task-type general --output outputs/airline/tasks_general.json
```

Trajectories without a state-changing call are skipped; `infeasible` tasks
additionally need a `transfer_to_human_agents` call. Steps from the first
transfer on are dropped. The LLM (`prompts/summarization.py`) writes the user
scenario (`reason_for_call`, `known_info`, `unknown_info`, `task_instructions`,
plus `infeasible_reason` / `actions_should_not_taken` / `actions_should_be_taken`
for infeasible tasks), and the trajectory's successful tool calls become the
gold `evaluation_criteria.actions`. The result is a standard τ² task.

### Phase 3 – validity check and filtering

```bash
python -m synthesis.validate --tasks outputs/airline/tasks_general.json \
    --trajectories outputs/airline/trajectories.json --task-type general \
    --output outputs/airline/validity_general.json

python -m synthesis.filter_tasks --tasks outputs/airline/tasks_general.json \
    --report outputs/airline/validity_general.json --threshold 7 \
    --output outputs/airline/tasks_general_filtered.json
```

The judge (`prompts/validity.py`) scores each task against its trajectory, 1–10:

* **realism** – would a real user ask this, with enough detail to be unambiguous?
* **necessity** – are all required writes / communicated numbers motivated, and nothing more?
* **correctness** – does the trajectory actually solve the stated request?
* **infeasibility_reasonability** (infeasible only) – are the forbidden actions and reason right?
* **changing_scenario_relevance** (changing only) – a pass/fail gate; failing sets the score to 0.

`overall_score` is the mean of the scored criteria. `filter_tasks` keeps tasks
at or above `--threshold` (optionally `--min-necessity`, `--min-infeasibility`)
and normalizes them for τ² evaluation (see its docstring).

### Phase 4 – conversations and SFT data

```bash
python -m synthesis.simulate --domain airline --tasks outputs/airline/tasks_general_filtered.json \
    --output outputs/airline/simulations_general.json --num-trials 2

python -m synthesis.extract_sft --simulations outputs/airline/simulations_general.json \
    --output outputs/airline/sft_general.jsonl
```

`simulate` runs the τ² LLM agent against the τ² user simulator on the kept
tasks and scores them with the τ² evaluator. `extract_sft` keeps successful
conversations (reward = 1, or `--success-basis db` for DB state only) and writes
one `{"messages": [...], "tools": [...]}` line per conversation in OpenAI chat
format; assistant reasoning is kept in `reasoning_content`.

The results file is a normal τ² results file, so `tau2 view` works on it.

## Data formats

`examples/<domain>/` holds a few records from a real run of each stage:

| file                               | stage       | content                                                    |
|------------------------------------|-------------|------------------------------------------------------------|
| `1_trajectories.json`              | explore     | `user_info`, `steps[]` (`tool_call`, `tool_response`, `llm_reasoning`, `execution_status`) |
| `2_tasks_general.json`             | summarize   | τ² tasks with `trajectory_id`, `user_scenario`, `evaluation_criteria` |
| `3_validity_report_general.json`   | validate    | per-criterion `score` / `explanation` / `issues`, `overall_score` |
| `4_tasks_general_filtered.json`    | filter      | tasks with `overall_score ≥ 7`                              |
| `5_sft.jsonl`                      | extract_sft | chat messages + tool schemas                               |

## Evaluating models

Generated tasks are regular τ² tasks, so `synthesis.simulate` doubles as an
evaluator for any agent model on them (`--agent-llm`, `--num-trials`); the τ²
metrics are printed at the end of the run. The original benchmark is still
available through the τ² CLI:

```bash
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 1
```

## Tests

```bash
pytest tests/test_synthesis.py          # offline pipeline tests
pytest tests                            # full τ² suite (some tests call an LLM)
```

## Acknowledgements and license

`src/tau2` and `data/tau2` are adapted from
[τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT, © Sierra
Research); the vendored copy adds infeasible-task evaluation (forbidden and
required actions) and extended-thinking support. Only the airline, retail and mock
domains are kept. If you use this code, please also cite τ²-bench:

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment},
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982},
}
```
