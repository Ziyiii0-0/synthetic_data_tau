"""Offline tests for the synthesis pipeline (no LLM calls)."""

import json
from pathlib import Path

import pytest

import synthesis  # noqa: F401  (registers explore domains)
from synthesis.common import AUTH_TOOLS, TASK_TYPES, load_api_graph
from synthesis.exploration.tool_sampler import APIGraphSampler
from synthesis.filter_tasks import normalize, passes
from synthesis.summarize import build_prompt, build_task, clean_steps, select_trajectories
from synthesis.validate import build_judge_prompt, overall_score
from tau2.registry import registry

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DOMAINS = ["airline", "retail"]


def load(domain, name):
    return json.loads((EXAMPLES / domain / name).read_text())


@pytest.mark.parametrize("domain", DOMAINS)
def test_explore_domain_uses_generated_db(domain):
    stock = registry.get_env_constructor(domain)().tools.db
    explore = registry.get_env_constructor(f"{domain}_explore")()
    assert explore.get_domain_name() == f"{domain}_explore"
    assert len(explore.tools.db.users) != len(stock.users)


@pytest.mark.parametrize("domain", DOMAINS)
def test_tool_sampler_returns_valid_subset(domain):
    tools = {n["api_name"] for n in load_api_graph(domain)["nodes"]}
    subset = APIGraphSampler(domain).sample_coherent_subset(8, AUTH_TOOLS[domain])
    assert len(subset) == 8
    assert set(subset) <= tools
    assert set(subset) & set(AUTH_TOOLS[domain])


def step(name):
    return {"tool_call": {"name": name, "arguments": {}}}


def test_clean_steps_cuts_at_transfer():
    trajectory = {"steps": [step("find_user_id_by_email"), step("get_order_details"),
                            step("find_user_id_by_name_zip"), step("transfer_to_human_agents"),
                            step("cancel_pending_order")]}
    names = [s["tool_call"]["name"] for s in clean_steps(trajectory)["steps"]]
    assert names == ["find_user_id_by_email", "get_order_details"]


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_summarization_prompt_and_task(domain, task_type):
    trajectories = select_trajectories(load(domain, "1_trajectories.json"), domain, "general")
    assert trajectories
    prompt = build_prompt(trajectories[0], domain, task_type)
    assert "Agent Actions Taken:" in prompt and "Return ONLY the JSON object" in prompt
    scenario = {"reason_for_call": "r", "known_info": "k", "unknown_info": None,
                "actions_should_not_taken": [{"tool_name": "calculate", "arguments": {}}]}
    task = build_task(trajectories[0], scenario, domain, write_actions=set())
    assert task["id"] == trajectories[0]["trajectory_id"]
    assert "actions_should_not_taken" not in task["user_scenario"]["instructions"]
    assert task["evaluation_criteria"]["actions"]


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_judge_prompt_formats(domain, task_type):
    task = load(domain, "2_tasks_general.json")[0]
    trajectory = next(t for t in load(domain, "1_trajectories.json")
                      if t["trajectory_id"] == task["trajectory_id"])
    prompt = build_judge_prompt(task, trajectory, task_type)
    assert task["user_scenario"]["instructions"]["reason_for_call"] in prompt
    assert ("# DOMAIN TIME" in prompt) == (domain == "airline" and task_type != "infeasible")


def test_overall_score():
    evals = {"realism": {"score": 9}, "necessity": {"score": 6}, "correctness": {"score": 9}}
    assert overall_score(evals, "general") == 8.0
    assert overall_score({**evals, "changing_scenario_relevance": {"score": False}}, "changing") == 0.0
    assert overall_score({**evals, "infeasibility_reasonability": {"score": 4}}, "infeasible") == 7.0


def test_filter_normalizes_tasks():
    report = {"overall_score": 8, "evaluations": {"necessity": {"score": 6}}}
    assert passes(report, 7, None, None) and not passes(report, 7, 7, None)

    task = {"trajectory_id": "traj_0001", "evaluation_criteria": {"communicate_info": ["-10.5"]}}
    assert normalize(task, "general")["evaluation_criteria"]["communicate_info"] == ["10.5"]

    infeasible = {
        "trajectory_id": "traj_0002",
        "user_scenario": {"instructions": {"domain": "retail", "actions_should_be_taken": [
            {"tool_name": "get_order_details", "arguments": {}},
            {"tool_name": "modify_user_address", "arguments": {}},
        ]}},
        "evaluation_criteria": {"actions": [], "communicate_info": []},
    }
    out = normalize(infeasible, "infeasible")
    assert [a["tool_name"] for a in out["user_scenario"]["instructions"]["actions_should_be_taken"]] == [
        "modify_user_address", "transfer_to_human_agents"]
    assert "actions" not in out["evaluation_criteria"]
