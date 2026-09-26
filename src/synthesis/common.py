"""Paths, domain helpers and LLM helpers shared by every pipeline stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from tau2.utils.utils import DATA_DIR

SYNTHESIS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SYNTHESIS_ROOT.parents[1]
ASSETS_DIR = SYNTHESIS_ROOT / "assets"
TAU2_DOMAINS_DIR = DATA_DIR / "tau2" / "domains"
OUTPUT_DIR = REPO_ROOT / "outputs"

DOMAINS = ("airline", "retail")
TASK_TYPES = ("general", "changing", "infeasible")

# Any LiteLLM model string works, e.g. "openai/gpt-4.1" or
# "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0".
DEFAULT_MODEL = os.environ.get("SYNTH_MODEL", "anthropic/claude-sonnet-4-5-20250929")

# Tools the explorer (and the sampled tool subset) should start with.
AUTH_TOOLS = {
    "airline": ["get_user_details"],
    "retail": ["find_user_id_by_email", "find_user_id_by_name_zip"],
}


def base_domain(domain: str) -> str:
    """Map ``airline_explore`` -> ``airline`` and validate the domain name."""
    base = domain.removesuffix("_explore")
    if base not in DOMAINS:
        raise ValueError(f"Unsupported domain {domain!r}; expected one of {DOMAINS}")
    return base


def asset_path(domain: str, name: str) -> Path:
    return ASSETS_DIR / base_domain(domain) / name


def load_policy(domain: str) -> str:
    return (TAU2_DOMAINS_DIR / base_domain(domain) / "policy.md").read_text()


def load_api_graph(domain: str) -> dict:
    return load_json(asset_path(domain, "api_graph.json"))


def get_write_actions(domain: str, include_generic: bool = False) -> set[str]:
    """Names of state-changing tools (``calculate``/``transfer`` count as GENERIC)."""
    tool_types = {"WRITE", "GENERIC"} if include_generic else {"WRITE"}
    return {
        node["api_name"]
        for node in load_api_graph(domain)["nodes"]
        if node.get("tool_type") in tool_types
    }


def load_json(path: str | Path) -> Any:
    """Load a JSON file, or a JSONL file as a list of objects."""
    path = Path(path)
    with path.open() as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def llm_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
) -> dict:
    """Send a single-turn prompt and parse the reply as a JSON object.

    Tolerates ``<think>...</think>`` prefixes and markdown code fences.
    """
    from litellm import completion

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        num_retries=3,
    )
    if api_base is not None:
        kwargs["api_base"] = api_base
    content = completion(**kwargs).choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"{model} returned empty content")

    content = content.strip()
    if "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM reply is not valid JSON ({e}): {content[:500]!r}") from e
