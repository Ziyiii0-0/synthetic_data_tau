"""Phase 4b: convert tau2 simulation results into chat-format SFT data.

Each kept simulation becomes one JSONL line ``{"messages": [...], "tools": [...]}``
in OpenAI chat format. Assistant turns keep the model's reasoning in
``reasoning_content`` (unless ``--no-thinking``). The system prompt is the
tau2 agent prompt with the domain policy; ``tools`` are the domain tool schemas.

By default only successful simulations (reward == 1) are kept; ``--success-basis db``
only requires the final database state to be correct.

Example:
    python -m synthesis.extract_sft --simulations outputs/airline/simulations_general.json \\
        --output outputs/airline/sft_general.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from synthesis.common import base_domain
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import Results, SimulationRun
from tau2.metrics.agent_metrics import is_successful
from tau2.registry import registry


def extract_thinking(message: AssistantMessage) -> Optional[str]:
    raw = message.raw_message
    if raw is None:
        return None
    get = raw.get if isinstance(raw, dict) else lambda key: getattr(raw, key, None)
    # Providers such as Bedrock return the same text in both fields; prefer the flat one.
    if get("reasoning_content"):
        return get("reasoning_content")
    blocks = []
    for block in get("thinking_blocks") or []:
        text = block.get("thinking") if isinstance(block, dict) else getattr(block, "thinking", None)
        if text:
            blocks.append(text)
    return "\n\n".join(blocks) or None


def is_success(simulation: SimulationRun, basis: str) -> bool:
    info = simulation.reward_info
    if info is None:
        return False
    if basis == "db":
        return info.db_check is not None and is_successful(info.db_check.db_reward)
    return is_successful(info.reward)


def to_sft_messages(simulation: SimulationRun, system_prompt: str, include_thinking: bool) -> list[dict]:
    messages: list[dict] = []
    for msg in simulation.messages:
        if isinstance(msg, SystemMessage):
            messages.append({"role": "system", "content": msg.content or ""})
        elif isinstance(msg, UserMessage):
            messages.append({"role": "user", "content": msg.content or ""})
        elif isinstance(msg, ToolMessage) and msg.requestor == "assistant":
            messages.append({"role": "tool", "content": msg.content or ""})
        elif isinstance(msg, AssistantMessage):
            out: dict = {"role": "assistant", "content": msg.content}
            if include_thinking and (thinking := extract_thinking(msg)):
                out["reasoning_content"] = thinking
            if msg.tool_calls:
                out["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(out)
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simulations", type=Path, nargs="+", required=True, help="tau2 results JSON file(s).")
    p.add_argument("--output", type=Path, required=True, help="SFT JSONL file to write.")
    p.add_argument("--keep", choices=["success", "failure", "all"], default="success")
    p.add_argument("--success-basis", choices=["reward", "db"], default="reward")
    p.add_argument("--no-thinking", action="store_true", help="Drop reasoning_content.")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w") as out:
        for path in args.simulations:
            results = Results.load(path)
            domain = base_domain(results.info.environment_info.domain_name)
            environment = registry.get_env_constructor(domain)()
            tools = [t.openai_schema for t in sorted(environment.get_tools(), key=lambda t: t.name)]
            system_prompt = SYSTEM_PROMPT.format(
                agent_instruction=AGENT_INSTRUCTION, domain_policy=environment.policy
            )
            kept = 0
            for simulation in results.simulations:
                success = is_success(simulation, args.success_basis)
                if (args.keep == "success" and not success) or (args.keep == "failure" and success):
                    continue
                messages = to_sft_messages(simulation, system_prompt, not args.no_thinking)
                out.write(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False) + "\n")
                kept += 1
            written += kept
            logger.info(f"{path}: kept {kept}/{len(results.simulations)} simulations")
    logger.info(f"Wrote {written} examples to {args.output}")


if __name__ == "__main__":
    main()
