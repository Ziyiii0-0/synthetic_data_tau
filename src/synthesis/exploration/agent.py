"""The explorer: an LLM that calls domain tools one at a time, without a user.

The explorer sees the domain policy, a sampled user profile and an example
trajectory, invents a plausible customer scenario and executes it against the
real tau2 environment. Every tool call and response is recorded.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Optional

from loguru import logger

from synthesis.common import AUTH_TOOLS, base_domain, get_write_actions
from synthesis.exploration.trajectory import ExplorationTrajectory, TrajectoryStep
from synthesis.prompts.exploration import build_exploration_prompt
from tau2.data_model.message import Message, SystemMessage, UserMessage
from tau2.environment.environment import Environment
from tau2.registry import registry
from tau2.utils.llm_utils import generate


class ExplorerAgent:
    def __init__(
        self,
        environment: Environment,
        llm: str,
        llm_args: Optional[dict] = None,
        max_steps: int = 10,
        user_info: Optional[dict] = None,
        example_trajectory: Optional[dict] = None,
        available_tools: Optional[list[str]] = None,
    ):
        self.environment = environment
        self.domain = base_domain(environment.get_domain_name())
        self.llm = llm
        self.llm_args = deepcopy(llm_args or {})
        self.max_steps = max_steps
        self.user_info = user_info
        self.available_tools = available_tools
        self.write_actions = get_write_actions(self.domain, include_generic=True)
        self.system_prompt = self._build_system_prompt(example_trajectory)

    def _build_system_prompt(self, example_trajectory: Optional[dict]) -> str:
        parts = [
            build_exploration_prompt(self.domain, AUTH_TOOLS[self.domain]),
            f"# Step Budget\nYou can call tools for at most {self.max_steps} total steps.",
            "# Domain Policy\nBelow is the domain policy. Use it to understand what "
            f"actions are valid and what workflows are expected:\n\n{self.environment.policy}",
        ]
        if self.user_info:
            parts.append(
                f"# User Information\n{json.dumps(self.user_info, indent=2)}\n\n"
                "Use this user information when calling tools that require user context."
            )
        if example_trajectory:
            example = [
                {
                    "tool_name": step["tool_call"]["name"],
                    "arguments": step["tool_call"].get("arguments", {}),
                }
                for step in example_trajectory["steps"]
                if "tool_call" in step
            ]
            parts.append(
                "# Example Reference\nBelow is an example of how another agent explored "
                "similar tools. Use this as inspiration, but don't copy it exactly:\n\n"
                + json.dumps(example, indent=2)
            )
        return "\n\n".join(parts)

    def run(self) -> ExplorationTrajectory:
        trajectory = ExplorationTrajectory(
            domain_name=self.environment.get_domain_name(),
            max_steps=self.max_steps,
            user_info=self.user_info,
        )
        tools = self.environment.get_tools()
        if self.available_tools is not None:
            tools = [t for t in tools if t.name in self.available_tools]
        tool_names = [t.name for t in tools]

        history: list[Message] = []
        total_cost = 0.0
        called_write = False
        for step_num in range(1, self.max_steps + 1):
            prompt = (
                f"Step {step_num}/{self.max_steps}: Choose a tool to call and provide "
                f"the necessary arguments. Available tools: {', '.join(tool_names)}"
            )
            # Nudge towards a state-changing action if none was made yet.
            steps_left = self.max_steps - step_num
            available_writes = [t for t in tool_names if t in self.write_actions]
            if steps_left <= 1 and not called_write and available_writes:
                prompt += (
                    f"\n\nIMPORTANT: You have {steps_left + 1} step(s) left and have NOT "
                    "yet called any state-changing action. You should now call a write "
                    "action to make a meaningful change. Available write actions: "
                    + ", ".join(available_writes)
                )
            history.append(UserMessage(role="user", content=prompt))

            try:
                message = generate(
                    model=self.llm,
                    tools=tools,
                    messages=[SystemMessage(role="system", content=self.system_prompt)]
                    + history,
                    **self.llm_args,
                )
            except Exception as e:
                logger.error(f"Step {step_num}: LLM call failed: {e}")
                trajectory.finalize("error", total_cost)
                return trajectory
            total_cost += message.cost or 0.0
            history.append(message)
            if not message.is_tool_call():
                logger.warning(f"Step {step_num}: no tool call; content={message.content!r}")
                continue

            for tool_call in message.tool_calls:
                response = self.environment.get_response(tool_call)
                history.append(response)
                trajectory.add_step(
                    TrajectoryStep(
                        step_number=step_num,
                        tool_call=tool_call,
                        tool_response=response,
                        llm_reasoning=message.content,
                        available_tools=tool_names,
                        execution_status="error" if response.error else "success",
                        error_message=response.content if response.error else None,
                    )
                )
                if not response.error and tool_call.name in self.write_actions:
                    called_write = True

        trajectory.finalize("max_steps_reached", total_cost)
        return trajectory


def run_exploration(domain_name: str, llm: str, **kwargs) -> ExplorationTrajectory:
    """Explore ``domain_name`` (e.g. ``airline_explore``) in a fresh environment."""
    environment = registry.get_env_constructor(domain_name)()
    return ExplorerAgent(environment=environment, llm=llm, **kwargs).run()
