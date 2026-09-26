"""In-memory representation of an exploration trajectory and its on-disk schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def _now() -> str:
    return datetime.now().isoformat()


def _to_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


class TrajectoryStep:
    """One tool call made by the explorer and the environment's response."""

    def __init__(
        self,
        step_number: int,
        tool_call: Any,
        tool_response: Any,
        llm_reasoning: Optional[str] = None,
        available_tools: Optional[list[str]] = None,
        execution_status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        self.step_number = step_number
        self.tool_call = tool_call
        self.tool_response = tool_response
        self.llm_reasoning = llm_reasoning
        self.available_tools = list(available_tools or [])
        self.execution_status = execution_status
        self.error_message = error_message
        self.timestamp = _now()

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "tool_call": _to_dict(self.tool_call),
            "tool_response": _to_dict(self.tool_response),
            "llm_reasoning": self.llm_reasoning,
            "available_tools": self.available_tools,
            "execution_status": self.execution_status,
            "error_message": self.error_message,
        }


class ExplorationTrajectory:
    """A complete exploration run: the sampled user plus every step taken."""

    def __init__(self, domain_name: str, max_steps: int, user_info: Optional[dict]):
        self.domain_name = domain_name
        self.max_steps = max_steps
        self.user_info = user_info
        self.steps: list[TrajectoryStep] = []
        self.start_time = _now()
        self.end_time: Optional[str] = None
        self.termination_reason: Optional[str] = None
        self.total_cost = 0.0
        self.trajectory_id: Optional[str] = None

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def get_success_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.execution_status == "success" for s in self.steps) / len(self.steps)

    def finalize(self, reason: str, total_cost: float) -> None:
        self.termination_reason = reason
        self.end_time = _now()
        self.total_cost = float(total_cost)

    def to_dict(self) -> dict:
        tool_usage: dict[str, int] = {}
        for step in self.steps:
            tool_usage[step.tool_call.name] = tool_usage.get(step.tool_call.name, 0) + 1
        return {
            "trajectory_id": self.trajectory_id,
            "domain_name": self.domain_name,
            "exploration_strategy": "explore",
            "max_steps": self.max_steps,
            "steps": [s.to_dict() for s in self.steps],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "termination_reason": self.termination_reason,
            "total_cost": self.total_cost,
            "user_info": self.user_info,
            "statistics": {
                "total_steps": len(self.steps),
                "success_rate": self.get_success_rate(),
                "tool_usage": tool_usage,
            },
        }
