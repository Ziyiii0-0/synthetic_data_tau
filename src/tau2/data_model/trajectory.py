"""
Data models for tracking agent exploration trajectories.

This module provides data structures to record complete exploration sessions,
including tool calls, responses, LLM reasoning, and execution metadata.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.utils.utils import get_now


ExecutionStatus = Literal["success", "error"]
TerminationReason = Literal["max_steps_reached", "error", "goal_achieved", "interrupted"]


class TrajectoryStep(BaseModel):
    """
    A single step in an exploration trajectory.
    """
    
    step_number: int = Field(description="Step number in the trajectory")
    timestamp: str = Field(description="Timestamp of the step", default_factory=get_now)
    tool_call: ToolCall = Field(description="The tool call made in this step")
    tool_response: ToolMessage = Field(description="The tool response received")
    llm_reasoning: Optional[str] = Field(
        description="LLM's reasoning/thinking process before making the tool call",
        default=None
    )
    available_tools: list[str] = Field(
        description="List of available tool names at this step",
        default_factory=list
    )
    execution_status: ExecutionStatus = Field(
        description="Whether the tool execution succeeded or failed"
    )
    error_message: Optional[str] = Field(
        description="Error message if execution failed",
        default=None
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "tool_call": {
                "id": self.tool_call.id,
                "name": self.tool_call.name,
                "arguments": self.tool_call.arguments,
                "requestor": self.tool_call.requestor,
            },
            "tool_response": {
                "role": self.tool_response.role,
                "content": self.tool_response.content,
                "id": self.tool_response.id,
                "requestor": self.tool_response.requestor,
                "error": self.tool_response.error,
            },
            "llm_reasoning": self.llm_reasoning,
            "available_tools": self.available_tools,
            "execution_status": self.execution_status,
            "error_message": self.error_message,
        }
    
    def __str__(self) -> str:
        """String representation of the step."""
        lines = [
            f"Step {self.step_number} [{self.execution_status.upper()}]",
            f"Time: {self.timestamp}",
            f"Tool: {self.tool_call.name}",
            f"Arguments: {json.dumps(self.tool_call.arguments, indent=2)}",
        ]
        if self.llm_reasoning:
            lines.append(f"Reasoning: {self.llm_reasoning}")
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        lines.append(f"Response: {self.tool_response.content}")
        return "\n".join(lines)


class ExplorationTrajectory(BaseModel):
    """
    Complete trajectory of an exploration session.
    """
    
    domain_name: str = Field(description="Name of the domain being explored")
    exploration_strategy: str = Field(description="Strategy used for exploration")
    max_steps: int = Field(description="Maximum number of steps configured")
    steps: list[TrajectoryStep] = Field(
        description="List of trajectory steps",
        default_factory=list
    )
    start_time: str = Field(
        description="Start time of the exploration",
        default_factory=get_now
    )
    end_time: Optional[str] = Field(
        description="End time of the exploration",
        default=None
    )
    termination_reason: Optional[TerminationReason] = Field(
        description="Reason for termination",
        default=None
    )
    total_cost: Optional[float] = Field(
        description="Total cost of LLM calls",
        default=0.0
    )
    goal_description: Optional[str] = Field(
        description="Goal description if using goal-oriented strategy",
        default=None
    )
    user_info: Optional[dict[str, Any]] = Field(
        description="User information provided to the agent",
        default=None
    )
    
    def add_step(self, step: TrajectoryStep):
        """Add a step to the trajectory."""
        self.steps.append(step)
    
    def finalize(
        self,
        termination_reason: TerminationReason,
        total_cost: Optional[float] = None
    ):
        """Finalize the trajectory with end time and termination reason."""
        self.end_time = get_now()
        self.termination_reason = termination_reason
        if total_cost is not None:
            self.total_cost = total_cost
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of tool executions."""
        if not self.steps:
            return 0.0
        successful = sum(1 for step in self.steps if step.execution_status == "success")
        return successful / len(self.steps)
    
    def get_tool_usage_stats(self) -> dict[str, int]:
        """Get statistics on tool usage."""
        stats = {}
        for step in self.steps:
            tool_name = step.tool_call.name
            stats[tool_name] = stats.get(tool_name, 0) + 1
        return stats
    
    def to_dict(self) -> dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "domain_name": self.domain_name,
            "exploration_strategy": self.exploration_strategy,
            "max_steps": self.max_steps,
            "steps": [step.to_dict() for step in self.steps],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "termination_reason": self.termination_reason,
            "total_cost": self.total_cost,
            "goal_description": self.goal_description,
            "user_info": self.user_info,
            "statistics": {
                "total_steps": len(self.steps),
                "success_rate": self.get_success_rate(),
                "tool_usage": self.get_tool_usage_stats(),
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert trajectory to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, filepath: str | Path):
        """Save trajectory to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str | Path) -> "ExplorationTrajectory":
        """Load trajectory from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Reconstruct the trajectory
        steps = []
        for step_data in data.get("steps", []):
            tool_call = ToolCall(**step_data["tool_call"])
            tool_response = ToolMessage(**step_data["tool_response"])
            step = TrajectoryStep(
                step_number=step_data["step_number"],
                timestamp=step_data["timestamp"],
                tool_call=tool_call,
                tool_response=tool_response,
                llm_reasoning=step_data.get("llm_reasoning"),
                available_tools=step_data.get("available_tools", []),
                execution_status=step_data["execution_status"],
                error_message=step_data.get("error_message"),
            )
            steps.append(step)
        
        return cls(
            domain_name=data["domain_name"],
            exploration_strategy=data["exploration_strategy"],
            max_steps=data["max_steps"],
            steps=steps,
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            termination_reason=data.get("termination_reason"),
            total_cost=data.get("total_cost", 0.0),
            goal_description=data.get("goal_description"),
            user_info=data.get("user_info"),
        )
    
    def __str__(self) -> str:
        """String representation of the trajectory."""
        lines = [
            f"Exploration Trajectory - {self.domain_name}",
            f"Strategy: {self.exploration_strategy}",
            f"Steps: {len(self.steps)}/{self.max_steps}",
            f"Success Rate: {self.get_success_rate():.1%}",
        ]
        if self.total_cost:
            lines.append(f"Total Cost: ${self.total_cost:.4f}")
        if self.termination_reason:
            lines.append(f"Termination: {self.termination_reason}")
        return "\n".join(lines)

