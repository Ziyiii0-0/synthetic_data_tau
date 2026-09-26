"""Sample a coherent subset of tools from a domain's API dependency graph.

Restricting each exploration to a different tool subset diversifies the
trajectories. The graph (``synthesis/assets/<domain>/api_graph.json``) lists
tools as nodes, "output of A feeds argument of B" edges, and common workflows.
"""

from __future__ import annotations

import random
from typing import Optional

from synthesis.common import load_api_graph


class APIGraphSampler:
    def __init__(self, domain: str):
        graph = load_api_graph(domain)
        self.nodes = {node["api_name"]: node for node in graph["nodes"]}
        self.workflow_patterns = graph.get("workflow_patterns", [])

        has_successor = {edge["from"] for edge in graph["edges"]}
        has_predecessor = {edge["to"] for edge in graph["edges"]}
        self.isolated_tools = [
            t for t in self.nodes if t not in has_successor and t not in has_predecessor
        ]
        self.entry_point_tools = [
            t for t in self.nodes if t in has_successor and t not in has_predecessor
        ]
        self.intermediate_tools = [
            t for t in self.nodes if t in has_successor and t in has_predecessor
        ]

    def sample_coherent_subset(
        self, target_size: int, auth_tools: Optional[list[str]] = None
    ) -> list[str]:
        """Sample ``target_size`` tools that form meaningful workflows.

        1. one authentication tool and one entry-point tool
        2. every tool mentioned in two random workflow patterns
        3. each isolated tool (e.g. ``calculate``) with probability 0.5
        4. fill up, preferring entry-point then intermediate tools
        """
        sampled: set[str] = set()
        if auth_tools:
            sampled.add(random.choice(auth_tools))
        sampled.add(random.choice(self.entry_point_tools))

        num_workflows = min(2, len(self.workflow_patterns))
        for workflow in random.sample(self.workflow_patterns, num_workflows):
            steps_text = " ".join(workflow.get("steps", []))
            for tool in self.nodes:
                if len(sampled) >= target_size:
                    break
                if tool in steps_text:
                    sampled.add(tool)

        for tool in self.isolated_tools:
            if len(sampled) >= target_size:
                break
            if random.random() < 0.5:
                sampled.add(tool)

        remaining = [t for t in self.nodes if t not in sampled]
        while len(sampled) < target_size and remaining:
            candidates = (
                [t for t in remaining if t in self.entry_point_tools]
                or [t for t in remaining if t in self.intermediate_tools]
                or remaining
            )
            tool = random.choice(candidates)
            sampled.add(tool)
            remaining.remove(tool)
        return list(sampled)
