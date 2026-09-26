"""Pool of users and example trajectories that the explorer samples from."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from synthesis.common import load_json, save_json
from tau2.registry import registry


class TrajectoryDB:
    """Users come from the domain database; examples are seeds plus generated runs.

    Every generated trajectory is added back to the example pool, so later
    explorations are conditioned on a growing, more diverse set of examples.
    """

    def __init__(self, domain_name: str, seed_files: Optional[list[str | Path]] = None):
        self.domain_name = domain_name
        self.users = self._load_users(domain_name)
        self.seed_trajectories: list[dict] = []
        for path in seed_files or []:
            self.seed_trajectories.extend(load_json(path))
        self.trajectories: list[dict] = []

    @staticmethod
    def _load_users(domain_name: str) -> list[dict[str, Any]]:
        db = registry.get_env_constructor(domain_name)().tools.db
        users = []
        for user_id, user in db.users.items():
            user = user.model_dump()
            user.setdefault("user_id", user_id)
            users.append(user)
        logger.info(f"Loaded {len(users)} users from the {domain_name} database")
        return users

    @property
    def examples(self) -> list[dict]:
        return self.seed_trajectories + self.trajectories

    def load(self, path: str | Path) -> None:
        """Resume: load previously generated trajectories."""
        self.trajectories = load_json(path)

    def save(self, path: str | Path) -> None:
        """Save generated trajectories (seed trajectories are not written)."""
        save_json(self.trajectories, path)

    def add(self, trajectory: dict) -> None:
        self.trajectories.append(trajectory)

    def sample_user(self) -> dict:
        return random.choice(self.users)

    def sample_example(self) -> Optional[dict]:
        return random.choice(self.examples) if self.examples else None

    def next_trajectory_id(self) -> str:
        """IDs continue after the highest ``traj_NNNN`` seen (seeds included)."""
        nums = [
            int(t["trajectory_id"].split("_")[1])
            for t in self.examples
            if str(t.get("trajectory_id", "")).startswith("traj_")
        ]
        return f"traj_{max(nums, default=-1) + 1:04d}"
