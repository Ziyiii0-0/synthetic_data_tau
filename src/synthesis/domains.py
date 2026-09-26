"""Register ``airline_explore`` and ``retail_explore`` in the tau2 registry.

An explore domain is the stock tau2 domain (same tools and policy) backed by the
larger LLM-generated database in ``synthesis/assets/<domain>/db_generated.json``.

Its task set is read from the JSON/JSONL file named by the ``SYNTH_TASKS_FILE``
environment variable (set by ``synthesis.simulate``); if it is unset the stock
tau2 tasks are returned.

Imported for its side effect by ``synthesis/__init__.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from synthesis.common import asset_path, load_json
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as airline_environment
from tau2.domains.airline.environment import get_tasks as airline_tasks
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as retail_environment
from tau2.domains.retail.environment import get_tasks as retail_tasks
from tau2.environment.environment import Environment
from tau2.registry import registry

TASKS_FILE_ENV = "SYNTH_TASKS_FILE"

_BASE_DOMAINS = {
    "airline": (FlightDB, airline_environment, airline_tasks),
    "retail": (RetailDB, retail_environment, retail_tasks),
}


def load_tasks_file(path: str | Path) -> list[Task]:
    """Load pipeline tasks (JSON list or JSONL); ``trajectory_id`` doubles as ``id``."""
    tasks = []
    for record in load_json(path):
        record = dict(record)
        record.setdefault("id", record.get("trajectory_id"))
        tasks.append(Task.model_validate(record))
    return tasks


def _make_environment_constructor(domain: str) -> Callable[..., Environment]:
    db_class, base_environment, _ = _BASE_DOMAINS[domain]

    def get_environment(db=None, solo_mode: bool = False) -> Environment:
        if db is None:
            db = db_class.load(asset_path(domain, "db_generated.json"))
        env = base_environment(db=db, solo_mode=solo_mode)
        return Environment(
            domain_name=f"{domain}_explore", policy=env.policy, tools=env.tools
        )

    return get_environment


def _make_tasks_loader(domain: str) -> Callable[[], list[Task]]:
    _, _, base_tasks = _BASE_DOMAINS[domain]

    def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
        path = os.environ.get(TASKS_FILE_ENV)
        if not path:
            return base_tasks()
        tasks = load_tasks_file(path)
        logger.info(f"{domain}_explore: loaded {len(tasks)} tasks from {path}")
        return tasks

    return get_tasks


def register_explore_domains() -> None:
    """Idempotently register every ``<domain>_explore`` variant."""
    for domain in _BASE_DOMAINS:
        name = f"{domain}_explore"
        if name not in registry.get_domains():
            registry.register_domain(_make_environment_constructor(domain), name)
        if name not in registry.get_task_sets():
            registry.register_tasks(_make_tasks_loader(domain), name)


register_explore_domains()
