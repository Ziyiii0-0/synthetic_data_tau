"""Synthetic task and SFT data generation for the tau2 airline and retail domains.

Importing the package registers the ``airline_explore`` / ``retail_explore``
domains (stock tools + generated databases) with the tau2 registry.
"""

from synthesis import domains  # noqa: F401  (registers explore domains)
