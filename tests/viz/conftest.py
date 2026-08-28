"""Fixtures for the visualization tests.

The trajectory fixtures are reused from the Phase 1 trajectory test suite
rather than redefined, so the two suites cannot drift apart.
"""

from __future__ import annotations

from tests.trajectory.conftest import (  # noqa: F401
    camera,
    conops,
    dogleg,
    due_east,
    model,
    orthogonal,
    parallel_north,
    returning,
    trajectory_set,
    tube,
)
