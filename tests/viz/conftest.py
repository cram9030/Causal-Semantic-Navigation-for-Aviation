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

#: Reused from the ground-truth test suite for the same reason - a rendered
#: label used in a viz test should stay in sync with what rasterize() tests
#: check, not drift into its own fixture.
from tests.data.ground_truth.conftest import crossing_streets, tile, transform  # noqa: F401,E402
