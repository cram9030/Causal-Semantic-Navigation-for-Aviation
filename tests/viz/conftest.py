"""Fixtures for the visualization tests.

Matplotlib is forced onto the non-interactive Agg backend so the figure tests
run headless in CI, and the trajectory fixtures are reused from the Phase 1
trajectory test suite rather than redefined.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from tests.trajectory.conftest import (  # noqa: E402,F401
    conops,
    corridor,
    dogleg,
    due_east,
    due_north,
    trajectory_set,
    tube,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure after each test so a long run doesn't accumulate them."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
