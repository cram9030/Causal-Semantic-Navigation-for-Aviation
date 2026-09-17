"""Shared fixtures for ground-truth rasterization/label/check tests.

The tile is small and axis-aligned, at a known origin (matching
``tests/trajectory/conftest.py``'s downtown San Jose origin), so pixel counts
and buffer geometry are hand-checkable rather than only self-consistent.
"""

from __future__ import annotations

import pytest
from rasterio.transform import from_bounds

from csnav.data.arcgis.models import Extent
from csnav.data.arcgis.streets import StreetSegment
from csnav.trajectory.coverage import TileRef

ORIGIN_LAT, ORIGIN_LON = 37.3382, -121.8863

#: A ~180 m square tile (0.0016 deg at this latitude is ~178 m N-S), 64x64 px.
TILE_BOUNDS = Extent(
    xmin=ORIGIN_LON - 0.0008, ymin=ORIGIN_LAT - 0.0008, xmax=ORIGIN_LON + 0.0008, ymax=ORIGIN_LAT + 0.0008
)
TILE_WIDTH_PX = 64
TILE_HEIGHT_PX = 64


@pytest.fixture
def tile() -> TileRef:
    return TileRef(level=18, row=100, col=200, bounds=TILE_BOUNDS)


@pytest.fixture
def transform():
    return from_bounds(
        TILE_BOUNDS.xmin, TILE_BOUNDS.ymin, TILE_BOUNDS.xmax, TILE_BOUNDS.ymax, TILE_WIDTH_PX, TILE_HEIGHT_PX
    )


@pytest.fixture
def crossing_streets() -> list[StreetSegment]:
    """One east-west segment (published 40 ft width) crossing one north-south segment (no width) mid-tile."""
    return [
        StreetSegment(
            object_id=1,
            parts=(((TILE_BOUNDS.xmin, ORIGIN_LAT), (TILE_BOUNDS.xmax, ORIGIN_LAT)),),
            attributes={"WIDTH": 40.0, "STREETNAME": "First St"},
        ),
        StreetSegment(
            object_id=2,
            parts=(((ORIGIN_LON, TILE_BOUNDS.ymin), (ORIGIN_LON, TILE_BOUNDS.ymax)),),
            attributes={},
        ),
    ]
