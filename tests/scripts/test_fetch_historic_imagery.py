"""Tests for the coverage-check / level-selection logic in the fetch script.

These exercise ``fetch_service`` (not just the underlying tile-math helpers)
against a synthetic service whose finest level has an astronomically large
tile grid but zero cached coverage - the same shape as the real
DPW_ImageryCached2025 failure this behavior was added to catch. The point
of these tests is to prove that rejecting that level stays cheap (bounded
sample requests, no full-grid materialization) rather than grinding through
millions of tiles as individual 404s.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest
import responses
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_historic_imagery as fhi  # noqa: E402
from csnav.data.arcgis.catalog import ArcGISCatalog  # noqa: E402
from csnav.data.arcgis.models import Extent, ServiceRef  # noqa: E402

BASE = "https://example.test/server/rest/services"
SERVICE_URL = f"{BASE}/Imagery/Fake/MapServer"
WMTS_URL = f"{SERVICE_URL}/WMTS/1.0.0/WMTSCapabilities.xml"

# AOI real-world-ish enough that level 20 below covers hundreds of millions
# of tiles - the exact scenario the coverage check exists to avoid grinding
# through tile-by-tile.
AOI = Extent(xmin=-122.1, ymin=37.2, xmax=-121.7, ymax=37.5, wkid=4326)


def _tile_info_json(*lods):
    return {
        "capabilities": "Map,Query,TilesOnly",
        "tileInfo": {
            "rows": 256, "cols": 256, "format": "PNG",
            "origin": {"x": -20037508.342787, "y": 20037508.342787},
            "spatialReference": {"wkid": 3857},
            "lods": [{"level": level, "resolution": resolution, "scale": 1.0} for level, resolution in lods],
        },
    }


# level 20: ~0.01 units/px -> hundreds of millions of tiles across AOI, none cached.
# level 3: ~5000 units/px -> a single tile across AOI, cached.
HUGE_REJECTED_LEVEL = 20
COVERED_LEVEL = 3
METADATA_HUGE_REJECT = _tile_info_json((COVERED_LEVEL, 5000.0), (HUGE_REJECTED_LEVEL, 0.01))

# level 8: ~100 units/px -> a handful of tiles across AOI, none cached - small
# enough to fully enumerate quickly, for the tests that deliberately bypass
# the coverage check and so *do* materialize the full grid for that level.
SMALL_REJECTED_LEVEL = 8
METADATA_SMALL_REJECT = _tile_info_json((COVERED_LEVEL, 5000.0), (SMALL_REJECTED_LEVEL, 100.0))


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def _mock_common(metadata: dict) -> None:
    responses.add(responses.GET, SERVICE_URL, json=metadata)
    responses.add(responses.GET, WMTS_URL, status=404)


def _calls_for_level(level: int) -> list:
    prefix = f"{SERVICE_URL}/tile/{level}/"
    return [c for c in responses.calls if c.request.url.startswith(prefix)]


def _ref() -> ServiceRef:
    return ServiceRef(folder="Imagery", name="Fake", service_type="MapServer")


@responses.activate
def test_auto_detect_skips_huge_uncovered_level_without_full_scan(tmp_path):
    _mock_common(METADATA_HUGE_REJECT)
    responses.add(responses.GET, re.compile(rf"{re.escape(SERVICE_URL)}/tile/{HUGE_REJECTED_LEVEL}/.*"), status=404)
    responses.add(
        responses.GET, re.compile(rf"{re.escape(SERVICE_URL)}/tile/{COVERED_LEVEL}/.*"),
        body=_png_bytes(), status=200,
    )

    catalog = ArcGISCatalog(base_url=BASE)
    written = fhi.fetch_service(_ref(), catalog, AOI, tmp_path, level=None, coverage_sample_size=10)

    assert written == 1
    tifs = list(tmp_path.rglob(f"{COVERED_LEVEL}_*.tif"))
    assert len(tifs) == 1

    # The rejected level's grid is far too large to enumerate - only the
    # bounded sample should ever have been requested.
    rejected_calls = _calls_for_level(HUGE_REJECTED_LEVEL)
    assert 0 < len(rejected_calls) <= 10


@responses.activate
def test_explicit_level_with_no_coverage_fails_fast(tmp_path):
    _mock_common(METADATA_HUGE_REJECT)
    responses.add(responses.GET, re.compile(rf"{re.escape(SERVICE_URL)}/tile/{HUGE_REJECTED_LEVEL}/.*"), status=404)

    catalog = ArcGISCatalog(base_url=BASE)
    written = fhi.fetch_service(
        _ref(), catalog, AOI, tmp_path, level=HUGE_REJECTED_LEVEL, coverage_sample_size=10,
    )

    assert written == 0
    assert list(tmp_path.rglob("*.tif")) == []
    rejected_calls = _calls_for_level(HUGE_REJECTED_LEVEL)
    assert 0 < len(rejected_calls) <= 10


@responses.activate
def test_skip_coverage_check_forces_explicit_level_through(tmp_path):
    _mock_common(METADATA_SMALL_REJECT)
    responses.add(responses.GET, re.compile(rf"{re.escape(SERVICE_URL)}/tile/{SMALL_REJECTED_LEVEL}/.*"), status=404)

    catalog = ArcGISCatalog(base_url=BASE)
    written = fhi.fetch_service(
        _ref(), catalog, AOI, tmp_path, level=SMALL_REJECTED_LEVEL,
        coverage_sample_size=10, skip_coverage_check=True,
    )

    # Every tile in the (small) grid was actually attempted, not short-circuited.
    assert written == 0
    total_grid_calls = _calls_for_level(SMALL_REJECTED_LEVEL)
    assert len(total_grid_calls) >= 4


@responses.activate
def test_skip_coverage_check_without_level_uses_finest(tmp_path):
    _mock_common(METADATA_SMALL_REJECT)
    responses.add(responses.GET, re.compile(rf"{re.escape(SERVICE_URL)}/tile/{SMALL_REJECTED_LEVEL}/.*"), status=404)

    catalog = ArcGISCatalog(base_url=BASE)
    written = fhi.fetch_service(
        _ref(), catalog, AOI, tmp_path, level=None,
        coverage_sample_size=10, skip_coverage_check=True,
    )

    assert written == 0
    # Landed on the finest level (8), not the coverage-having coarser one (3).
    assert _calls_for_level(SMALL_REJECTED_LEVEL)
    assert _calls_for_level(COVERED_LEVEL) == []


@responses.activate
def test_resume_skips_tiles_already_on_disk(tmp_path):
    # Pin to the covered level directly (rather than level=None) so this
    # test isn't entangled with the huge-level auto-detect scan - it's
    # testing resume behavior, not level selection.
    _mock_common(METADATA_HUGE_REJECT)
    responses.add(
        responses.GET, re.compile(rf"{re.escape(SERVICE_URL)}/tile/{COVERED_LEVEL}/.*"),
        body=_png_bytes(), status=200,
    )

    catalog = ArcGISCatalog(base_url=BASE)
    first = fhi.fetch_service(_ref(), catalog, AOI, tmp_path, level=COVERED_LEVEL, coverage_sample_size=10)
    assert first == 1

    responses.calls.reset()
    second = fhi.fetch_service(_ref(), catalog, AOI, tmp_path, level=COVERED_LEVEL, coverage_sample_size=10)

    assert second == 0  # nothing new written - already on disk
    # The covered level's grid is a single tile here, so the preflight
    # coverage probe (which runs regardless of what's already on disk) is
    # the only request - the main download loop skips it before ever
    # calling fetch_tile_auto again.
    assert len(_calls_for_level(COVERED_LEVEL)) == 1
