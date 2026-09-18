"""Tests for scripts/discover_imagery_levels.py.

Covers the two things that actually matter for this script:

1. :func:`discover_levels` records each service's *own* finest-covered
   level independently - the exact "2011 only has level 19, everything else
   has 21" scenario this script exists for, pinning each vintage to its own
   ceiling rather than a shared floor.
2. :func:`update_params_levels` rewrites only the ``imagery.levels`` block
   in a params.yaml-shaped file, leaves every comment untouched, replaces
   (not merges) the block's entries, and refuses to guess (leaving the file
   unchanged) if it can't find exactly one ``  levels:`` line to update.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import responses
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import discover_imagery_levels as dil  # noqa: E402
from csnav.data.arcgis.catalog import ArcGISCatalog  # noqa: E402
from csnav.data.arcgis.models import Extent  # noqa: E402

BASE = "https://example.test/server/rest/services"
AOI = Extent(xmin=-122.1, ymin=37.2, xmax=-121.7, ymax=37.5, wkid=4326)

# level 3: ~5000 units/px -> a single tile across AOI.
# level 5: ~1000 units/px -> a handful of tiles across AOI.
COARSE_LEVEL = 3
FINE_LEVEL = 5


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


METADATA_BOTH_LEVELS = _tile_info_json((COARSE_LEVEL, 5000.0), (FINE_LEVEL, 1000.0))


def _mock_service(name: str, has_fine_coverage: bool) -> None:
    service_url = f"{BASE}/Imagery/{name}/MapServer"
    responses.add(responses.GET, service_url, json=METADATA_BOTH_LEVELS)
    responses.add(responses.GET, f"{service_url}/WMTS/1.0.0/WMTSCapabilities.xml", status=404)
    responses.add(
        responses.GET, re.compile(rf"{re.escape(service_url)}/tile/{COARSE_LEVEL}/.*"), status=200, body=b"x",
    )
    fine_status = 200 if has_fine_coverage else 404
    responses.add(
        responses.GET, re.compile(rf"{re.escape(service_url)}/tile/{FINE_LEVEL}/.*"), status=fine_status, body=b"x",
    )


@responses.activate
def test_discover_levels_pins_each_service_to_its_own_ceiling():
    # One recent vintage has fine coverage; one older vintage only has the
    # coarser level - mirrors DPW_ImageryCached2025 (21) vs
    # DPW_ImageryCached2011 (19) in the real catalog. Each should keep its
    # OWN highest level, not be dragged down to match the other.
    responses.add(responses.GET, f"{BASE}/", json={"folders": ["Imagery"], "services": []})
    responses.add(
        responses.GET, f"{BASE}/Imagery", json={
            "folders": [],
            "services": [
                {"name": "Imagery/DPW_Imagery2025", "type": "MapServer"},
                {"name": "Imagery/DPW_Imagery2011", "type": "MapServer"},
            ],
        },
    )
    _mock_service("DPW_Imagery2025", has_fine_coverage=True)
    _mock_service("DPW_Imagery2011", has_fine_coverage=False)

    catalog = ArcGISCatalog(base_url=BASE)
    results = dil.discover_levels(catalog, AOI, name_contains="DPW_Imagery", coverage_sample_size=10)

    # Keyed by the short service name (ref.name), matching the output
    # subfolder / --levels-file lookup key fetch_historic_imagery.py uses.
    assert results["DPW_Imagery2025"] == FINE_LEVEL
    assert results["DPW_Imagery2011"] == COARSE_LEVEL


PARAMS_TEXT = """\
imagery:
  output_dir: data/raw/dpw_imagery
  # A hand-written comment explaining the pin - must survive the rewrite
  # unchanged, word for word, including this second line.
  levels: {}

streets:
  output: data/raw/csj_streets/aoi.geojson
"""


def test_update_params_levels_only_changes_the_levels_block(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(PARAMS_TEXT)

    dil.update_params_levels(params_file, {"DPW_Imagery2025": 21, "DPW_Imagery2011": 19})

    new_text = params_file.read_text()
    assert "    DPW_Imagery2011: 19" in new_text
    assert "    DPW_Imagery2025: 21" in new_text
    # Every comment line is untouched.
    assert "# A hand-written comment explaining the pin - must survive the rewrite" in new_text
    assert "# unchanged, word for word, including this second line." in new_text
    # Nothing outside the levels block changed.
    for line in ["imagery:", "  output_dir: data/raw/dpw_imagery", "streets:", "  output: data/raw/csj_streets/aoi.geojson"]:
        assert line in new_text

    assert yaml.safe_load(new_text)["imagery"]["levels"] == {"DPW_Imagery2025": 21, "DPW_Imagery2011": 19}


def test_update_params_levels_replaces_rather_than_merges(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(PARAMS_TEXT)

    dil.update_params_levels(params_file, {"DPW_Imagery2025": 21, "DPW_Imagery2011": 19})
    # A vintage disappears from the catalog on the next refresh.
    dil.update_params_levels(params_file, {"DPW_Imagery2025": 21})

    reloaded = yaml.safe_load(params_file.read_text())["imagery"]["levels"]
    assert reloaded == {"DPW_Imagery2025": 21}


def test_update_params_levels_refuses_to_guess_with_no_levels_line(tmp_path):
    params_file = tmp_path / "params.yaml"
    original = "imagery:\n  output_dir: data/raw/dpw_imagery\n"
    params_file.write_text(original)

    with pytest.raises(SystemExit):
        dil.update_params_levels(params_file, {"DPW_Imagery2025": 21})

    assert params_file.read_text() == original


def test_update_params_levels_refuses_to_guess_with_multiple_levels_lines(tmp_path):
    params_file = tmp_path / "params.yaml"
    original = "imagery:\n  levels: {}\nother:\n  levels: {}\n"
    params_file.write_text(original)

    with pytest.raises(SystemExit):
        dil.update_params_levels(params_file, {"DPW_Imagery2025": 21})

    assert params_file.read_text() == original
