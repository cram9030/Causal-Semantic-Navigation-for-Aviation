"""End-to-end run of the Phase 1 visualization script over the pilot scenario."""

import json
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_manifests as bm  # noqa: E402
import visualize_trajectories as vt  # noqa: E402

PILOT_SCENARIO = REPO_ROOT / "configs" / "scenarios" / "san_jose_downtown.yaml"


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_visualize_writes_a_figure_and_one_map_per_trajectory(tmp_path, monkeypatch):
    from csnav.trajectory.config import load_scenario

    _run(
        vt,
        ["--scenario", str(PILOT_SCENARIO), "--output-dir", str(tmp_path), "--no-tiles", "--dpi", "80"],
        monkeypatch,
    )

    scenario = load_scenario(PILOT_SCENARIO)
    assert (tmp_path / "trajectory_graph.png").stat().st_size > 5_000
    assert (tmp_path / "trajectory_set.html").exists()
    for trajectory in scenario.trajectory_set.trajectories:
        assert (tmp_path / f"trajectory_{trajectory.id}.html").exists()


def test_visualize_tube_radius_override_reaches_the_rendered_maps(tmp_path, monkeypatch):
    _run(
        vt,
        [
            "--scenario", str(PILOT_SCENARIO),
            "--output-dir", str(tmp_path),
            "--tube-radius", "500",
            "--no-tiles",
            "--no-imagery",
            "--dpi", "80",
        ],
        monkeypatch,
    )

    html = (tmp_path / "trajectory_t_p.html").read_text(encoding="utf-8")
    assert "tube corridor (500 m radius)" in html
    assert "geo.sanjoseca.gov" not in html  # --no-imagery


def test_build_manifests_pins_a_bundle_from_an_archived_streets_pull(tmp_path, monkeypatch):
    from csnav.trajectory.manifest import ManifestBundle

    streets_path = tmp_path / "streets.geojson"
    streets_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-121.8900, 37.3300], [-121.8402, 37.3648]],
                        },
                        "properties": {"OBJECTID": 1, "STREETNAME": "Test Diagonal", "WIDTH": 40},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle.json"

    _run(
        bm,
        [
            "--scenario", str(PILOT_SCENARIO),
            "--streets-geojson", str(streets_path),
            "--output", str(output),
            "--tube-radius", "300",
            "--no-tiles",
        ],
        monkeypatch,
    )

    bundle = ManifestBundle.load(output)
    assert bundle.trajectory_set_id == "san_jose_downtown"
    assert bundle.parameters["tube_radius_m"] == 300.0
    assert bundle.streets_source == str(streets_path)
    assert all(manifest.tube_radius == 300.0 for manifest in bundle.manifests)
    assert any(manifest.candidate_roads for manifest in bundle.manifests)


def test_build_manifests_requires_a_scenario(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_manifests"])
    with pytest.raises(SystemExit):
        bm.main()
