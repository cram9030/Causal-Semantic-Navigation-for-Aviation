#!/usr/bin/env python3
"""Build and pin the per-window landmark manifests for a trajectory set.

This is the offline precompute of `docs/INTEGRATION_PLAN.md` §3.3, run once per
flight-planning cycle: for every window of every candidate route in ``T``, grow
the RNP tube by the sensor's ground reach, query CSJ Streets against that
envelope, and record the candidate roads, their intersections, and the imagery
tiles the window covers. The result is written as one pinned JSON bundle that
the runtime "possible roads" node looks up - it never re-queries CSJ Streets.

Every transition rule is covered the same way, over the *family* of paths it
admits (`csnav.trajectory.transition`) - not just the candidate routes. A
transition may begin at any arc length along its source, so the aircraft can
legitimately be anywhere the family sweeps while a hand-off is under way; the
manifest has to say what could be seen from there too, or the "possible roads"
lookup goes empty for every slice flown during a transition. Pass
`--no-transitions` to build candidate-route manifests only.

Street geometry comes either from the live layer (resolved by name via
``ArcGISCatalog.find_layer``) or, with ``--streets-geojson``, from an archived
pull written by ``scripts/fetch_csj_streets.py``. Prefer the archive when
rebuilding a manifest that must match an earlier cycle: the live layer refreshes
weekly and will not reproduce it.

Example (live)::

    python scripts/build_manifests.py \\
        --scenario configs/scenarios/san_jose_downtown.yaml \\
        --output data/manifests/san_jose_downtown_r250.json

Example (from an archived streets pull, plus a review map)::

    python scripts/build_manifests.py \\
        --scenario configs/scenarios/san_jose_downtown.yaml \\
        --streets-geojson data/raw/csj_streets/downtown.geojson \\
        --output data/manifests/san_jose_downtown_r250.json \\
        --map out/viz/manifests.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.arcgis.catalog import ArcGISCatalog, DEFAULT_BASE_URL  # noqa: E402
from csnav.data.arcgis.streets import CSJStreetsClient, segments_from_geojson  # noqa: E402
from csnav.data.arcgis.tiles import web_mercator_tile_info  # noqa: E402
from csnav.data.lidar import LidarElevationClient  # noqa: E402
from csnav.trajectory.config import load_scenario  # noqa: E402
from csnav.trajectory.coverage import agl_from_elevation, height_as_agl  # noqa: E402
from csnav.trajectory.manifest_builder import ManifestBuilder, StaticStreetsSource  # noqa: E402
from csnav.trajectory.trajectory import X0_NODE  # noqa: E402

logger = logging.getLogger("build_manifests")

DEFAULT_ROOT = "OPN"
DEFAULT_SERVICE_NAME_CONTAINS = "OpenDataService"
DEFAULT_LAYER_NAME_CONTAINS = "Streets"


def build_streets_source(args: argparse.Namespace):
    """Resolve the street geometry source: archived GeoJSON, an explicit layer URL, or discovery."""
    if args.streets_geojson:
        data = json.loads(Path(args.streets_geojson).read_text(encoding="utf-8"))
        segments = segments_from_geojson(data)
        logger.info("loaded %d street segments from %s", len(segments), args.streets_geojson)
        return StaticStreetsSource(segments, source_label=str(args.streets_geojson))

    layer_url = args.layer_url
    if not layer_url:
        catalog = ArcGISCatalog(base_url=args.base_url)
        layer_url = catalog.find_layer(
            DEFAULT_LAYER_NAME_CONTAINS, root=DEFAULT_ROOT, service_name_contains=DEFAULT_SERVICE_NAME_CONTAINS
        )
        logger.info("resolved Streets layer: %s", layer_url)
    return CSJStreetsClient(layer_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, type=Path, help="scenario YAML (configs/scenarios/*.yaml)")
    parser.add_argument("--output", required=True, type=Path, help="path to write the pinned manifest bundle JSON")
    parser.add_argument("--streets-geojson", type=Path, default=None, help="archived CSJ Streets pull to build from")
    parser.add_argument("--layer-url", default=None, help="skip discovery and query this Streets layer URL")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="ArcGIS REST services directory root")
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=None,
        help="override the scenario's tube radius, in meters - re-run with different values to sweep it",
    )
    parser.add_argument(
        "--tile-level",
        type=int,
        default=None,
        help="imagery cache level to record tiles at; defaults to the scenario's conops.tile_level",
    )
    parser.add_argument("--no-tiles", action="store_true", help="don't record imagery tiles in the manifests")
    parser.add_argument(
        "--elevation",
        action="store_true",
        help="derive AGL from USGS 3DEP ground elevation instead of treating waypoint height as AGL",
    )
    parser.add_argument(
        "--per-window-query",
        action="store_true",
        help="query CSJ Streets per window instead of once per trajectory (slower, tighter bounding boxes)",
    )
    parser.add_argument(
        "--no-transitions",
        action="store_true",
        help="only build manifests for candidate routes, skipping every transition family",
    )
    parser.add_argument("--map", type=Path, default=None, help="also write a folium review map of the built bundle")
    parser.add_argument(
        "--map-landmarks",
        action="store_true",
        help="include each window's roads and intersections in --map (off by default; a lot of geometry)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable DEBUG-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    scenario = load_scenario(args.scenario)
    if args.tube_radius is not None:
        scenario = scenario.with_tube_radius(args.tube_radius, label=f"sweep_{args.tube_radius:g}m")
        logger.info("tube radius overridden to %.1f m", args.tube_radius)

    conops = scenario.conops
    tile_level = args.tile_level if args.tile_level is not None else conops.tile_level
    use_tiles = not args.no_tiles and tile_level is not None

    agl_provider = height_as_agl
    if args.elevation:
        client = LidarElevationClient()
        agl_provider = agl_from_elevation(lambda lon, lat: client.identify(lon, lat))
        logger.info("AGL will be derived from USGS 3DEP ground elevation")

    builder = ManifestBuilder(
        streets=build_streets_source(args),
        tile_info=web_mercator_tile_info() if use_tiles else None,
        tile_level=tile_level if use_tiles else None,
        agl_provider=agl_provider,
    )

    bundle = builder.build_set(
        scenario.trajectory_set,
        conops,
        per_window_query=args.per_window_query,
        include_transitions=not args.no_transitions,
    )
    destination = bundle.save(args.output)
    transition_windows = sum(
        len(bundle.for_transition(rule.source, rule.target))
        for rule in scenario.trajectory_set.transitions
        if rule.source != X0_NODE
    )
    logger.info(
        "wrote %s: %d windows (%d over transition families), %d roads, %d intersections, %d distinct tiles",
        destination,
        len(bundle.manifests),
        transition_windows,
        sum(len(manifest.candidate_roads) for manifest in bundle.manifests),
        sum(len(manifest.intersections) for manifest in bundle.manifests),
        len(bundle.all_tiles()),
    )

    if args.map:
        from csnav.viz.map_view import bundle_map, save_map

        logger.info(
            "wrote %s",
            save_map(
                bundle_map(
                    scenario.trajectory_set,
                    bundle,
                    show_landmarks=args.map_landmarks,
                    transition_model=conops.transition,
                ),
                args.map,
            ),
        )


if __name__ == "__main__":
    main()
