#!/usr/bin/env python3
"""Pull CSJ ``Streets`` centerlines for an AOI, in EPSG:4326.

Phase 0 data collection (see `docs/INTEGRATION_PLAN.md` §5): San Jose
publishes street centerlines - with width/lane attributes, refreshed weekly -
as one layer inside a shared ArcGIS Server service. Rather than hardcoding
which service currently hosts it (that has already changed once, per
`docs/phase0_csj_streets_lidar.md`), this script resolves the layer by name
via ``ArcGISCatalog.find_layer``, queries it (optionally restricted to a
bounding box), and writes the results as a GeoJSON ``FeatureCollection``.

This is a one-shot pull for inspecting/caching the dataset locally - it is
*not* the "possible roads" runtime lookup (which only ever reads the
precomputed, per-trajectory-window manifest built in Phase 1) or a live
per-frame query.

Example::

    python scripts/fetch_csj_streets.py \\
        --bbox -121.95 37.30 -121.85 37.36 \\
        --output data/raw/csj_streets/downtown.geojson
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.arcgis.catalog import ArcGISCatalog, DEFAULT_BASE_URL  # noqa: E402
from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.data.arcgis.streets import CSJStreetsClient  # noqa: E402

logger = logging.getLogger("fetch_csj_streets")

DEFAULT_SERVICE_NAME_CONTAINS = "OpenDataService"
DEFAULT_LAYER_NAME_CONTAINS = "Streets"


def resolve_layer_url(args: argparse.Namespace) -> str:
    if args.layer_url:
        return args.layer_url
    catalog = ArcGISCatalog(base_url=args.base_url)
    layer_url = catalog.find_layer(
        args.layer_name_contains,
        root=args.root,
        service_name_contains=args.service_name_contains,
    )
    logger.info("resolved Streets layer: %s", layer_url)
    return layer_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="ArcGIS REST services directory root")
    parser.add_argument(
        "--service-name-contains", default=DEFAULT_SERVICE_NAME_CONTAINS,
        help="substring used to find the service hosting the Streets layer",
    )
    parser.add_argument(
        "--layer-name-contains", default=DEFAULT_LAYER_NAME_CONTAINS,
        help="substring used to find the Streets layer within a matching service",
    )
    parser.add_argument(
        "--root", default="",
        help="catalog folder to search under (default: the whole services directory)",
    )
    parser.add_argument(
        "--layer-url", default=None,
        help="skip discovery and query this layer URL directly (e.g. .../MapServer/60)",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="restrict the query to this EPSG:4326 envelope (default: the whole layer)",
    )
    parser.add_argument("--where", default="1=1", help="ArcGIS SQL WHERE clause (default: all features)")
    parser.add_argument("--output", type=Path, required=True, help="output .geojson path")
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    layer_url = resolve_layer_url(args)
    client = CSJStreetsClient(layer_url, page_size=args.page_size)

    bbox = None
    if args.bbox:
        bbox = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)

    segments = client.query(bbox=bbox, where=args.where)
    logger.info("fetched %d street segment(s)", len(segments))

    feature_collection = {
        "type": "FeatureCollection",
        "features": [s.to_geojson_feature() for s in segments],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(feature_collection))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
