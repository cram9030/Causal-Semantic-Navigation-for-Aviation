#!/usr/bin/env python3
"""Pull CSJ ``Street Intersections`` locations for an AOI, in EPSG:4326.

A separate layer from `scripts/fetch_csj_streets.py`'s centerlines - San Jose
publishes named intersection *locations* (point features: name, leg count,
traffic control type) as their own dataset, confirmed at
``https://geo.sanjoseca.gov/server/rest/services/OPN/OPN_OpenDataService/MapServer/276``
("Street Intersections", ``geometryType: esriGeometryPoint``). See
`csnav.data.arcgis.intersections`'s module docstring for the full confirmed
schema and why this is metadata enrichment for a derived intersection, not a
replacement for it - this layer has no polygon geometry to rasterize.

This is a one-shot pull for caching the dataset locally, mirroring
``fetch_csj_streets.py`` - it is *not* a live per-frame query. Ground truth
for a given imagery vintage should read an archived pull
(``--street-intersections-geojson`` on ``scripts/build_ground_truth.py``),
not the live layer.

``--where`` defaults to ``INTTYPE='Intersection'``, excluding this layer's
other reference-point types (``End``, ``Muni``, ``Non-Intersection``,
``Ramp``, ``Range Exception``) - see the module docstring above for why.

Example::

    uv run python scripts/fetch_csj_intersections.py \\
        --bbox -121.95 37.30 -121.85 37.36 \\
        --output data/raw/csj_intersections/downtown.geojson

Example (re-check the schema after a catalog reorganization, before trusting
``--where``/field names again)::

    uv run python scripts/fetch_csj_intersections.py --list-fields
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.arcgis.intersections import (  # noqa: E402
    DEFAULT_LAYER_URL,
    DEFAULT_WHERE,
    CSJIntersectionsClient,
)
from csnav.data.arcgis.models import Extent  # noqa: E402

logger = logging.getLogger("fetch_csj_intersections")


def print_fields(metadata: dict) -> None:
    """Print every field this layer has - name, type, alias, and (for a coded-value domain
    field) every valid stored code with its display label. Mirrors
    ``fetch_csj_streets.py``'s ``print_fields`` - see there for the rationale."""
    fields = metadata.get("fields") or []
    if not fields:
        print("(no field metadata returned by this layer)")
        return
    for field in fields:
        name = field.get("name")
        line = f"{name} ({field.get('type')})"
        alias = field.get("alias")
        if alias and alias != name:
            line += f" alias={alias!r}"
        print(line)
        domain = field.get("domain") or {}
        if domain.get("type") == "codedValue":
            for coded_value in domain.get("codedValues", []):
                print(f"    stored value {coded_value.get('code')!r} -> {coded_value.get('name')!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--layer-url", default=DEFAULT_LAYER_URL,
        help=f"query this layer URL (default: the confirmed layer, {DEFAULT_LAYER_URL!r})",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="restrict the query to this EPSG:4326 envelope (default: the whole layer)",
    )
    parser.add_argument(
        "--where", default=DEFAULT_WHERE,
        help=f"ArcGIS SQL WHERE clause (default: {DEFAULT_WHERE!r} - see the module docstring; "
        "pass '1=1' for every row, or use --list-fields to find a different filter)",
    )
    parser.add_argument("--output", type=Path, required=False, help="output .geojson path (not needed with --list-fields)")
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument(
        "--list-fields", action="store_true",
        help="print this layer's field names/types/coded-value domains and exit, without "
        "querying or writing anything",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    client = CSJIntersectionsClient(args.layer_url, page_size=args.page_size)

    if args.list_fields:
        print_fields(client.get_metadata())
        return

    if args.output is None:
        raise SystemExit("--output is required (unless --list-fields)")

    bbox = None
    if args.bbox:
        bbox = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)

    intersections = client.query(bbox=bbox, where=args.where)
    logger.info("fetched %d street intersection(s)", len(intersections))

    feature_collection = {
        "type": "FeatureCollection",
        "features": [i.to_geojson_feature() for i in intersections],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(feature_collection))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
