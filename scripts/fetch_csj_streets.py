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

**"Streets" is an ambiguous name in CSJ's catalog** - more than one layer's
name contains it (at least a full-attribute streets layer with width/lane
fields, and a separate, sparser "Street Centerlines" reference/geocoding
layer with no width field at all), and which one ``ArcGISCatalog.find_layer``
resolves to by substring match has already changed between sessions with no
code change on this side - CSJ's catalog reorganizing is exactly the kind
of drift `docs/phase0_csj_streets_lidar.md` already flags as a known risk.
The layer also isn't only street centerlines even within one match - it can
mix in other feature classes (ramps, alleys, driveways) whose geometry sits
close enough to a real street to occlude it when later rasterized (see
`docs/phase2_ground_truth_rasterization.md`'s "Overlapping/occluding
segments" section - this is what a real ground-truth build's silently-wrong
OBJECTIDs turned out to trace back to).

Because of that ambiguity, **there is no default ``--where`` filter beyond
"every feature"** - picking a wrong field/value silently would just
reintroduce the same class of bug with different symptoms. Three flags exist
specifically to nail this down without guessing:

* ``--list-layers`` - print every layer whose name matches
  ``--layer-name-contains`` (not just the first, unlike plain discovery),
  so you can see every candidate and pin the right one with ``--layer-url``.
* ``--list-fields`` - print the resolved layer's fields (name, type, alias,
  and every stored code + display label for a coded-value domain field).
* ``--distinct-values FIELD`` - print every value a plain string field
  actually contains (e.g. ``DESIGNATION``/``DESCRIPTION``), for a field with
  no coded-value domain to read off from metadata alone.

Example (find the right layer, then field, then filter, after a query
error or a suspiciously sparse pull)::

    uv run python scripts/fetch_csj_streets.py --list-layers
    uv run python scripts/fetch_csj_streets.py --layer-url <the right one> --list-fields
    uv run python scripts/fetch_csj_streets.py --layer-url <the right one> --distinct-values DESIGNATION

``--historic-moment`` requests the network as it stood at a past edit moment
instead of today's - useful for pairing ground-truth labels
(``scripts/build_ground_truth.py``) with a historic imagery vintage rather
than the current network, since roads change over time. This only returns
something different from the current pull if CSJ Streets has ArcGIS
*archiving* enabled server-side, which is **not confirmed** for this layer -
see ``docs/phase2_ground_truth_rasterization.md``. Check
``CSJStreetsClient.get_metadata()``'s ``archivingInfo`` field first; if
archiving isn't enabled, this flag has no effect and the current network is
returned regardless of the moment requested.

Example::

    uv run python scripts/fetch_csj_streets.py \\
        --bbox -121.95 37.30 -121.85 37.36 \\
        --output data/raw/csj_streets/downtown.geojson

Example (a historic moment, if the layer supports it)::

    uv run python scripts/fetch_csj_streets.py \\
        --bbox -121.95 37.30 -121.85 37.36 \\
        --historic-moment 2019-01-01T00:00:00Z \\
        --output data/raw/csj_streets/downtown_2019.geojson
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

DEFAULT_ROOT = "OPN"
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


def print_fields(metadata: dict) -> None:
    """Print every field this layer has - name, type, alias, and (for a coded-value domain
    field) every valid stored code with its display label.

    The direct fix for a ``--where`` clause that fails with an ArcGIS query
    error: the field name may not exist on this layer, or the field may be a
    coded-value domain where the SQL comparison must use the *stored* code
    (often a short integer/string), not the human-readable label a person
    would use to describe it (e.g. matching a UI dropdown's display text
    like "Street Centerline" against the field fails if the layer actually
    stores a code like ``1`` or ``"SC"`` for that value).
    """
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
        "--root", default=DEFAULT_ROOT,
        help=(
            f"catalog folder to search under (default: {DEFAULT_ROOT!r}, where the Streets "
            "layer lives at the time of writing; pass '' to search the whole services "
            "directory if that's changed)"
        ),
    )
    parser.add_argument(
        "--layer-url", default=None,
        help="skip discovery and query this layer URL directly (e.g. .../MapServer/60)",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="restrict the query to this EPSG:4326 envelope (default: the whole layer)",
    )
    parser.add_argument(
        "--where", default="1=1",
        help="ArcGIS SQL WHERE clause (default: every feature - see the module docstring for why "
        "there is no more specific default here; use --list-fields/--distinct-values to find one)",
    )
    parser.add_argument(
        "--historic-moment", default=None,
        help=(
            "ISO 8601 timestamp (or epoch milliseconds) to read the layer as of, via ArcGIS's "
            "historicMoment parameter - only has an effect if this layer has archiving enabled "
            "(unconfirmed; see the module docstring). Omit for the current network."
        ),
    )
    parser.add_argument(
        "--output", type=Path, required=False,
        help="output .geojson path (not needed with --list-fields)",
    )
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument(
        "--list-fields", action="store_true",
        help="print this layer's field names/types/coded-value domains and exit, without querying "
        "or writing anything - use this to find the right --where field/value after a query error",
    )
    parser.add_argument(
        "--list-layers", action="store_true",
        help="print every layer whose name matches --layer-name-contains (not just the first one "
        "discovery would pick) and exit - use this when a query error or a suspiciously sparse "
        "field list suggests --layer-url/discovery landed on the wrong 'Streets'-named layer",
    )
    parser.add_argument(
        "--distinct-values", default=None, metavar="FIELD",
        help="print every distinct value FIELD actually contains on the resolved layer and exit, "
        "without writing anything - for a plain string field with no coded-value domain to read "
        "off from --list-fields alone",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.list_layers:
        catalog = ArcGISCatalog(base_url=args.base_url)
        matches = catalog.find_layers(
            args.layer_name_contains, root=args.root, service_name_contains=args.service_name_contains
        )
        if not matches:
            print(f"(no layer matching {args.layer_name_contains!r} found)")
        for url, name in matches:
            print(f"{name}: {url}")
        return

    layer_url = resolve_layer_url(args)
    client = CSJStreetsClient(layer_url, page_size=args.page_size)

    if args.list_fields:
        print_fields(client.get_metadata())
        return

    bbox = None
    if args.bbox:
        bbox = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)

    if args.distinct_values:
        values = client.query_distinct_values(args.distinct_values, where=args.where, bbox=bbox)
        print(f"{len(values)} distinct value(s) for {args.distinct_values!r}:")
        for value in values:
            print(f"  {value!r}")
        return

    if args.output is None:
        raise SystemExit("--output is required (unless --list-fields/--list-layers/--distinct-values)")

    segments = client.query(bbox=bbox, where=args.where, historic_moment=args.historic_moment)
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
