#!/usr/bin/env python3
"""Pull San Jose's Imagery & Elevation LIDAR product for an AOI, in EPSG:4326.

Phase 0 data collection (see `docs/INTEGRATION_PLAN.md` §5): fetches a
georeferenced elevation raster over a bounding box from San Jose's LIDAR
ImageServer, for later use in AGL correction and FOV occlusion modeling
(`docs/INTEGRATION_PLAN.md` §2) - not just visualization. The service is
resolved by name via ``ArcGISCatalog.discover_services`` rather than
hardcoded, for the same reason imagery services are discovered rather than
assumed (see `docs/phase0_arcgis_tile_client.md`).

Two modes:

* Default: export a raster covering ``--bbox`` and write it as a GeoTIFF.
* ``--identify LON LAT``: print a single point's elevation and exit, without
  fetching a raster - useful for a quick reachability/sanity check.

Example::

    python scripts/fetch_lidar_elevation.py \\
        --bbox -121.95 37.30 -121.85 37.36 \\
        --output data/raw/lidar/downtown_dem.tif
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.arcgis.catalog import ArcGISCatalog, DEFAULT_BASE_URL  # noqa: E402
from csnav.data.arcgis.elevation import LidarElevationClient, load_elevation_tile  # noqa: E402
from csnav.data.arcgis.models import Extent  # noqa: E402

logger = logging.getLogger("fetch_lidar_elevation")

DEFAULT_NAME_CONTAINS = "Elevation"


def resolve_service_url(args: argparse.Namespace) -> str:
    if args.service_url:
        return args.service_url
    catalog = ArcGISCatalog(base_url=args.base_url)
    matches = catalog.discover_services(
        root=args.root, name_contains=args.name_contains, service_types=("ImageServer",)
    )
    if not matches:
        raise SystemExit(
            f"no ImageServer matching {args.name_contains!r} found under {args.root or '(root)'!r} - "
            "pass --service-url directly if you already know it"
        )
    if len(matches) > 1:
        logger.warning(
            "%d ImageServer(s) matched %r; using the first: %s",
            len(matches), args.name_contains, matches[0].full_name,
        )
    service_url = catalog.service_rest_url(matches[0])
    logger.info("resolved elevation service: %s", service_url)
    return service_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="ArcGIS REST services directory root")
    parser.add_argument(
        "--name-contains", default=DEFAULT_NAME_CONTAINS,
        help="substring used to find the LIDAR elevation ImageServer",
    )
    parser.add_argument(
        "--root", default="",
        help="catalog folder to search under (default: the whole services directory)",
    )
    parser.add_argument(
        "--service-url", default=None, help="skip discovery and use this ImageServer URL directly"
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="EPSG:4326 envelope to export (required unless --identify is given)",
    )
    parser.add_argument("--width", type=int, default=1024, help="output raster width in pixels")
    parser.add_argument("--height", type=int, default=1024, help="output raster height in pixels")
    parser.add_argument("--pixel-type", default="F32", help="ArcGIS exportImage pixel type (default: F32)")
    parser.add_argument("--output", type=Path, help="output GeoTIFF path (required unless --identify is given)")
    parser.add_argument(
        "--identify", type=float, nargs=2, default=None, metavar=("LON", "LAT"),
        help="print the elevation at a single point instead of exporting a raster",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    service_url = resolve_service_url(args)
    client = LidarElevationClient(service_url)

    if args.identify:
        lon, lat = args.identify
        elevation = client.identify(lon, lat)
        print(elevation if elevation is not None else "NoData")
        return

    if args.bbox is None or args.output is None:
        raise SystemExit("--bbox and --output are required unless --identify is given")

    extent = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)
    image_bytes = client.export_elevation(extent, width=args.width, height=args.height, pixel_type=args.pixel_type)
    tile = load_elevation_tile(image_bytes, extent, width=args.width, height=args.height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tile.to_geotiff(args.output)
    logger.info("wrote %s (%dx%d)", args.output, args.width, args.height)


if __name__ == "__main__":
    main()
