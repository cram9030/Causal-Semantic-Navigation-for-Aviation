#!/usr/bin/env python3
"""Fetch ground elevation for an AOI from USGS 3DEP, in EPSG:4326.

Phase 0 data collection (see `docs/INTEGRATION_PLAN.md` §5): San Jose's own
LIDAR product (Valley Water) turned out to be contour lines, not a raster
DEM - see `docs/phase0_csj_streets_lidar.md` for that investigation and why
this uses USGS's 3D Elevation Program (3DEP) national elevation ImageServer
instead (`csnav.data.lidar`). It's a live per-request raster query, so
unlike a Valley Water-style whole-archive download, every run fetches
exactly the requested AOI/point - nothing is cached locally.

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

from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.data.lidar import DEFAULT_SERVICE_URL, LidarElevationClient  # noqa: E402

logger = logging.getLogger("fetch_lidar_elevation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--service-url", default=DEFAULT_SERVICE_URL,
        help="USGS 3DEP elevation ImageServer URL (default: the national 3DEPElevation service)",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="EPSG:4326 envelope to fetch (required unless --identify is given)",
    )
    parser.add_argument("--output", type=Path, help="output GeoTIFF path (required if --bbox is given)")
    parser.add_argument("--width", type=int, default=512, help="output raster width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="output raster height in pixels (default: 512)")
    parser.add_argument("--pixel-type", default="F32", help="ArcGIS exportImage pixel type (default: F32)")
    parser.add_argument(
        "--identify", type=float, nargs=2, default=None, metavar=("LON", "LAT"),
        help="print the elevation at a single point instead of fetching a raster",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    client = LidarElevationClient(service_url=args.service_url)

    if args.identify:
        lon, lat = args.identify
        elevation = client.identify(lon, lat)
        print(elevation if elevation is not None else "NoData")
        return

    if args.bbox is None or args.output is None:
        raise SystemExit("--bbox and --output are required unless --identify is given")

    extent = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)
    tile = client.read_window(extent, width=args.width, height=args.height, pixel_type=args.pixel_type)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tile.to_geotiff(args.output)
    logger.info("wrote %s (%dx%d)", args.output, tile.width, tile.height)


if __name__ == "__main__":
    main()
