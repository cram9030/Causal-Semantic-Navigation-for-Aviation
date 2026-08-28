#!/usr/bin/env python3
"""Fetch San Jose's LIDAR-derived elevation product for an AOI, in EPSG:4326.

Phase 0 data collection (see `docs/INTEGRATION_PLAN.md` §5): unlike CSJ
Streets and San Jose's own imagery, the LIDAR DEM behind San Jose's
"Imagery & Elevation" data isn't served through `geo.sanjoseca.gov`'s
ArcGIS Server - Valley Water (Santa Clara Valley Water District) publishes
it as two static, whole-county ZIP downloads (``--product 1ft``/``5ft``).
There's nothing to discover here, so this script just downloads + extracts
the chosen product once (cached under ``--cache-dir``; skipped on a later
run unless ``--overwrite``), then reads the window covering ``--bbox`` (or a
tiny window around ``--identify``'s point) out of whichever raster(s) the
archive contains, mosaicking/reprojecting to EPSG:4326 as needed - see
`csnav.data.lidar` for details.

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
from csnav.data.lidar import LIDAR_PRODUCT_URLS, LidarElevationClient  # noqa: E402

logger = logging.getLogger("fetch_lidar_elevation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--product", choices=sorted(LIDAR_PRODUCT_URLS), default="5ft",
        help="which whole-county DEM resolution to use (default: 5ft; 1ft is a much larger download)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/raw/lidar_archive"),
        help="where the downloaded archive and extracted rasters are cached across runs",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-download/re-extract even if already cached under --cache-dir",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=None, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="EPSG:4326 envelope to read (required unless --identify is given)",
    )
    parser.add_argument("--output", type=Path, help="output GeoTIFF path (required unless --identify is given)")
    parser.add_argument(
        "--identify", type=float, nargs=2, default=None, metavar=("LON", "LAT"),
        help="print the elevation at a single point instead of reading a window",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    client = LidarElevationClient(cache_dir=args.cache_dir, product=args.product)
    raster_paths = client.ensure_local(overwrite=args.overwrite)
    logger.info("%d raster file(s) available under %s", len(raster_paths), client.extract_dir)

    if args.identify:
        lon, lat = args.identify
        elevation = client.identify(lon, lat)
        print(elevation if elevation is not None else "NoData")
        return

    if args.bbox is None or args.output is None:
        raise SystemExit("--bbox and --output are required unless --identify is given")

    extent = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)
    tile = client.read_window(extent)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tile.to_geotiff(args.output)
    logger.info("wrote %s (%dx%d)", args.output, tile.width, tile.height)


if __name__ == "__main__":
    main()
