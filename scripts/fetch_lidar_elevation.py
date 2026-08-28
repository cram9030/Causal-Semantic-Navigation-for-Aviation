#!/usr/bin/env python3
"""Fetch San Jose's LIDAR-derived elevation product, in EPSG:4326.

Phase 0 data collection (see `docs/INTEGRATION_PLAN.md` §5): unlike CSJ
Streets and San Jose's own imagery, the LIDAR DEM behind San Jose's
"Imagery & Elevation" data isn't served through `geo.sanjoseca.gov`'s
ArcGIS Server - Valley Water (Santa Clara Valley Water District) publishes
it as two static, whole-county ZIP downloads (``--product 1ft``/``5ft``),
with no scoped-query support at the source.

**Important:** the *entire* chosen product always downloads, regardless of
``--bbox`` - there is no way to fetch less, since Valley Water only offers
the whole-county archive. ``--bbox``/``--output`` (or ``--identify``) only
scope what gets *read out* of the already-downloaded local cache into a
small GeoTIFF (or a single value) - they do not reduce what's fetched over
the network or make the download itself faster. The archive and its
extracted contents are cached under ``--cache-dir`` and skipped on a later
run unless ``--overwrite``, so that (potentially very large) download only
happens once. Run with neither ``--bbox`` nor ``--identify`` to just
prefetch the archive and report what raster source(s) were found in it,
without reading or writing anything else.

Example::

    # one-time prefetch + inventory (no bbox/identify needed)
    python scripts/fetch_lidar_elevation.py --product 5ft

    # read a small AOI out of the (already-downloaded) archive
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
        help=(
            "EPSG:4326 envelope to read out of the local cache into --output "
            "(does not affect what's downloaded - see the note above)"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        help="output GeoTIFF path for --bbox's window (required if --bbox is given)",
    )
    parser.add_argument(
        "--identify", type=float, nargs=2, default=None, metavar=("LON", "LAT"),
        help="print the elevation at a single point read out of the local cache",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    client = LidarElevationClient(cache_dir=args.cache_dir, product=args.product)
    raster_paths = client.ensure_local(overwrite=args.overwrite)
    logger.info("%d raster source(s) available under %s", len(raster_paths), client.extract_dir)

    if args.identify:
        lon, lat = args.identify
        elevation = client.identify(lon, lat)
        print(elevation if elevation is not None else "NoData")
        return

    if args.bbox is None:
        # Prefetch-only mode: the archive is already downloaded/extracted
        # above (unconditionally) - nothing more to do without a bbox/point.
        for path in raster_paths:
            print(path)
        return

    if args.output is None:
        raise SystemExit("--output is required when --bbox is given")

    extent = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)
    tile = client.read_window(extent)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tile.to_geotiff(args.output)
    logger.info("wrote %s (%dx%d)", args.output, tile.width, tile.height)


if __name__ == "__main__":
    main()
