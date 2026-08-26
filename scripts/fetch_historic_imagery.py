#!/usr/bin/env python3
"""Fetch San Jose DPW imagery tiles across *all* historic vintages for an AOI.

Phase 0 data collection: discovers every ``DPW_Imagery*`` service published
under the ``Imagery`` folder of geo.sanjoseca.gov (the current cached
basemap, ``DPW_ImageryCached``, plus every dated historic capture), fetches
the tiles covering the requested area of interest for each one, reprojects
them from EPSG:3857 to EPSG:4326, and writes one GeoTIFF per tile under
``<output-dir>/<service-name>/``.

Training data for this project needs the full historic archive, not just the
newest imagery, so this script never limits itself to the most recent
service - every match returned by the catalog is fetched.

Re-running the same command resumes rather than re-downloading: a tile whose
output GeoTIFF already exists is skipped unless ``--overwrite`` is passed.

Example::

    python scripts/fetch_historic_imagery.py \\
        --bbox -121.95 37.30 -121.85 37.36 \\
        --output-dir data/raw/dpw_imagery
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.arcgis.catalog import ArcGISCatalog, DEFAULT_BASE_URL, extract_year  # noqa: E402
from csnav.data.arcgis.client import ArcGISTileClient  # noqa: E402
from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.data.arcgis.projections import extent_4326_to_3857  # noqa: E402
from csnav.data.arcgis.reproject import reproject_tile_to_4326  # noqa: E402
from csnav.data.arcgis.tiles import tile_bounds, tiles_covering_extent  # noqa: E402

logger = logging.getLogger("fetch_historic_imagery")


def fetch_service(
    ref,
    catalog: ArcGISCatalog,
    aoi_4326: Extent,
    output_dir: Path,
    level: int | None,
    overwrite: bool = False,
) -> int:
    service_url = catalog.service_rest_url(ref)
    client = ArcGISTileClient(service_url)
    meta = client.get_metadata()

    if not meta.supports_tiles:
        logger.warning("skipping %s: not a cached tile service", ref.full_name)
        return 0

    tile_info = meta.tile_info
    target_level = level if level is not None else tile_info.max_level
    aoi_3857 = extent_4326_to_3857(aoi_4326)
    resolution = tile_info.lod_for_level(target_level).resolution
    logger.info(
        "%s: fetching level %d (%.3f units/px) - not every tile in the AOI's grid is "
        "necessarily cached at this resolution; missing individual tiles are expected "
        "and skipped, not an error",
        ref.full_name, target_level, resolution,
    )

    dest = output_dir / ref.name
    dest.mkdir(parents=True, exist_ok=True)

    # Materialized (not left as a generator) so tqdm can show a total/ETA
    # instead of just a bare counter - AOI tile counts are small enough
    # (thousands, not millions) for this to be cheap.
    tile_coords = list(tiles_covering_extent(tile_info, target_level, aoi_3857))

    written = 0
    skipped = 0
    missing = 0
    failed = 0
    progress = tqdm(tile_coords, desc=ref.name, unit="tile", leave=False)
    for row, col in progress:
        out_path = dest / f"{target_level}_{row}_{col}.tif"
        if out_path.exists() and not overwrite:
            # Already downloaded by an earlier run - resume rather than
            # re-fetching and re-warping it.
            skipped += 1
            progress.set_postfix(written=written, skipped=skipped, missing=missing, failed=failed)
            continue

        bounds = tile_bounds(tile_info, target_level, row, col)
        try:
            image_bytes = client.fetch_tile_auto(target_level, row, col)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # No cached tile at this level/row/col - normal at fine zoom
                # levels where only part of the AOI has that much detail.
                logger.debug("no cached tile for %s z%d/%d/%d", ref.name, target_level, row, col)
                missing += 1
            else:
                logger.exception("failed to fetch %s tile z%d/%d/%d", ref.name, target_level, row, col)
                failed += 1
            progress.set_postfix(written=written, skipped=skipped, missing=missing, failed=failed)
            continue
        except Exception:  # noqa: BLE001 - keep collecting the rest of the AOI
            logger.exception("failed to fetch %s tile z%d/%d/%d", ref.name, target_level, row, col)
            failed += 1
            progress.set_postfix(written=written, skipped=skipped, missing=missing, failed=failed)
            continue

        reprojected = reproject_tile_to_4326(image_bytes, bounds)
        # Write under a temp name and rename into place atomically, so a run
        # interrupted mid-write never leaves a partial file that a later
        # resume would mistake for a completed download.
        tmp_path = out_path.with_name(out_path.name + ".part")
        reprojected.to_geotiff(tmp_path)
        tmp_path.replace(out_path)
        written += 1
        progress.set_postfix(written=written, skipped=skipped, missing=missing, failed=failed)
    progress.close()

    logger.info(
        "%s: wrote %d tiles, %d already downloaded, %d not cached at this level, %d failed (year=%s)",
        ref.full_name, written, skipped, missing, failed, extract_year(ref.name),
    )
    if written == 0 and skipped == 0 and missing > 0 and failed == 0:
        logger.warning(
            "%s: every requested tile was missing at level %d - try a coarser --level "
            "(this service may only have deep-zoom coverage for part of the AOI)",
            ref.full_name, target_level,
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="ArcGIS REST services directory root"
    )
    parser.add_argument(
        "--name-contains", default="DPW_Imagery",
        help="substring used to match imagery service names (matches all historic vintages)",
    )
    parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=None, help="tile LOD level; defaults to each service's finest level")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-fetch tiles even if their output GeoTIFF already exists (default: skip them)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    catalog = ArcGISCatalog(base_url=args.base_url)
    services = catalog.discover_imagery_services(name_contains=args.name_contains)
    if not services:
        logger.error("no imagery services matched %r under the catalog", args.name_contains)
        raise SystemExit(1)

    logger.info("found %d imagery service(s): %s", len(services), ", ".join(s.name for s in services))

    aoi = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)

    total = 0
    for ref in tqdm(services, desc="services", unit="service"):
        total += fetch_service(ref, catalog, aoi, args.output_dir, args.level, overwrite=args.overwrite)

    logger.info("done: %d tiles written across %d service(s)", total, len(services))


if __name__ == "__main__":
    main()
