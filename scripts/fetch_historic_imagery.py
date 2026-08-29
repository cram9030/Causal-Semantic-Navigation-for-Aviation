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

Before committing to a full run, each service's chosen level is sanity
checked with a small sample of tile requests spread across the AOI. Some
ArcGIS caches only generate tiles for part of an extent at their finest
zoom level (or not at all for a given AOI) - without this check, a level
with zero coverage would silently grind through every tile in the AOI as
individual 404s, which can take hours to fail. When no ``--level`` is
given, the default is *not* simply "the finest level" - it's the finest
level that the sample check finds any coverage for, probed from finest to
coarsest.

Example::

    uv run python scripts/fetch_historic_imagery.py \\
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
from csnav.data.arcgis.models import Extent, TileInfo  # noqa: E402
from csnav.data.arcgis.projections import extent_4326_to_3857  # noqa: E402
from csnav.data.arcgis.reproject import reproject_tile_to_4326  # noqa: E402
from csnav.data.arcgis.tiles import (  # noqa: E402
    sample_tiles_covering_extent,
    tile_bounds,
    tile_count_covering_extent,
    tiles_covering_extent,
)

logger = logging.getLogger("fetch_historic_imagery")

# How many tiles to sample when checking whether a level has any cached
# coverage at all for the AOI, before committing to a full run. Spread
# across the AOI rather than clustered, so a small sample can distinguish
# "nothing cached here" from "normal, scattered gaps" in seconds instead of
# requesting every one of what can be millions of tiles at a fine level.
COVERAGE_SAMPLE_SIZE = 25


def _has_any_coverage(
    client: ArcGISTileClient,
    tile_info: TileInfo,
    level: int,
    aoi_3857: Extent,
    sample_size: int = COVERAGE_SAMPLE_SIZE,
) -> bool:
    """Cheaply check whether *any* tile exists at ``level`` within the AOI.

    Uses :func:`sample_tiles_covering_extent`, which never materializes the
    full tile grid for the level being checked - important because a level
    we ultimately reject can still cover millions (or, for a wide AOI at a
    fine level, billions) of tiles.

    A 404 means "not cached here", which is expected and doesn't stop the
    sample. Any other error (network issue, 5xx, ...) is a real problem and
    is left to propagate rather than being mistaken for "no coverage".
    """
    sample = sample_tiles_covering_extent(tile_info, level, aoi_3857, sample_size)
    for row, col in sample:
        try:
            client.fetch_tile_auto(level, row, col)
            return True
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                continue
            raise
    return False


def fetch_service(
    ref,
    catalog: ArcGISCatalog,
    aoi_4326: Extent,
    output_dir: Path,
    level: int | None,
    overwrite: bool = False,
    coverage_sample_size: int = COVERAGE_SAMPLE_SIZE,
    skip_coverage_check: bool = False,
) -> int:
    service_url = catalog.service_rest_url(ref)
    client = ArcGISTileClient(service_url)
    meta = client.get_metadata()

    if not meta.supports_tiles:
        logger.warning("skipping %s: not a cached tile service", ref.full_name)
        return 0

    tile_info = meta.tile_info
    aoi_3857 = extent_4326_to_3857(aoi_4326)

    if level is not None:
        target_level = level
        total = tile_count_covering_extent(tile_info, target_level, aoi_3857)
        if (
            not skip_coverage_check
            and total > 0
            and not _has_any_coverage(client, tile_info, target_level, aoi_3857, coverage_sample_size)
        ):
            logger.error(
                "%s: level %d has no cached tiles anywhere in this AOI (sampled %d of "
                "%d tiles) - pick a different --level, or pass --skip-coverage-check if "
                "you believe this is a false negative. Levels this service has: %s",
                ref.full_name, target_level, min(total, coverage_sample_size),
                total, sorted(lod.level for lod in tile_info.lods),
            )
            return 0
    elif skip_coverage_check:
        # No level given and no check requested - fall back to the naive
        # "finest level" default, uncoverage-checked, as before this feature
        # existed.
        target_level = tile_info.max_level
        logger.warning(
            "%s: --skip-coverage-check set, using finest level %d without checking for "
            "coverage first",
            ref.full_name, target_level,
        )
    else:
        # The naive default would be "the finest level", but that's often
        # only cached for part of the AOI - or none of it, as at San Jose's
        # deepest levels. Probe from finest to coarsest and use the first
        # one with any sampled coverage instead. Only counts/samples are
        # computed per candidate level - never the full tile list - since a
        # level we end up rejecting can still cover a huge grid.
        target_level = None
        for lod in sorted(tile_info.lods, key=lambda lod: lod.level, reverse=True):
            if tile_count_covering_extent(tile_info, lod.level, aoi_3857) == 0:
                continue
            if _has_any_coverage(client, tile_info, lod.level, aoi_3857, coverage_sample_size):
                target_level = lod.level
                break
            logger.info(
                "%s: level %d (%.3f units/px) has no sampled coverage for this AOI, "
                "trying a coarser level",
                ref.full_name, lod.level, lod.resolution,
            )
        if target_level is None:
            logger.error("%s: no cached tiles found for this AOI at any level - skipping", ref.full_name)
            return 0

    # Only materialized now, for the single level actually being fetched.
    tile_coords = list(tiles_covering_extent(tile_info, target_level, aoi_3857))
    resolution = tile_info.lod_for_level(target_level).resolution
    logger.info(
        "%s: fetching level %d (%.3f units/px), %d tile(s) in the AOI's grid - not every "
        "one is necessarily cached; missing individual tiles are expected and skipped, "
        "not an error",
        ref.full_name, target_level, resolution, len(tile_coords),
    )

    dest = output_dir / ref.name
    dest.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument(
        "--level", type=int, default=None,
        help=(
            "tile LOD level; defaults to the finest level the coverage check finds any "
            "cached tiles at for this AOI (see --coverage-sample-size)"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-fetch tiles even if their output GeoTIFF already exists (default: skip them)",
    )
    parser.add_argument(
        "--coverage-sample-size", type=int, default=COVERAGE_SAMPLE_SIZE,
        help=(
            "tiles to sample when checking a level has any cached coverage before "
            f"committing to a full run (default: {COVERAGE_SAMPLE_SIZE})"
        ),
    )
    parser.add_argument(
        "--skip-coverage-check", action="store_true",
        help=(
            "skip the coverage sample check entirely: with --level, fetch it "
            "unconditionally; without --level, fall back to the finest level "
            "unchecked"
        ),
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
        total += fetch_service(
            ref, catalog, aoi, args.output_dir, args.level,
            overwrite=args.overwrite, coverage_sample_size=args.coverage_sample_size,
            skip_coverage_check=args.skip_coverage_check,
        )

    logger.info("done: %d tiles written across %d service(s)", total, len(services))


if __name__ == "__main__":
    main()
