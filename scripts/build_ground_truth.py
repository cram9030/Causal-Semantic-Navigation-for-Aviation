#!/usr/bin/env python3
"""Rasterize CSJ street geometry into panoptic ground-truth labels over imagery tiles.

Phase 2 (`docs/INTEGRATION_PLAN.md` §5): "rasterize ground-truth panoptic
labels using CSJ street geometry/widths over San Jose imagery tiles". For
each imagery tile this writes a `csnav.data.ground_truth.PanopticLabel` -
a 2-band GeoTIFF (semantic class id, instance id) plus a JSON sidecar - under
``--output-dir``, pixel-aligned to that tile's own imagery raster.

Two ways to choose which tiles get labeled:

* **Default - every tile under ``--imagery-dir``.** Scans for the
  ``{level}_{row}_{col}.tif`` files `scripts/fetch_historic_imagery.py`
  already writes and rasterizes all of them - the full-AOI grid Mask2Former
  training needs, independent of any particular trajectory set.
* **``--manifest`` - only the tiles a pinned `ManifestBundle` references**
  (`ManifestBundle.all_tiles()`). Narrows the run to one CONOPS/trajectory
  set's actual coverage - useful for a quick regional-sensitivity check or
  for verifying a manifest's tile list against real imagery without
  rasterizing the whole AOI. Tiles the manifest references but whose imagery
  file isn't present under ``--imagery-dir`` are skipped (logged, not an
  error) exactly like a missing fetch tile is elsewhere in this pipeline.

Street geometry always comes from an archived GeoJSON pull
(``--streets-geojson``, e.g. written by ``scripts/fetch_csj_streets.py``),
never a live query - CSJ Streets refreshes weekly and ground truth for a
given imagery vintage should stay pinned to the street network as it stood
when that vintage was flown/captured. Pairing older imagery with a matching
historic street snapshot (rather than today's network) is what
``fetch_csj_streets.py --historic-moment`` is for, where the live layer turns
out to support it - see `docs/phase2_ground_truth_rasterization.md`.

Example (full AOI grid, current imagery + current streets)::

    uv run python scripts/build_ground_truth.py \\
        --imagery-dir data/raw/dpw_imagery/DPW_ImageryCached2025 \\
        --streets-geojson data/raw/csj_streets/aoi.geojson \\
        --output-dir data/ground_truth/current

Example (scoped to one pinned manifest bundle, for a sensitivity check)::

    uv run python scripts/build_ground_truth.py \\
        --imagery-dir data/raw/dpw_imagery/DPW_ImageryCached2025 \\
        --streets-geojson data/raw/csj_streets/aoi.geojson \\
        --manifest data/manifests/san_jose_downtown.json \\
        --output-dir out/ground_truth_review/san_jose_downtown_r250
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shapely.geometry import box  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

import rasterio  # noqa: E402

from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.data.arcgis.streets import segment_geometry, segments_from_geojson  # noqa: E402
from csnav.data.ground_truth.labels import PanopticLabel  # noqa: E402
from csnav.data.ground_truth.rasterize import (  # noqa: E402
    DEFAULT_INTERSECTION_RADIUS_M,
    DEFAULT_INTERSECTION_SNAP_M,
    DEFAULT_WIDTH_M,
    GroundTruthBuilder,
)
from csnav.trajectory.coverage import TileRef  # noqa: E402
from csnav.trajectory.manifest import ManifestBundle  # noqa: E402

logger = logging.getLogger("build_ground_truth")

#: Matches the ``{level}_{row}_{col}.tif`` naming `fetch_historic_imagery.py` writes.
_TILE_FILENAME = re.compile(r"^(\d+)_(\d+)_(\d+)\.tif$")

#: EPSG:4326 is the only CRS this module rasterizes onto - see
#: `csnav.data.arcgis.reproject.reproject_tile_to_4326`, which is what
#: produces the imagery GeoTIFFs this script reads.
EXPECTED_CRS = "EPSG:4326"


def _tiles_from_imagery_dir(imagery_dir: Path) -> list[TileRef]:
    """Every ``{level}_{row}_{col}.tif`` under ``imagery_dir``, bounds read from the raster itself."""
    tiles: list[TileRef] = []
    for path in sorted(imagery_dir.glob("*.tif")):
        match = _TILE_FILENAME.match(path.name)
        if not match:
            logger.debug("skipping %s: does not match {level}_{row}_{col}.tif", path.name)
            continue
        level, row, col = (int(group) for group in match.groups())
        with rasterio.open(path) as src:
            xmin, ymin, xmax, ymax = src.bounds
        tiles.append(TileRef(level=level, row=row, col=col, bounds=Extent(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)))
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagery-dir", type=Path, required=True, help="directory of {level}_{row}_{col}.tif tiles")
    parser.add_argument("--streets-geojson", type=Path, required=True, help="archived CSJ Streets GeoJSON pull")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="restrict to the tiles referenced by this pinned ManifestBundle JSON, instead of every "
        "tile under --imagery-dir",
    )
    parser.add_argument("--default-width-m", type=float, default=DEFAULT_WIDTH_M)
    parser.add_argument("--intersection-radius-m", type=float, default=DEFAULT_INTERSECTION_RADIUS_M)
    parser.add_argument("--intersection-snap-m", type=float, default=DEFAULT_INTERSECTION_SNAP_M)
    parser.add_argument(
        "--overwrite", action="store_true", help="re-rasterize tiles even if their label already exists"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    segments = segments_from_geojson(json.loads(args.streets_geojson.read_text(encoding="utf-8")))
    logger.info("loaded %d street segments from %s", len(segments), args.streets_geojson)
    geometries = [segment_geometry(segment) for segment in segments]
    tree = STRtree(geometries)

    if args.manifest is not None:
        tiles = list(ManifestBundle.load(args.manifest).all_tiles())
        logger.info("restricting to %d tile(s) referenced by manifest %s", len(tiles), args.manifest)
    else:
        tiles = _tiles_from_imagery_dir(args.imagery_dir)
        logger.info("found %d tile(s) under %s", len(tiles), args.imagery_dir)

    builder = GroundTruthBuilder(
        default_width_m=args.default_width_m,
        intersection_radius_m=args.intersection_radius_m,
        intersection_snap_m=args.intersection_snap_m,
    )

    written = 0
    skipped_existing = 0
    skipped_missing_imagery = 0
    empty = 0
    for tile in tiles:
        stem = f"{tile.level}_{tile.row}_{tile.col}"
        imagery_path = args.imagery_dir / f"{stem}.tif"
        raster_path = args.output_dir / f"{stem}.tif"
        if raster_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        if not imagery_path.exists():
            logger.debug("no imagery for tile %s at %s - skipping", stem, imagery_path)
            skipped_missing_imagery += 1
            continue

        with rasterio.open(imagery_path) as src:
            if str(src.crs) != EXPECTED_CRS:
                raise ValueError(f"{imagery_path} is {src.crs}, expected {EXPECTED_CRS}")
            width, height, transform = src.width, src.height, src.transform

        tile_box = box(tile.bounds.xmin, tile.bounds.ymin, tile.bounds.xmax, tile.bounds.ymax)
        candidates = [segments[i] for i in tree.query(tile_box)]

        label = builder.rasterize(
            candidates, tile, width, height, transform,
            streets_source=str(args.streets_geojson),
            imagery_source=str(imagery_path),
        )
        label.save(args.output_dir)
        written += 1
        if not label.segments:
            empty += 1

    logger.info(
        "done: wrote %d label(s) (%d background-only), %d already present, %d missing imagery",
        written, empty, skipped_existing, skipped_missing_imagery,
    )


if __name__ == "__main__":
    main()
