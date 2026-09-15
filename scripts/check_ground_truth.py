#!/usr/bin/env python3
"""Run automated sanity checks over a rasterized panoptic label set.

Wraps `csnav.data.ground_truth.checks.check_label_directory` as a CLI: reads
every label under ``--labels-dir`` (as written by ``scripts/build_ground_truth.py``)
and reports, per tile, whether its semantic/instance bands are internally
consistent with their own JSON sidecar (shape match, every instance id
accounted for, no orphan pixels), plus road/intersection pixel coverage and
how often the default fallback width had to be used.

This is the systematic half of ground-truth QA - it catches structural bugs
across an entire label set without a human looking at any of it. Visual
review (does a rasterized road actually sit on the road in the imagery) is
`scripts/visualize_ground_truth.py`'s job instead.

Exits non-zero if any tile has an ``"error"``-severity issue, so this is
usable as a CI gate on the ``build_ground_truth`` DVC stage's output.

``--default-width-report`` writes a CSV, one row per rasterized road that
fell back to the default width (``csnav.data.ground_truth.rasterize.DEFAULT_WIDTH_M``)
instead of a published CSJ width - the OBJECTID, name, and *every* raw CSJ
attribute for that segment, so a suspiciously uniform/narrow road width
(e.g. every road rendering at roughly the default, with no parking-lane
allowance) can be tracked back to specific OBJECTIDs and checked against
CSJ's own data. If most/all roads are falling back, the likely cause is
`csnav.data.arcgis.streets.WIDTH_FIELD_CANDIDATES` simply not matching the
real field name in the live schema - the ``attributes`` column shows
exactly what CSJ actually published, so the real field name (if there is
one) is visible directly in the CSV. Streamed row-by-row - safe against a
full-AOI label set of hundreds of thousands of tiles.

Example::

    uv run python scripts/check_ground_truth.py \\
        --labels-dir data/ground_truth/current \\
        --report out/ground_truth_check_report.json \\
        --default-width-report out/ground_truth_default_width_segments.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.ground_truth.checks import check_label_directory, iter_default_width_segments  # noqa: E402

logger = logging.getLogger("check_ground_truth")


def _write_default_width_report(labels_dir: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile", "objectid", "name", "width_m", "attributes"])
        for tile_key, segment in iter_default_width_segments(labels_dir):
            writer.writerow(
                [tile_key, segment.segment_id or "", segment.name or "", segment.width_m, json.dumps(segment.attributes)]
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None, help="write the full JSON report here")
    parser.add_argument(
        "--default-width-report", type=Path, default=None,
        help="write a CSV of every OBJECTID that fell back to the default width, with its raw CSJ attributes",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    result = check_label_directory(args.labels_dir)
    if not result.tiles:
        logger.warning("no labels found under %s", args.labels_dir)

    for tile in result.tiles:
        for issue in tile.issues:
            log = logger.error if issue.severity == "error" else logger.warning
            log("%s: %s", tile.tile_key, issue.message)

    logger.info(
        "checked %d tile(s): %d error(s), %d warning(s)",
        len(result.tiles), result.error_count, result.warning_count,
    )

    if args.report is not None:
        result.save(args.report)
        logger.info("wrote report to %s", args.report)

    if args.default_width_report is not None:
        count = _write_default_width_report(args.labels_dir, args.default_width_report)
        logger.info("wrote %d default-width segment(s) to %s", count, args.default_width_report)

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
