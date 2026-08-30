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

Example::

    uv run python scripts/check_ground_truth.py \\
        --labels-dir data/ground_truth/current \\
        --report out/ground_truth_check_report.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.ground_truth.checks import check_label_directory  # noqa: E402

logger = logging.getLogger("check_ground_truth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None, help="write the full JSON report here")
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

    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
