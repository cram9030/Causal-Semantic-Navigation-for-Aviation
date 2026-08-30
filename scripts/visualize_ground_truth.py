#!/usr/bin/env python3
"""Render the ground-truth review map and QA gallery for a rasterized label set.

Produces, from one ``scripts/build_ground_truth.py`` output directory:

* ``<--map>`` - a folium review map (`csnav.viz.ground_truth_view`): every
  tile's footprint, and every rasterized road/intersection vectorized back
  out of the label rasters and drawn where it actually sits, over San Jose
  imagery. The "does this look right geographically" view.
* ``<--gallery-dir>/index.html`` - a static paging gallery
  (`csnav.viz.ground_truth_gallery`): one imagery/label/thumbnail PNG triple
  per tile, with a slider to blend imagery and label at any strength, arrow-key
  navigation, and a flaggable checklist - built for quickly moving through and
  exhaustively verifying a large label set by eye, tile by tile.

Both need the label set's paired imagery (to draw the basemap tooltip context
for the map, and to render the gallery's imagery PNGs) - pass the same
``--imagery-dir`` the labels were built against.

Example::

    uv run python scripts/visualize_ground_truth.py \\
        --labels-dir data/ground_truth/current \\
        --imagery-dir data/raw/dpw_imagery/DPW_ImageryCached2025 \\
        --map out/viz/ground_truth_map.html \\
        --gallery-dir out/viz/ground_truth_gallery
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.ground_truth.labels import PanopticLabel  # noqa: E402

logger = logging.getLogger("visualize_ground_truth")


def _load_labels(labels_dir: Path) -> list[PanopticLabel]:
    labels = [PanopticLabel.load(sidecar.with_suffix(".tif"), sidecar) for sidecar in sorted(labels_dir.glob("*.json"))]
    if not labels:
        raise SystemExit(f"no labels found under {labels_dir}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--imagery-dir", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=None, help="path to write the folium review map to")
    parser.add_argument("--gallery-dir", type=Path, default=None, help="directory to write the QA gallery into")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.map is None and args.gallery_dir is None:
        raise SystemExit("pass at least one of --map / --gallery-dir")

    labels = _load_labels(args.labels_dir)
    logger.info("loaded %d label(s) from %s", len(labels), args.labels_dir)

    if args.map is not None:
        from csnav.viz.ground_truth_view import ground_truth_review_map, save_ground_truth_map

        fmap = ground_truth_review_map(labels)
        path = save_ground_truth_map(fmap, args.map)
        logger.info("wrote review map to %s", path)

    if args.gallery_dir is not None:
        from csnav.viz.ground_truth_gallery import build_gallery

        missing = 0
        pairs = []
        for label in labels:
            imagery_path = args.imagery_dir / f"{label.stem}.tif"
            if not imagery_path.exists():
                logger.warning("no imagery for tile %s at %s - skipping in gallery", label.stem, imagery_path)
                missing += 1
                continue
            pairs.append((label, imagery_path))
        if not pairs:
            raise SystemExit(f"no label had matching imagery under {args.imagery_dir}")
        index = build_gallery(pairs, args.gallery_dir)
        logger.info("wrote gallery for %d tile(s) (%d skipped, missing imagery) to %s", len(pairs), missing, index)


if __name__ == "__main__":
    main()
