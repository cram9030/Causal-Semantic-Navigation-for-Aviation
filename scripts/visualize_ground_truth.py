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

**Memory**: labels are loaded from disk one at a time, never all at once - a
full-AOI label set can be hundreds of tiles, each carrying two full-resolution
rasters, and holding them all in memory simultaneously is what used to get
this script killed by the OOM killer on a memory-constrained devcontainer
(SIGKILL, exit 137, no traceback - a killed process can't raise one). Only
the (much smaller) vectorized map features and per-tile gallery images stay
around across the whole run.

**Selecting a subset**: pass ``--limit N`` or ``--sample N`` to only process
part of a large label set. A full-AOI label set is easily hundreds of
thousands of tiles, and two things stop scaling well before raw Python
memory does even with the one-at-a-time loading above: the review map draws
everything into *one* map, so its vectorized-feature list still grows with
however many tiles are included; and the gallery's own ``index.html`` embeds
one entry (and builds one thumbnail DOM node) per tile, which a browser
can't render at that count regardless of how the PNGs were generated. Use
``--manifest <bundle.json>`` on ``scripts/build_ground_truth.py`` to build
(and then visualize) a much smaller set scoped to one trajectory/scenario's
actual coverage instead, if what you actually need is a human-reviewable
sample rather than every training tile.

**Resuming a large gallery run**: the gallery skips re-rendering a tile
whose three PNGs already exist, so an interrupted run (hundreds of
thousands of files takes a while) can just be re-invoked with the same
arguments to pick up where it left off. Pass ``--overwrite`` to force a
full re-render instead. A later run into the same ``--gallery-dir`` doesn't
have to use the same ``--limit``/``--sample`` selection either - the page
accumulates every tile any run has ever rendered into it (via a
``tiles.json`` manifest sidecar the gallery keeps for exactly this), so a
smaller follow-up selection still leaves everything from a larger earlier
run visible rather than dropping it from the page.

**This resumability cuts both ways**: the skip-if-exists check has no way
to tell that ``--labels-dir``'s data changed since a tile's 3 PNGs were
last written - only that they exist. Re-running against the same
``--gallery-dir`` after rebuilding the labels upstream (a streets re-fetch,
a rasterization fix) without ``--overwrite`` will silently keep showing the
*old* images - this script logs a warning when it detects an existing
``--gallery-dir`` with no ``--overwrite`` passed, precisely because this
has already been mistaken for a rendering bug once. See
``docs/phase2_ground_truth_rasterization.md``'s "Refreshing after an
upstream fix" section for the full sequence this fits into.

Example (whole label set)::

    uv run python scripts/visualize_ground_truth.py \\
        --labels-dir data/ground_truth/current \\
        --imagery-dir data/raw/dpw_imagery/DPW_ImageryCached2025 \\
        --map out/viz/ground_truth_map.html \\
        --gallery-dir out/viz/ground_truth_gallery

Example (a representative 50-tile sample only, e.g. for a quick check or a
memory-constrained environment)::

    uv run python scripts/visualize_ground_truth.py \\
        --labels-dir data/ground_truth/current \\
        --imagery-dir data/raw/dpw_imagery/DPW_ImageryCached2025 \\
        --sample 50 \\
        --map out/viz/ground_truth_map_sample.html \\
        --gallery-dir out/viz/ground_truth_gallery_sample
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.ground_truth.labels import PanopticLabel  # noqa: E402

logger = logging.getLogger("visualize_ground_truth")


#: Matches `scripts/build_ground_truth.py`'s own ``{level}_{row}_{col}`` tile stem naming.
_TILE_STEM = re.compile(r"^(\d+)_(\d+)_(\d+)$")


def _tile_sort_key(path: Path) -> tuple[int, int, int, str]:
    """Sort key for a tile sidecar path: numeric ``(level, row, col)`` where the stem parses as
    one, the raw name otherwise (defensive - every stem this pipeline writes parses).

    Plain string sorting of ``{level}_{row}_{col}.json`` is not a numeric sort - unpadded
    numbers compare wrong (``"19_1_1"`` sorts before ``"21_1_1"`` only by luck of the leading
    digit, and a mix of zoom levels/regions in one ``--labels-dir`` can otherwise put an
    unrelated run of tiles first for no reason connected to their actual level/position). This
    only changes *order*, not which tiles are includable - see `_paths_with_imagery` for the
    fix that actually matters when a labels directory has outgrown its imagery.
    """
    match = _TILE_STEM.match(path.stem)
    if match is None:
        return (-1, -1, -1, path.stem)
    level, row, col = (int(group) for group in match.groups())
    return (level, row, col, "")


def _sidecar_paths(labels_dir: Path) -> list[Path]:
    """Every label's ``.json`` sidecar path under ``labels_dir``, sorted - cheap: no rasters opened."""
    paths = sorted(labels_dir.glob("*.json"), key=_tile_sort_key)
    if not paths:
        raise SystemExit(f"no labels found under {labels_dir}")
    return paths


def _paths_with_imagery(paths: list[Path], imagery_dir: Path) -> list[Path]:
    """``paths`` narrowed to those with a matching imagery file under ``imagery_dir``.

    Existence checks only (no raster reads) - the same check `_iter_gallery_pairs` already
    does lazily per tile, just done eagerly here, before `_select` commits to a specific
    ``--limit``/``--sample`` subset. Without this, a gallery's selection is drawn from
    *every* label in ``--labels-dir`` regardless of whether imagery for it still exists - and a
    labels directory that has outgrown or outlived the imagery it was built against (rebuilt
    against a narrower ``--imagery-dir``, or carrying tiles from an earlier, larger/different
    pull) can leave `--limit N` picking N tiles that will *never* render, failing with "no label
    had matching imagery" even though plenty of renderable tiles exist elsewhere in the set -
    a real incident, see docs/phase2_ground_truth_rasterization.md.
    """
    return [path for path in paths if (imagery_dir / f"{path.stem}.tif").exists()]


def _select(paths: list[Path], limit: int | None, sample: int | None) -> list[Path]:
    """Narrow ``paths`` to a subset per ``--limit``/``--sample``, or return them unchanged."""
    if limit is not None:
        selected = paths[:limit]
        logger.info("--limit %d: using the first %d of %d tile(s)", limit, len(selected), len(paths))
        return selected
    if sample is not None:
        if sample >= len(paths):
            return paths
        # Evenly spaced indices across the sorted set, not just a run from the
        # start - a representative spread across the AOI rather than
        # whichever corner sorts first.
        step = len(paths) / sample
        indices = sorted({int(i * step) for i in range(sample)})
        selected = [paths[i] for i in indices]
        logger.info("--sample %d: using %d evenly-spaced tile(s) of %d", sample, len(selected), len(paths))
        return selected
    return paths


def _iter_labels(paths: list[Path]) -> Iterator[PanopticLabel]:
    """Load one label from disk at a time - never more than one tile's rasters in memory here."""
    for sidecar in paths:
        yield PanopticLabel.load(sidecar.with_suffix(".tif"), sidecar)


def _iter_gallery_pairs(
    paths: list[Path], imagery_dir: Path, counts: dict[str, int]
) -> Iterator[tuple[PanopticLabel, Path]]:
    """Yield ``(label, imagery_path)`` lazily, skipping tiles with no matching imagery file.

    ``counts`` is a plain dict the caller passes in and reads after the
    generator is fully consumed (e.g. by :func:`csnav.viz.ground_truth_gallery.build_gallery`
    finishing its own single pass) - a generator can't return a summary any
    other way without giving up its laziness.
    """
    for label in _iter_labels(paths):
        imagery_path = imagery_dir / f"{label.stem}.tif"
        if not imagery_path.exists():
            logger.warning("no imagery for tile %s at %s - skipping in gallery", label.stem, imagery_path)
            counts["missing"] += 1
            continue
        counts["matched"] += 1
        yield label, imagery_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--imagery-dir", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=None, help="path to write the folium review map to")
    parser.add_argument("--gallery-dir", type=Path, default=None, help="directory to write the QA gallery into")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="only process the first N tiles (sorted by tile key) instead of the whole label set",
    )
    selection.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="only process N tiles, evenly spaced across the sorted label set",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-render gallery images even if they already exist (default: skip and resume)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if args.map is None and args.gallery_dir is None:
        raise SystemExit("pass at least one of --map / --gallery-dir")

    all_paths = _sidecar_paths(args.labels_dir)

    #: Past this many tiles, both outputs stop being practical regardless of
    #: Python-side memory: the map's own feature count and the gallery's
    #: per-tile DOM/file count are what run out next. Not a hard limit -
    #: just a nudge toward --sample/--manifest before spending the time.
    LARGE_SELECTION_WARNING_THRESHOLD = 500

    def _warn_if_unbounded_and_large(destination: str, selected: list[Path]) -> None:
        if args.limit is None and args.sample is None and len(selected) > LARGE_SELECTION_WARNING_THRESHOLD:
            logger.warning(
                "%d tiles selected for the %s with no --limit/--sample. At this scale: the review "
                "map holds every tile's vectorized geometry in one map (can still exhaust memory, "
                "and no browser can usefully render millions of features anyway); the gallery "
                "writes 3 files per tile (can mean hundreds of thousands of files on disk) and "
                "embeds one entry per tile in a single HTML page a browser can't render at very "
                "large counts either. Consider --sample N for a human-reviewable subset, or "
                "build/visualize a --manifest-scoped label set instead if you want a specific "
                "trajectory's coverage rather than every training tile.",
                len(selected), destination,
            )

    if args.map is not None:
        from csnav.viz.ground_truth_view import ground_truth_review_map, save_ground_truth_map

        map_paths = _select(all_paths, args.limit, args.sample)
        logger.info("%d label(s) selected for the map from %s (%d total)", len(map_paths), args.labels_dir, len(all_paths))
        _warn_if_unbounded_and_large("map", map_paths)

        fmap = ground_truth_review_map(_iter_labels(map_paths))
        path = save_ground_truth_map(fmap, args.map)
        logger.info("wrote review map to %s", path)

    if args.gallery_dir is not None:
        from csnav.viz.ground_truth_gallery import build_gallery

        images_dir = args.gallery_dir / "images"
        if not args.overwrite and images_dir.exists() and any(images_dir.iterdir()):
            logger.warning(
                "%s already has rendered images from a previous run. Without --overwrite, any "
                "tile whose 3 output files already exist is left exactly as it was rendered "
                "before - NOT re-rendered from --labels-dir's current data. If --labels-dir was "
                "rebuilt since this gallery directory was last written (a streets re-fetch, a "
                "rasterization fix, anything upstream), this run will silently keep showing the "
                "old images unless you pass --overwrite or point --gallery-dir at a new, empty "
                "directory. See docs/phase2_ground_truth_rasterization.md's \"Refreshing after an "
                "upstream fix\" section.",
                args.gallery_dir,
            )

        # Filter to imagery-backed tiles *before* --limit/--sample commits to a subset - a
        # labels_dir that has outgrown or outlived --imagery-dir's current tile set (rebuilt
        # against a narrower/different pull, tiles left over from an earlier run) must not let
        # --limit N pick N tiles that can never render (see _paths_with_imagery's docstring).
        gallery_candidates = _paths_with_imagery(all_paths, args.imagery_dir)
        if len(gallery_candidates) < len(all_paths):
            logger.warning(
                "%d of %d label(s) under %s have no matching imagery file under %s - most likely "
                "leftover from an earlier build against different/wider imagery. They are excluded "
                "from the gallery's --limit/--sample selection entirely (so a limited run always "
                "picks from tiles that can actually render), but they still count toward "
                "check_ground_truth.py's totals and warnings. Consider a clean rebuild of "
                "--labels-dir if that number is large - see "
                "docs/phase2_ground_truth_rasterization.md's \"Refreshing after an upstream fix\" "
                "section.",
                len(all_paths) - len(gallery_candidates), len(all_paths), args.labels_dir, args.imagery_dir,
            )

        gallery_paths = _select(gallery_candidates, args.limit, args.sample)
        logger.info(
            "%d label(s) selected for the gallery from %s (%d total, %d with matching imagery)",
            len(gallery_paths), args.labels_dir, len(all_paths), len(gallery_candidates),
        )
        _warn_if_unbounded_and_large("gallery", gallery_paths)

        counts = {"matched": 0, "missing": 0}
        index = build_gallery(
            _iter_gallery_pairs(gallery_paths, args.imagery_dir, counts), args.gallery_dir, overwrite=args.overwrite
        )
        if counts["matched"] == 0:
            raise SystemExit(f"no label had matching imagery under {args.imagery_dir}")
        logger.info(
            "wrote gallery for %d tile(s) (%d skipped, missing imagery) to %s",
            counts["matched"], counts["missing"], index,
        )


if __name__ == "__main__":
    main()
