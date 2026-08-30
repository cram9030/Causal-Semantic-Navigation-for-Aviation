"""Automated sanity checks over rasterized panoptic labels.

Complements the visual QA tooling in `csnav.viz.ground_truth_gallery` /
`csnav.viz.ground_truth_view`: those are for a person paging through tiles by
eye, this is for catching systematic problems (a bad buffer, a shape
mismatch, a schema drift) across a whole label set without looking at any of
them - the kind of check that should run in CI or right after
`scripts/build_ground_truth.py`, not only when someone remembers to look.

Checks are deliberately structural/statistical, not "does this look like a
road": that judgment call is what the gallery is for. What this module can
say without a human: does the raster actually match its own sidecar, is
every non-background pixel accounted for by a segment, and how much of the
label set is leaning on ``default_width_m`` rather than a real CSJ width
(worth knowing, since Phase 2's confusion-matrix noise priors are only as
good as the geometry Mask2Former is being fit against).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csnav.data.ground_truth.labels import PanopticClass, PanopticLabel


@dataclass(frozen=True)
class CheckIssue:
    """One finding from :func:`check_label`. ``severity="error"`` fails the tile; ``"warning"`` does not."""

    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "message": self.message}


@dataclass(frozen=True)
class TileCheckReport:
    """Everything :func:`check_label` found for one tile."""

    tile_key: str
    issues: tuple[CheckIssue, ...]
    road_pixel_fraction: float
    intersection_pixel_fraction: float
    default_width_fraction: float

    @property
    def ok(self) -> bool:
        """No ``"error"``-severity issue - warnings alone don't fail a tile."""
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "tile_key": self.tile_key,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "road_pixel_fraction": self.road_pixel_fraction,
            "intersection_pixel_fraction": self.intersection_pixel_fraction,
            "default_width_fraction": self.default_width_fraction,
        }


#: Fraction of a label set's segments using the fallback width above which
#: `check_label` warns that CSJ's own width attribute is thin for this area -
#: a data-quality signal worth surfacing, not a hard failure.
DEFAULT_WIDTH_WARN_FRACTION = 0.5


def check_label(label: PanopticLabel) -> TileCheckReport:
    """Structural/statistical checks for one label, independent of any other tile."""
    issues: list[CheckIssue] = []
    tile_key = f"{label.tile.level}/{label.tile.row}/{label.tile.col}"

    if label.semantic.shape != label.instance.shape:
        issues.append(
            CheckIssue("error", f"semantic/instance shape mismatch: {label.semantic.shape} vs {label.instance.shape}")
        )

    known_ids = {segment.instance_id for segment in label.segments}
    present_ids = {int(value) for value in np.unique(label.instance) if value != 0}
    missing = sorted(present_ids - known_ids)
    if missing:
        issues.append(CheckIssue("error", f"instance ids rasterized with no segments_info entry: {missing}"))
    unused = sorted(known_ids - present_ids)
    if unused:
        issues.append(CheckIssue("warning", f"segments_info entries never rasterized (occluded by later draws): {unused}"))

    duplicate_ids = len(label.segments) - len(known_ids)
    if duplicate_ids > 0:
        issues.append(CheckIssue("error", f"{duplicate_ids} duplicate instance_id value(s) in segments_info"))

    background_with_instance = int(np.count_nonzero((label.semantic == PanopticClass.BACKGROUND) & (label.instance != 0)))
    if background_with_instance:
        issues.append(
            CheckIssue("error", f"{background_with_instance} background pixel(s) carry a nonzero instance id")
        )
    foreground_without_instance = int(
        np.count_nonzero((label.semantic != PanopticClass.BACKGROUND) & (label.instance == 0))
    )
    if foreground_without_instance:
        issues.append(
            CheckIssue("error", f"{foreground_without_instance} non-background pixel(s) carry instance id 0")
        )

    unknown_classes = sorted(set(np.unique(label.semantic).tolist()) - {int(c) for c in PanopticClass})
    if unknown_classes:
        issues.append(CheckIssue("error", f"semantic band contains unrecognized class id(s): {unknown_classes}"))

    total_pixels = label.semantic.size
    road_fraction = float(np.count_nonzero(label.semantic == PanopticClass.ROAD)) / total_pixels
    intersection_fraction = float(np.count_nonzero(label.semantic == PanopticClass.INTERSECTION)) / total_pixels

    default_used = sum(1 for segment in label.segments if segment.default_width_used)
    default_fraction = default_used / len(label.segments) if label.segments else 0.0
    if default_fraction > DEFAULT_WIDTH_WARN_FRACTION:
        issues.append(
            CheckIssue(
                "warning",
                f"{default_fraction:.0%} of segments in this tile fell back to the default width "
                "(CSJ width attribute mostly missing here)",
            )
        )

    return TileCheckReport(
        tile_key=tile_key,
        issues=tuple(issues),
        road_pixel_fraction=road_fraction,
        intersection_pixel_fraction=intersection_fraction,
        default_width_fraction=default_fraction,
    )


@dataclass(frozen=True)
class LabelSetReport:
    """Aggregate result of running :func:`check_label` over every label under a directory."""

    tiles: tuple[TileCheckReport, ...]

    @property
    def ok(self) -> bool:
        return all(tile.ok for tile in self.tiles)

    @property
    def error_count(self) -> int:
        return sum(1 for tile in self.tiles for issue in tile.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for tile in self.tiles for issue in tile.issues if issue.severity == "warning")

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tile_count": len(self.tiles),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination


def check_label_directory(labels_dir: str | Path) -> LabelSetReport:
    """Run :func:`check_label` over every ``*.json`` sidecar under ``labels_dir``.

    Sorted by filename stem so a report's tile order is stable/diffable
    across reruns, matching `csnav.trajectory.manifest.ManifestBundle`'s own
    "reviewable, diffable artifact" convention for pinned output.
    """
    labels_dir = Path(labels_dir)
    reports = []
    for sidecar in sorted(labels_dir.glob("*.json")):
        raster_path = sidecar.with_suffix(".tif")
        if not raster_path.exists():
            reports.append(
                TileCheckReport(
                    tile_key=sidecar.stem,
                    issues=(CheckIssue("error", f"missing raster for sidecar {sidecar.name}"),),
                    road_pixel_fraction=0.0,
                    intersection_pixel_fraction=0.0,
                    default_width_fraction=0.0,
                )
            )
            continue
        label = PanopticLabel.load(raster_path, sidecar)
        reports.append(check_label(label))
    return LabelSetReport(tiles=tuple(reports))
