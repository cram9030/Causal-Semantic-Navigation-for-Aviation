"""Rasterized panoptic ground-truth label for one imagery tile.

Phase 2 (`docs/INTEGRATION_PLAN.md` §5): "rasterize ground-truth panoptic
labels using CSJ street geometry/widths over San Jose imagery tiles". A
:class:`PanopticLabel` is the output of that rasterization for a single
imagery tile - a semantic class band plus an instance-id band, pixel-aligned
to that tile's own imagery raster (same width/height/transform/CRS), so a
training loader can pair the two by array index with no further alignment
step.

Two classes only (`PanopticClass`): road and intersection, matching the split
`csnav.trajectory.manifest.ManifestLandmark`/`ManifestIntersection` already
make for the same reason - integration plan §3.4's Mask2Former match step
detects road and intersection instances separately. Everything else is
background.

Storage: one 2-band ``uint32`` GeoTIFF per tile (band 1 semantic class id,
band 2 instance id) in EPSG:4326, plus a JSON sidecar carrying per-instance
metadata (`SegmentInfo`) and provenance - deliberately close to COCO
panoptic's own "id-encoded raster + segments_info" shape, so a later
Mask2Former training script can convert to that format without this module
reimplementing PNG id-packing or RLE encoding itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine

from csnav.data.arcgis.models import Extent
from csnav.trajectory.coverage import TileRef

#: Label schema version, bumped when the on-disk format changes so an older
#: pinned label set isn't silently misread - same convention as
#: `csnav.trajectory.manifest.MANIFEST_SCHEMA_VERSION`.
LABEL_SCHEMA_VERSION = 1


class PanopticClass(IntEnum):
    """Semantic classes rasterized into a label's semantic band."""

    BACKGROUND = 0
    ROAD = 1
    INTERSECTION = 2


@dataclass(frozen=True)
class SegmentInfo:
    """Metadata for one non-zero instance id in a label's instance band.

    ``segment_id`` is the source `csnav.data.arcgis.streets.StreetSegment`'s
    id for a ``ROAD`` instance; ``intersection_segment_ids`` are the segment
    ids meeting there for an ``INTERSECTION`` instance (mirrors
    `csnav.trajectory.manifest.ManifestIntersection.segment_ids`). Exactly one
    of the two is populated, matching ``class_id``.
    """

    instance_id: int
    class_id: int
    segment_id: str | None = None
    intersection_segment_ids: tuple[str, ...] = ()
    name: str | None = None
    width_m: float | None = None
    default_width_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "class_id": self.class_id,
            "segment_id": self.segment_id,
            "intersection_segment_ids": list(self.intersection_segment_ids),
            "name": self.name,
            "width_m": self.width_m,
            "default_width_used": self.default_width_used,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SegmentInfo":
        return cls(
            instance_id=raw["instance_id"],
            class_id=raw["class_id"],
            segment_id=raw.get("segment_id"),
            intersection_segment_ids=tuple(raw.get("intersection_segment_ids", ())),
            name=raw.get("name"),
            width_m=raw.get("width_m"),
            default_width_used=raw.get("default_width_used", False),
        )


def _extent_to_dict(extent: Extent) -> dict[str, Any]:
    return {"xmin": extent.xmin, "ymin": extent.ymin, "xmax": extent.xmax, "ymax": extent.ymax, "wkid": extent.wkid}


def _extent_from_dict(raw: dict[str, Any]) -> Extent:
    return Extent(xmin=raw["xmin"], ymin=raw["ymin"], xmax=raw["xmax"], ymax=raw["ymax"], wkid=raw.get("wkid", 4326))


def _tile_stem(tile: TileRef) -> str:
    """Filename stem for one tile - matches `scripts/fetch_historic_imagery.py`'s own
    ``{level}_{row}_{col}`` naming, so a label and its source imagery share a basename."""
    return f"{tile.level}_{tile.row}_{tile.col}"


@dataclass
class PanopticLabel:
    """Rasterized ground truth for one imagery tile, aligned to its pixel grid.

    ``semantic``/``instance`` are 2D ``uint32`` arrays of identical shape.
    ``instance`` is 0 wherever ``semantic`` is `PanopticClass.BACKGROUND`, and
    every other value has exactly one corresponding entry in ``segments``.
    ``transform``/``crs``/the array shape must match the tile's source
    imagery raster exactly - this module never derives them independently.
    """

    tile: TileRef
    semantic: np.ndarray
    instance: np.ndarray
    transform: Affine
    crs: str
    segments: tuple[SegmentInfo, ...]
    streets_source: str | None = None
    imagery_source: str | None = None

    def __post_init__(self) -> None:
        if self.semantic.shape != self.instance.shape:
            raise ValueError(
                f"semantic/instance shape mismatch: {self.semantic.shape} vs {self.instance.shape}"
            )

    @property
    def stem(self) -> str:
        return _tile_stem(self.tile)

    def paths(self, output_dir: str | Path) -> tuple[Path, Path]:
        """The ``(raster_path, sidecar_path)`` :meth:`save`/:meth:`load` use for this tile."""
        base = Path(output_dir) / self.stem
        return base.with_suffix(".tif"), base.with_suffix(".json")

    def save(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Write the label as a 2-band GeoTIFF plus a JSON sidecar, creating ``output_dir``."""
        raster_path, sidecar_path = self.paths(output_dir)
        raster_path.parent.mkdir(parents=True, exist_ok=True)

        data = np.stack([self.semantic.astype(np.uint32), self.instance.astype(np.uint32)])
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=2,
            dtype=data.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as dst:
            dst.write(data)
            dst.set_band_description(1, "semantic_class_id")
            dst.set_band_description(2, "instance_id")

        payload = {
            "schema_version": LABEL_SCHEMA_VERSION,
            "tile": {
                "level": self.tile.level,
                "row": self.tile.row,
                "col": self.tile.col,
                "bounds": _extent_to_dict(self.tile.bounds),
            },
            "classes": {klass.name.lower(): int(klass) for klass in PanopticClass},
            "streets_source": self.streets_source,
            "imagery_source": self.imagery_source,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return raster_path, sidecar_path

    @classmethod
    def load(cls, raster_path: str | Path, sidecar_path: str | Path | None = None) -> "PanopticLabel":
        """Read a label previously written by :meth:`save`.

        ``sidecar_path`` defaults to ``raster_path`` with its suffix swapped
        to ``.json``, matching what :meth:`save` writes alongside it.
        """
        raster_path = Path(raster_path)
        sidecar_path = Path(sidecar_path) if sidecar_path is not None else raster_path.with_suffix(".json")
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        version = raw.get("schema_version", LABEL_SCHEMA_VERSION)
        if version != LABEL_SCHEMA_VERSION:
            raise ValueError(
                f"label schema version {version} != {LABEL_SCHEMA_VERSION}; rebuild the label rather "
                "than reading it with a mismatched reader"
            )

        with rasterio.open(raster_path) as src:
            data = src.read()
            crs = str(src.crs)
            transform = src.transform

        tile_raw = raw["tile"]
        tile = TileRef(
            level=tile_raw["level"],
            row=tile_raw["row"],
            col=tile_raw["col"],
            bounds=_extent_from_dict(tile_raw["bounds"]),
        )
        return cls(
            tile=tile,
            semantic=data[0],
            instance=data[1],
            transform=transform,
            crs=crs,
            segments=tuple(SegmentInfo.from_dict(item) for item in raw["segments"]),
            streets_source=raw.get("streets_source"),
            imagery_source=raw.get("imagery_source"),
        )
