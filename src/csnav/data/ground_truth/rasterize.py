"""Rasterize CSJ street geometry into panoptic ground-truth labels, per imagery tile.

Implements the `GroundTruthBuilder.rasterize(streets, tile) -> PanopticLabel`
role sketched in `docs/INTEGRATION_PLAN.md` §7's UML diagram. Two deliberate
departures from that sketch, both documented in
`docs/phase2_ground_truth_rasterization.md`:

1. **No live `ArcGISTileClient` call.** The imagery pixel grid (width, height,
   affine transform) this module rasterizes onto is supplied by the caller,
   read from an already-fetched, already-reprojected imagery GeoTIFF (the
   output of `scripts/fetch_historic_imagery.py`) - this keeps rasterization
   itself a pure function of geometry (testable without a real raster file or
   network access) and guarantees pixel-for-pixel alignment with whatever
   imagery a training loader actually reads.
2. **Streets come from an in-memory `StreetSegment` list**, not a live
   `CSJStreetsClient` call - the same "pin to an archived pull, don't re-query
   the weekly-refreshed live layer" reasoning `StaticStreetsSource` already
   applies in `csnav.trajectory.manifest_builder`.

Everything metric (buffering a centerline by its roadway width, sizing a
derived intersection) happens in a `LocalFrame` anchored at each tile's own
center - tiles are small enough (tens to a couple hundred meters across) that
one anchor per tile introduces no meaningful ENU projection error, and it
means this module never needs a trajectory to reuse `csnav.geometry.shapes`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiLineString, Point as ShapelyPoint, box
from shapely.strtree import STRtree

from csnav.data.arcgis.streets import StreetSegment, segment_geometry, street_name, street_width_m
from csnav.data.ground_truth.labels import PanopticClass, PanopticLabel, SegmentInfo
from csnav.geometry import shapes
from csnav.geometry.local_frame import LocalFrame
from csnav.trajectory.coverage import TileRef

logger = logging.getLogger(__name__)

#: Fallback roadway width (meters) applied when a street segment carries none
#: of `csnav.data.arcgis.streets.WIDTH_FIELD_CANDIDATES` - CSJ's width
#: attribute is not published for every segment. A configurable input
#: (`params.yaml`'s ``ground_truth.default_width_m``), not a constant baked
#: into a call site - see `docs/phase2_ground_truth_rasterization.md` for the
#: one-lane-each-way rationale.
DEFAULT_WIDTH_M = 6.0

#: Radius (meters) a derived intersection is rasterized as. Distinct from
#: `csnav.trajectory.manifest_builder.DEFAULT_INTERSECTION_SNAP`, which is a
#: clustering tolerance for deciding two junction points are "the same
#: intersection", not a rendered size.
DEFAULT_INTERSECTION_RADIUS_M = 3.0

#: Clustering tolerance (meters) for merging nearby computed junction points -
#: mirrors `csnav.trajectory.manifest_builder.DEFAULT_INTERSECTION_SNAP`.
DEFAULT_INTERSECTION_SNAP_M = 2.0

_LineGeom = LineString | MultiLineString


def _segment_id(segment: StreetSegment, fallback_index: int) -> str:
    if segment.object_id is not None:
        return str(segment.object_id)
    return f"seg-{fallback_index}"


@dataclass(frozen=True)
class _RoadMeta:
    segment_id: str
    name: str | None
    width_m: float
    default_width_used: bool


def _add_to_cluster(
    clusters: list[tuple[list[float], list[float], set[str]]],
    east: float,
    north: float,
    ids: set[str],
    snap: float,
) -> None:
    """Merge an ENU junction point into an existing cluster within ``snap`` meters, or start one.

    Same clustering rule as `csnav.trajectory.manifest_builder._add_to_cluster`,
    duplicated here rather than imported: that one operates over a
    trajectory's `LocalFrame` and windowed candidate roads, this one over a
    single tile's clipped centerlines - close enough in shape, different
    enough in context, that sharing the one function wasn't worth coupling
    the two modules over.
    """
    for easts, norths, cluster_ids in clusters:
        centre_east = sum(easts) / len(easts)
        centre_north = sum(norths) / len(norths)
        if (east - centre_east) ** 2 + (north - centre_north) ** 2 <= snap**2:
            easts.append(east)
            norths.append(north)
            cluster_ids.update(ids)
            return
    clusters.append(([east], [north], set(ids)))


def _derive_intersections(
    lines_enu: Sequence[_LineGeom], segment_ids: Sequence[str], snap: float
) -> list[tuple[float, float, tuple[str, ...]]]:
    """Junction points between clipped centerlines, snapped and de-duplicated, in ENU meters."""
    if len(lines_enu) < 2:
        return []

    tree = STRtree(list(lines_enu))
    clusters: list[tuple[list[float], list[float], set[str]]] = []
    for index, geometry in enumerate(lines_enu):
        for other_index in tree.query(geometry):
            other_index = int(other_index)
            if other_index <= index:
                continue
            meeting = geometry.intersection(lines_enu[other_index])
            for point in shapes.point_parts(meeting):
                ids = {segment_ids[index], segment_ids[other_index]}
                _add_to_cluster(clusters, point.x, point.y, ids, snap)

    return [
        (sum(easts) / len(easts), sum(norths) / len(norths), tuple(sorted(ids)))
        for easts, norths, ids in clusters
    ]


@dataclass
class GroundTruthBuilder:
    """Rasterizes CSJ street geometry into a `PanopticLabel`, one imagery tile at a time.

    ``default_width_m``/``intersection_radius_m``/``intersection_snap_m`` are
    swept/versioned inputs (CLAUDE.md's "config lives in versioned config
    files" convention) - callers read them from `params.yaml`'s
    ``ground_truth`` section rather than relying on the defaults here.
    """

    default_width_m: float = DEFAULT_WIDTH_M
    intersection_radius_m: float = DEFAULT_INTERSECTION_RADIUS_M
    intersection_snap_m: float = DEFAULT_INTERSECTION_SNAP_M

    def __post_init__(self) -> None:
        if self.default_width_m <= 0.0:
            raise ValueError(f"default_width_m must be > 0, got {self.default_width_m}")
        if self.intersection_radius_m <= 0.0:
            raise ValueError(f"intersection_radius_m must be > 0, got {self.intersection_radius_m}")
        if self.intersection_snap_m <= 0.0:
            raise ValueError(f"intersection_snap_m must be > 0, got {self.intersection_snap_m}")

    def rasterize(
        self,
        streets: Sequence[StreetSegment],
        tile: TileRef,
        width: int,
        height: int,
        transform: Affine,
        streets_source: str | None = None,
        imagery_source: str | None = None,
    ) -> PanopticLabel:
        """Rasterize ``streets`` onto ``tile``'s ``(width, height)`` pixel grid.

        ``transform`` is the affine WGS84-degrees-to-pixel mapping for that
        grid - callers read it straight off the tile's source imagery
        GeoTIFF, so the returned label lines up with it exactly.
        ``streets`` need not already be clipped to the tile: only the portion
        intersecting ``tile.bounds`` is used. Returns an all-background label
        (valid, not an error) when nothing intersects.
        """
        bounds = tile.bounds
        tile_box = box(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)
        frame = LocalFrame(
            origin_lat=(bounds.ymin + bounds.ymax) / 2.0, origin_lon=(bounds.xmin + bounds.xmax) / 2.0
        )

        clipped_enu: list[_LineGeom] = []
        road_meta: list[_RoadMeta] = []
        for index, segment in enumerate(streets):
            clipped = segment_geometry(segment).intersection(tile_box)
            parts = shapes.line_parts(clipped)
            if not parts:
                continue
            clipped_line: _LineGeom = (
                LineString(parts[0]) if len(parts) == 1 else MultiLineString([list(p) for p in parts])
            )
            clipped_enu.append(shapes.to_enu(clipped_line, frame))

            raw_width = street_width_m(segment.attributes)
            default_used = raw_width is None or raw_width <= 0.0
            road_meta.append(
                _RoadMeta(
                    segment_id=_segment_id(segment, index),
                    name=street_name(segment.attributes),
                    width_m=self.default_width_m if default_used else raw_width,
                    default_width_used=default_used,
                )
            )

        segments: list[SegmentInfo] = []
        burn_semantic: list[tuple[Any, int]] = []
        burn_instance: list[tuple[Any, int]] = []
        next_id = 1

        for index in sorted(range(len(clipped_enu)), key=lambda i: road_meta[i].segment_id):
            meta = road_meta[index]
            buffered_wgs84 = shapes.to_wgs84(clipped_enu[index].buffer(meta.width_m / 2.0), frame)
            polygons = shapes.polygon_parts(buffered_wgs84)
            if not polygons:
                continue
            instance_id = next_id
            next_id += 1
            for polygon in polygons:
                burn_semantic.append((polygon, int(PanopticClass.ROAD)))
                burn_instance.append((polygon, instance_id))
            segments.append(
                SegmentInfo(
                    instance_id=instance_id,
                    class_id=int(PanopticClass.ROAD),
                    segment_id=meta.segment_id,
                    name=meta.name,
                    width_m=meta.width_m,
                    default_width_used=meta.default_width_used,
                )
            )

        junctions = _derive_intersections(
            clipped_enu, [meta.segment_id for meta in road_meta], self.intersection_snap_m
        )
        for east, north, ids in junctions:
            buffered_wgs84 = shapes.to_wgs84(ShapelyPoint(east, north).buffer(self.intersection_radius_m), frame)
            polygons = shapes.polygon_parts(buffered_wgs84)
            if not polygons:
                continue
            instance_id = next_id
            next_id += 1
            for polygon in polygons:
                burn_semantic.append((polygon, int(PanopticClass.INTERSECTION)))
                burn_instance.append((polygon, instance_id))
            segments.append(
                SegmentInfo(
                    instance_id=instance_id,
                    class_id=int(PanopticClass.INTERSECTION),
                    intersection_segment_ids=ids,
                )
            )

        out_shape = (height, width)
        if burn_semantic:
            semantic = rio_rasterize(
                burn_semantic, out_shape=out_shape, transform=transform, fill=int(PanopticClass.BACKGROUND),
                dtype=np.uint32,
            )
            instance = rio_rasterize(burn_instance, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint32)
        else:
            semantic = np.full(out_shape, int(PanopticClass.BACKGROUND), dtype=np.uint32)
            instance = np.zeros(out_shape, dtype=np.uint32)

        return PanopticLabel(
            tile=tile,
            semantic=semantic,
            instance=instance,
            transform=transform,
            crs="EPSG:4326",
            segments=tuple(segments),
            streets_source=streets_source,
            imagery_source=imagery_source,
        )
