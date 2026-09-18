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

**A road is buffered before it is clipped to the tile, not after.** Buffering
a centerline that has already been cut down to the tile's own box leaves a
rounded end cap sitting exactly on the cut - wherever a street crosses a tile
edge, or CSJ splits one physical street into a fresh ``OBJECTID`` at a
cross-street, the road looks like it pinches into a semicircle rather than
continuing straight through. The fix is the other order: clip each
candidate's centerline to a small margin around the tile (``BUFFER_PAD_M`` -
enough to bound the `LocalFrame` projection error and keep any *genuine* cut
end safely outside the tile, not enough to matter for realistic road widths
or CSJ's own block spacing), merge it with any other candidate that is really
the same street continuing through (same CSJ master-street id, same width,
same name - see `_chain_runs`), buffer *that* with flat end caps, and only
then intersect the resulting polygon with the tile's exact box. A tile-edge
crossing comes out as a straight cut flush with the tile boundary (matching
whatever the neighboring tile draws on its side), and an in-tile CSJ split
that is really the same street continuing comes out as one unbroken ribbon
instead of two capped stubs.

**A derived intersection's radius scales with the roads meeting there.** A
fixed radius for every intersection (the previous behavior) visibly pinches
a wide arterial down to a circle narrower than its own pavement wherever it
crosses another road. `_intersection_radius` floors the radius at half the
widest intersecting road's rasterized width, so the intersection is always at
least as wide as the road it's part of; `intersection_radius_m` remains the
floor for two narrow streets crossing, exactly as before.
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

from csnav.data.arcgis.streets import StreetSegment, segment_geometry, street_master_id, street_name, street_width_m
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

#: Floor (meters) a derived intersection's radius is rasterized as, when no
#: intersecting road is wide enough to demand a bigger one -
#: `_intersection_radius` below never lets the *actual* radius fall below
#: half the widest intersecting road, since a circle narrower than that would
#: visibly pinch the road as it enters the intersection. Distinct from
#: `csnav.trajectory.manifest_builder.DEFAULT_INTERSECTION_SNAP`, which is a
#: clustering tolerance for deciding two junction points are "the same
#: intersection", not a rendered size.
DEFAULT_INTERSECTION_RADIUS_M = 3.0

#: Clustering tolerance (meters) for merging nearby computed junction points -
#: mirrors `csnav.trajectory.manifest_builder.DEFAULT_INTERSECTION_SNAP`. Also
#: reused as the endpoint-adjacency tolerance for chaining contiguous same-
#: street segments together (see `_chain_runs`): both are "are these two
#: points really the same point" checks over the same kind of CSJ-digitized
#: geometry, so a second constant for the same notion of closeness would be
#: redundant rather than more precise.
DEFAULT_INTERSECTION_SNAP_M = 2.0

#: Margin (meters) a road's centerline is clipped to *before* buffering,
#: beyond the tile it is being rasterized for - see "buffer the padded
#: centerline, then clip the polygon to the tile" in this module's docstring.
#: Two independent reasons this needs to be more than zero but can stay
#: small: (1) `LocalFrame`'s flat-earth ENU approximation is only accurate
#: near its anchor (this tile's own center) - buffering a centerline that
#: reaches far beyond the tile in that frame would reintroduce the same kind
#: of geometric error the per-tile anchor was chosen to avoid; (2) wherever a
#: road's rasterized buffer is truncated for being computationally bounded
#: rather than because the road actually ends there, that flat cut needs to
#: land safely outside the tile so the final tile-box clip never exposes it.
#: 50 m clears the widest realistic CSJ roadway (a freeway's `FOCWIDTH` tops
#: out well under 40 m, so even doubling that for margin fits) with room to
#: spare against typical San Jose block lengths (CSJ's own local-street grid
#: runs roughly 80-150 m intersection-to-intersection), so a genuine chain
#: adjacency one block over is still found.
BUFFER_PAD_M = 50.0

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
    attributes: dict[str, Any]
    master_id: str | None


def _endpoints(line: LineString) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = list(line.coords)
    return coords[0], coords[-1]


def _reversed_line(line: LineString) -> LineString:
    return LineString(list(line.coords)[::-1])


def _close(a: tuple[float, float], b: tuple[float, float], tolerance: float) -> bool:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= tolerance**2


def _chain_runs(
    items: Sequence[tuple[int, LineString]], snap: float
) -> list[tuple[list[int], LineString]]:
    """Group same-identity centerlines into maximal end-to-end chains, in ENU meters.

    ``items`` already share one merge key (same logical street, same width,
    same default-width status, same name - see :meth:`GroundTruthBuilder.rasterize`)
    - this only decides *adjacency*: whether two of them actually meet
    end-to-end closely enough (within ``snap`` meters) to rasterize as one
    continuous ribbon instead of leaving a capped seam at their shared point.

    Greedy nearest-endpoint extension, in no particular order: a branch under
    one identity key (rare - e.g. a street that forks while keeping the same
    master id) is not chased past the first extension found in either
    direction, so the un-chained remainder comes back as its own run rather
    than being merged incorrectly. Returns one ``(member_indices, merged_line)``
    tuple per chain, including length-1 chains for anything that didn't
    adjoin another member.
    """
    remaining = list(items)
    runs: list[tuple[list[int], LineString]] = []
    while remaining:
        index, line = remaining.pop()
        run_indices = [index]
        run_line = line
        extended = True
        while extended and remaining:
            extended = False
            start, end = _endpoints(run_line)
            for position, (other_index, other_line) in enumerate(remaining):
                other_start, other_end = _endpoints(other_line)
                if _close(end, other_start, snap):
                    run_line = LineString(list(run_line.coords) + list(other_line.coords)[1:])
                elif _close(end, other_end, snap):
                    run_line = LineString(list(run_line.coords) + list(_reversed_line(other_line).coords)[1:])
                elif _close(start, other_end, snap):
                    run_line = LineString(list(other_line.coords) + list(run_line.coords)[1:])
                elif _close(start, other_start, snap):
                    run_line = LineString(list(_reversed_line(other_line).coords) + list(run_line.coords)[1:])
                else:
                    continue
                run_indices.append(other_index)
                remaining.pop(position)
                extended = True
                break
        runs.append((run_indices, run_line))
    return runs


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


def _intersection_radius(ids: Sequence[str], width_by_id: dict[str, float], floor: float) -> float:
    """The radius (meters) to rasterize one derived intersection at.

    Never smaller than ``floor`` (``intersection_radius_m``), and never
    smaller than half the widest road meeting there - a circle narrower than
    that would visibly pinch that road's own pavement down to fit the
    intersection instead of continuing through it at full width.
    """
    widest = max((width_by_id.get(segment_id, 0.0) for segment_id in ids), default=0.0)
    return max(floor, widest / 2.0)


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
        tile_box_enu = shapes.to_enu(tile_box, frame)
        # A slightly larger box than the tile itself to clip each candidate's
        # centerline to before buffering - see this module's docstring for
        # why buffering happens against this padded box, not the tile box.
        pad_box_enu = tile_box_enu.buffer(BUFFER_PAD_M, cap_style="square", join_style="mitre")

        clipped_enu: list[_LineGeom] = []
        padded_enu: list[_LineGeom] = []
        road_meta: list[_RoadMeta] = []
        for index, segment in enumerate(streets):
            geometry = segment_geometry(segment)
            clipped = geometry.intersection(tile_box)
            parts = shapes.line_parts(clipped)
            if not parts:
                continue
            clipped_line: _LineGeom = (
                LineString(parts[0]) if len(parts) == 1 else MultiLineString([list(p) for p in parts])
            )
            clipped_enu.append(shapes.to_enu(clipped_line, frame))

            geometry_enu = shapes.to_enu(geometry, frame)
            padded_parts = shapes.line_parts(geometry_enu.intersection(pad_box_enu))
            if not padded_parts:
                # Should not happen - the tile-clipped line above is already
                # non-empty, and the padded box strictly contains the tile
                # box - but fall back to the tile-clipped line itself rather
                # than risk an IndexError over a boundary floating-point
                # edge case.
                padded_enu.append(clipped_enu[-1])
            else:
                padded_enu.append(
                    LineString(padded_parts[0]) if len(padded_parts) == 1
                    else MultiLineString([list(p) for p in padded_parts])
                )

            raw_width = street_width_m(segment.attributes)
            default_used = raw_width is None or raw_width <= 0.0
            road_meta.append(
                _RoadMeta(
                    segment_id=_segment_id(segment, index),
                    name=street_name(segment.attributes),
                    width_m=self.default_width_m if default_used else raw_width,
                    default_width_used=default_used,
                    attributes=dict(segment.attributes),
                    master_id=street_master_id(segment.attributes),
                )
            )

        segments: list[SegmentInfo] = []
        burn_semantic: list[tuple[Any, int]] = []
        burn_instance: list[tuple[Any, int]] = []
        next_id = 1

        # Group same-identity, same-width candidates so a CSJ split doesn't
        # rasterize as two capped stubs where the street plainly continues -
        # see `_chain_runs`. A candidate with no master-street id, or whose
        # padded centerline isn't a single LineString (rare - only where the
        # padded box clips it into disjoint pieces), is rasterized on its own.
        by_key: dict[tuple[Any, ...], list[tuple[int, LineString]]] = {}
        runs: list[tuple[list[int], _LineGeom]] = []
        for local_index, meta in enumerate(road_meta):
            geometry = padded_enu[local_index]
            if meta.master_id is None or not isinstance(geometry, LineString):
                runs.append(([local_index], geometry))
                continue
            key = (meta.master_id, meta.width_m, meta.default_width_used, meta.name)
            by_key.setdefault(key, []).append((local_index, geometry))
        for items in by_key.values():
            runs.extend(_chain_runs(items, self.intersection_snap_m))
        runs.sort(key=lambda run: min(road_meta[i].segment_id for i in run[0]))

        for member_indices, merged_line in runs:
            member_indices = sorted(member_indices, key=lambda i: road_meta[i].segment_id)
            meta = road_meta[member_indices[0]]
            buffered_enu = merged_line.buffer(meta.width_m / 2.0, cap_style="flat").intersection(tile_box_enu)
            buffered_wgs84 = shapes.to_wgs84(buffered_enu, frame)
            polygons = shapes.polygon_parts(buffered_wgs84)
            if not polygons:
                continue
            instance_id = next_id
            next_id += 1
            for polygon in polygons:
                burn_semantic.append((polygon, int(PanopticClass.ROAD)))
                burn_instance.append((polygon, instance_id))
            member_ids = tuple(road_meta[i].segment_id for i in member_indices)
            segments.append(
                SegmentInfo(
                    instance_id=instance_id,
                    class_id=int(PanopticClass.ROAD),
                    segment_id=member_ids[0],
                    segment_ids=member_ids,
                    name=meta.name,
                    width_m=meta.width_m,
                    default_width_used=meta.default_width_used,
                    attributes=meta.attributes,
                )
            )

        width_by_id = {meta.segment_id: meta.width_m for meta in road_meta}
        # Which merged road run each original CSJ segment id ended up in -
        # used below to drop a "junction" that _derive_intersections finds
        # between two CSJ rows that were merged into the very same road
        # instance. Without this, every merged split point (the whole reason
        # for merging in the first place) would still stamp a spurious
        # intersection on top of itself, since junction detection runs on
        # the original per-segment lines and has no notion of the merge.
        run_by_segment_id = {
            road_meta[member_index].segment_id: run_number
            for run_number, (member_indices, _) in enumerate(runs)
            for member_index in member_indices
        }
        junctions = _derive_intersections(
            clipped_enu, [meta.segment_id for meta in road_meta], self.intersection_snap_m
        )
        for east, north, ids in junctions:
            if len({run_by_segment_id.get(segment_id) for segment_id in ids}) <= 1:
                continue  # every road meeting here was merged into one instance - not a real junction
            radius = _intersection_radius(ids, width_by_id, self.intersection_radius_m)
            buffered_wgs84 = shapes.to_wgs84(ShapelyPoint(east, north).buffer(radius), frame)
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
