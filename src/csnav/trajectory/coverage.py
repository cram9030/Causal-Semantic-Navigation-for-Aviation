"""What ground a trajectory window can see, and which imagery tiles cover it.

Two things sit between the tube model and the manifest builder:

1. **The visible footprint** - the tube corridor for a window, grown by the
   ground radius of the sensor field of view at that window's height above
   ground. Integration plan §3.3 scopes a manifest to "the roads/intersections
   that could possibly be visible from *any* state within the tube at that
   window", which is exactly this shape, not the bare tube.
2. **The tile set in view** - which cached imagery tiles that footprint
   covers. Those are the tiles the ground-truth builder needs panoptic labels
   for (§4, "manifests are consumed both by the ground-truth builder ... and
   by the runtime Possible roads node"), and they are what the Phase 1
   visualization draws.

**Frames.** Footprints are WGS84 (EPSG:4326) shapely polygons in (lon, lat)
order. Metric growth happens inside :mod:`csnav.trajectory.tube`, in the
trajectory's local ENU frame. Tile math runs in the tile scheme's own spatial
reference (EPSG:3857 for the San Jose imagery caches) and is converted back to
EPSG:4326 for storage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from csnav.data.arcgis.models import Extent, TileInfo
from csnav.data.arcgis.projections import extent_3857_to_4326, extent_4326_to_3857
from csnav.data.arcgis.tiles import tile_bounds, tile_row_col_range
from csnav.geometry.fov import FieldOfView
from csnav.trajectory.trajectory import Trajectory, TrajectoryWindow
from csnav.trajectory.tube import TubeModel, polygon_bounds
from csnav.trajectory.waypoints import Waypoint

#: Signature of an AGL lookup: given a waypoint (WGS84 degrees, height in
#: meters above the WGS84 ellipsoid), return height above *ground* in meters.
AglProvider = Callable[[Waypoint], float]


def height_as_agl(waypoint: Waypoint) -> float:
    """Default :data:`AglProvider`: treat a waypoint's height as already AGL.

    Correct only where ground elevation is ~0 or where the flight plan was
    authored in AGL to begin with. For real San Jose terrain, pass
    :func:`agl_from_elevation` wrapping
    :class:`csnav.data.lidar.LidarElevationClient` instead - integration plan
    §2 is explicit that GPS-derived height is ellipsoidal/MSL and needs a
    ground-elevation correction before it means AGL.
    """
    return max(waypoint.height, 0.0)


def agl_from_elevation(elevation_at: Callable[[float, float], float | None]) -> AglProvider:
    """Build an :data:`AglProvider` from a ``(lon, lat) -> ground elevation`` lookup.

    ``elevation_at`` returns ground elevation in meters (e.g.
    :meth:`csnav.data.lidar.LidarElevationClient.identify`), or ``None`` where
    the elevation source has no data - in which case the waypoint's height is
    used unchanged, matching :func:`height_as_agl`. Result is clamped at 0 m:
    a negative AGL would otherwise produce a negative FOV ground radius.
    """

    def provider(waypoint: Waypoint) -> float:
        ground = elevation_at(waypoint.lon, waypoint.lat)
        if ground is None:
            return max(waypoint.height, 0.0)
        return max(waypoint.height - ground, 0.0)

    return provider


@dataclass(frozen=True)
class TileRef:
    """One imagery tile in a service's own tile scheme, with its WGS84 bounds.

    ``level``/``row``/``col`` address the tile exactly as
    :meth:`csnav.data.arcgis.client.ArcGISTileClient.fetch_tile` expects them.
    ``bounds`` is the tile's footprint in EPSG:4326 - stored in WGS84 per
    CLAUDE.md even though the tile grid itself is defined in EPSG:3857.
    """

    level: int
    row: int
    col: int
    bounds: Extent

    @property
    def key(self) -> str:
        """``level/row/col`` - the stable identifier used in manifests and filenames."""
        return f"{self.level}/{self.row}/{self.col}"

    def to_geojson_feature(self) -> dict:
        """GeoJSON ``Polygon`` feature of the tile footprint, in EPSG:4326."""
        b = self.bounds
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [b.xmin, b.ymin],
                        [b.xmax, b.ymin],
                        [b.xmax, b.ymax],
                        [b.xmin, b.ymax],
                        [b.xmin, b.ymin],
                    ]
                ],
            },
            "properties": {"level": self.level, "row": self.row, "col": self.col, "key": self.key},
        }


def max_agl(trajectory: Trajectory, window: TrajectoryWindow, agl_provider: AglProvider, samples: int = 8) -> float:
    """Largest height above ground, in meters, sampled across a window's span.

    The *maximum* is used deliberately: the FOV ground radius grows with
    height, so sizing a window's search footprint from its highest point keeps
    the manifest a superset of what could actually be seen anywhere in the
    window. ``samples`` is the number of evenly spaced arc-length samples
    (plus both endpoints).
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    span = window.end_distance - window.start_distance
    points = [trajectory.point_at(window.start_distance + span * step / samples) for step in range(samples + 1)]
    return max(agl_provider(point) for point in points)


def visible_footprint(
    trajectory: Trajectory,
    tube: TubeModel,
    window: TrajectoryWindow | None = None,
    field_of_view: FieldOfView | None = None,
    agl_provider: AglProvider = height_as_agl,
) -> Polygon:
    """Ground area that could be seen from anywhere inside the tube, in WGS84 (lon, lat).

    The tube corridor grown radially by the FOV ground radius at the window's
    maximum AGL. With ``field_of_view=None`` this degenerates to the bare tube
    corridor - useful for showing containment alone. ``window=None`` covers the
    whole trajectory (its maximum AGL over all waypoints).
    """
    if field_of_view is None:
        return tube.corridor(trajectory, window=window)

    if window is None:
        agl = max(agl_provider(waypoint) for waypoint in trajectory.waypoints)
    else:
        agl = max_agl(trajectory, window, agl_provider)

    return tube.corridor(trajectory, window=window, extra_buffer=field_of_view.ground_radius(agl))


def tiles_for_footprint(
    footprint: Polygon,
    tile_info: TileInfo,
    level: int,
    max_tiles: int = 100_000,
) -> tuple[TileRef, ...]:
    """Tiles at ``level`` whose footprint actually intersects ``footprint``.

    ``footprint`` is a WGS84 (lon, lat) polygon. Tiles are enumerated over the
    footprint's bounding box in the tile scheme's spatial reference, then
    filtered by real polygon intersection - a corridor is a thin diagonal
    shape, so its bounding box can hold several times as many tiles as it
    genuinely covers.

    ``max_tiles`` guards against an accidentally fine ``level``: at
    ~1.9 cm/px, one square kilometre is tens of thousands of tiles, and
    enumerating a whole flight at that level would exhaust memory. Exceeding
    it raises rather than silently truncating.
    """
    bounds_4326 = polygon_bounds(footprint)
    row_min, row_max, col_min, col_max = tile_row_col_range(
        tile_info, level, _to_tile_sr(bounds_4326, tile_info)
    )
    total = max(row_max - row_min + 1, 0) * max(col_max - col_min + 1, 0)
    if total > max_tiles:
        raise ValueError(
            f"level {level} covers {total} tiles for this footprint, over the {max_tiles} limit; "
            "use a coarser level or a smaller window"
        )

    refs: list[TileRef] = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            native = tile_bounds(tile_info, level, row, col)
            wgs84 = extent_3857_to_4326(native) if native.wkid != 4326 else native
            if box(wgs84.xmin, wgs84.ymin, wgs84.xmax, wgs84.ymax).intersects(footprint):
                refs.append(TileRef(level=level, row=row, col=col, bounds=wgs84))
    return tuple(refs)


def _to_tile_sr(extent: Extent, tile_info: TileInfo) -> Extent:
    """Reproject an EPSG:4326 extent into the tile scheme's spatial reference."""
    if extent.wkid == tile_info.wkid:
        return extent
    if tile_info.wkid in (3857, 102100, 102113):
        return extent_4326_to_3857(extent)
    raise ValueError(f"unsupported tile scheme spatial reference: wkid={tile_info.wkid}")


def merge_tiles(tile_groups: Iterable[Iterable[TileRef]]) -> tuple[TileRef, ...]:
    """De-duplicate tiles across windows/trajectories, ordered by ``level/row/col``.

    Adjacent windows overlap at their shared boundary, so the same tile is
    routinely produced by more than one window; the ground-truth builder wants
    each tile once.
    """
    unique: dict[tuple[int, int, int], TileRef] = {}
    for group in tile_groups:
        for ref in group:
            unique.setdefault((ref.level, ref.row, ref.col), ref)
    return tuple(unique[key] for key in sorted(unique))


def footprint_union(footprints: Iterable[Polygon]) -> Polygon:
    """Union of several WGS84 footprints into one geometry."""
    return unary_union(list(footprints))
