"""Precomputed landmark manifests - the offline output Phase 1 exists to produce.

Integration plan §3.3: for each trajectory window, the roads and intersections
that could possibly be visible from any state inside the tube, computed **once,
offline**, and pinned for the flight-planning cycle. At runtime the "possible
roads" node of the slice DAG (§3.4) is a lookup against this structure -
:meth:`LandmarkManifest.query` - and never a live spatial query against the
full CSJ Streets network (CLAUDE.md, "what not to do").

Everything stored here is WGS84 (EPSG:4326). Manifests serialize to plain
JSON/GeoJSON so a pinned manifest is a reviewable, diffable artifact rather
than a pickle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, MultiLineString, Point as ShapelyPoint, Polygon, shape

from csnav.data.arcgis.models import Extent
from csnav.trajectory.coverage import TileRef
from csnav.trajectory.trajectory import TrajectoryWindow

#: Manifest format version, bumped when the on-disk schema changes so a pinned
#: manifest from an earlier flight-planning cycle is not silently misread.
MANIFEST_SCHEMA_VERSION = 1


def _extent_to_dict(extent: Extent) -> dict[str, Any]:
    return {
        "xmin": extent.xmin,
        "ymin": extent.ymin,
        "xmax": extent.xmax,
        "ymax": extent.ymax,
        "wkid": extent.wkid,
    }


def _extent_from_dict(raw: dict[str, Any]) -> Extent:
    return Extent(
        xmin=raw["xmin"], ymin=raw["ymin"], xmax=raw["xmax"], ymax=raw["ymax"], wkid=raw.get("wkid", 4326)
    )


@dataclass(frozen=True)
class ManifestLandmark:
    """One candidate road in a window's manifest, in EPSG:4326.

    ``parts`` mirrors :class:`csnav.data.arcgis.streets.StreetSegment.parts` -
    a tuple of parts, each a tuple of ``(lon, lat)`` vertices - clipped to the
    window's visible footprint, so a long arterial contributes only the stretch
    this window could actually see. ``offset`` is the minimum horizontal
    distance in **meters** from the trajectory centerline to that clipped
    geometry, measured in the window's local ENU frame; it is evidence-side
    metadata (how far off-track a match implies the aircraft is), not part of
    the tube definition. ``width`` is the roadway width in meters where the
    CSJ attributes carry one, else ``None``.
    """

    segment_id: str
    parts: tuple[tuple[tuple[float, float], ...], ...]
    offset: float
    width: float | None = None
    name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict, compare=False)

    def geometry(self) -> LineString | MultiLineString:
        """This landmark's clipped centerline as a shapely geometry, in WGS84 (lon, lat)."""
        if len(self.parts) == 1:
            return LineString(self.parts[0])
        return MultiLineString([list(part) for part in self.parts])

    def to_geojson_feature(self) -> dict[str, Any]:
        geometry = (
            {"type": "LineString", "coordinates": [list(pt) for pt in self.parts[0]]}
            if len(self.parts) == 1
            else {"type": "MultiLineString", "coordinates": [[list(pt) for pt in part] for part in self.parts]}
        )
        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "segment_id": self.segment_id,
                "name": self.name,
                "offset_m": self.offset,
                "width_m": self.width,
                "attributes": self.attributes,
            },
        }

    @classmethod
    def from_geojson_feature(cls, feature: dict[str, Any]) -> "ManifestLandmark":
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") == "LineString":
            parts = (tuple((pt[0], pt[1]) for pt in coordinates),)
        else:
            parts = tuple(tuple((pt[0], pt[1]) for pt in part) for part in coordinates)
        properties = feature.get("properties") or {}
        return cls(
            segment_id=properties["segment_id"],
            parts=parts,
            offset=properties.get("offset_m", 0.0),
            width=properties.get("width_m"),
            name=properties.get("name"),
            attributes=properties.get("attributes") or {},
        )


@dataclass(frozen=True)
class ManifestIntersection:
    """A junction between two or more manifest roads, in EPSG:4326.

    ``lat``/``lon`` in decimal degrees; ``segment_ids`` are the
    :attr:`ManifestLandmark.segment_id` values meeting there. Intersections are
    derived from the manifest's own clipped centerlines - they are a distinct
    landmark class for the Mask2Former match step (§3.4), which detects road
    and intersection instances separately.
    """

    lat: float
    lon: float
    segment_ids: tuple[str, ...]

    def to_geojson_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
            "properties": {"segment_ids": list(self.segment_ids)},
        }

    @classmethod
    def from_geojson_feature(cls, feature: dict[str, Any]) -> "ManifestIntersection":
        lon, lat = feature["geometry"]["coordinates"]
        return cls(lat=lat, lon=lon, segment_ids=tuple(feature.get("properties", {}).get("segment_ids", ())))


@dataclass(frozen=True)
class LandmarkManifest:
    """Everything the runtime may look up for one trajectory window.

    ``tube_radius`` (meters), ``max_agl`` (meters above ground) and
    ``ground_reach`` (meters the camera could see beyond the ground track) are
    recorded with the manifest rather than assumed by its readers: the radius is
    a swept experimental parameter (CLAUDE.md core decision 4), so two manifests
    for the same window at different radii - or at the same radius with a
    different sensor pose - must be distinguishable after the fact.
    ``footprint`` is the visible-ground polygon the landmark query was issued
    against, in WGS84 (lon, lat); ``envelope`` is its bounding box.
    """

    window: TrajectoryWindow
    tube_radius: float
    max_agl: float
    ground_reach: float
    envelope: Extent
    footprint: Polygon
    candidate_roads: tuple[ManifestLandmark, ...]
    intersections: tuple[ManifestIntersection, ...] = ()
    tiles: tuple[TileRef, ...] = ()

    @property
    def window_id(self) -> str:
        """The key this manifest is stored and looked up under."""
        return self.window.window_id

    def query(self, fov_footprint: Polygon | None = None) -> tuple[ManifestLandmark, ...]:
        """Roads whose geometry intersects ``fov_footprint`` - the runtime lookup.

        ``fov_footprint`` is a WGS84 (lon, lat) polygon, typically the ground
        footprint of the predicted state's field of view (§3.4, "manifest
        n FOV(t)"). This is a pure in-memory filter over already-precomputed
        geometry: no network call, no query against CSJ Streets. Passing
        ``None`` returns the whole manifest.
        """
        if fov_footprint is None:
            return self.candidate_roads
        return tuple(road for road in self.candidate_roads if road.geometry().intersects(fov_footprint))

    def query_intersections(self, fov_footprint: Polygon | None = None) -> tuple[ManifestIntersection, ...]:
        """Intersections inside ``fov_footprint`` (WGS84 lon/lat polygon); all of them if ``None``."""
        if fov_footprint is None:
            return self.intersections
        return tuple(
            junction
            for junction in self.intersections
            if fov_footprint.intersects(ShapelyPoint(junction.lon, junction.lat))
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, with all geometry as GeoJSON in EPSG:4326."""
        return {
            "window": {
                "trajectory_id": self.window.trajectory_id,
                "index": self.window.index,
                "window_id": self.window.window_id,
                "start_distance_m": self.window.start_distance,
                "end_distance_m": self.window.end_distance,
                "start_time_s": self.window.start_time,
                "end_time_s": self.window.end_time,
            },
            "tube_radius_m": self.tube_radius,
            "max_agl_m": self.max_agl,
            "ground_reach_m": self.ground_reach,
            "envelope": _extent_to_dict(self.envelope),
            "footprint": self.footprint.__geo_interface__,
            "candidate_roads": [road.to_geojson_feature() for road in self.candidate_roads],
            "intersections": [junction.to_geojson_feature() for junction in self.intersections],
            "tiles": [
                {"level": tile.level, "row": tile.row, "col": tile.col, "bounds": _extent_to_dict(tile.bounds)}
                for tile in self.tiles
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LandmarkManifest":
        window_raw = raw["window"]
        window = TrajectoryWindow(
            trajectory_id=window_raw["trajectory_id"],
            index=window_raw["index"],
            start_distance=window_raw["start_distance_m"],
            end_distance=window_raw["end_distance_m"],
            start_time=window_raw["start_time_s"],
            end_time=window_raw["end_time_s"],
        )
        return cls(
            window=window,
            tube_radius=raw["tube_radius_m"],
            max_agl=raw["max_agl_m"],
            ground_reach=raw.get("ground_reach_m", 0.0),
            envelope=_extent_from_dict(raw["envelope"]),
            footprint=shape(raw["footprint"]),
            candidate_roads=tuple(
                ManifestLandmark.from_geojson_feature(feature) for feature in raw["candidate_roads"]
            ),
            intersections=tuple(
                ManifestIntersection.from_geojson_feature(feature) for feature in raw.get("intersections", [])
            ),
            tiles=tuple(
                TileRef(
                    level=tile["level"],
                    row=tile["row"],
                    col=tile["col"],
                    bounds=_extent_from_dict(tile["bounds"]),
                )
                for tile in raw.get("tiles", [])
            ),
        )


@dataclass(frozen=True)
class ManifestBundle:
    """Every window's manifest for one trajectory set, pinned as one artifact.

    Integration plan §3.3: a manifest is pinned to the flight-planning cycle it
    was built for and is *not* rebuilt when CSJ Streets refreshes weekly. The
    provenance recorded here - ``pinned_at``, ``streets_source``, and the
    ``parameters`` the build ran with (tube radii, window length, field of
    view, tile level) - is what makes that pinning auditable: a later run can
    tell which manifest it is holding and what it was built from.
    """

    trajectory_set_id: str
    manifests: tuple[LandmarkManifest, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    streets_source: str | None = None
    pinned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def by_window_id(self, window_id: str) -> LandmarkManifest:
        """The manifest pinned for ``window_id`` - the runtime's per-slice entry point."""
        for manifest in self.manifests:
            if manifest.window_id == window_id:
                return manifest
        raise KeyError(f"no manifest for window {window_id!r} in bundle {self.trajectory_set_id!r}")

    def for_trajectory(self, trajectory_id: str) -> tuple[LandmarkManifest, ...]:
        """All window manifests belonging to one trajectory, in window order."""
        return tuple(
            manifest
            for manifest in self.manifests
            if manifest.window.trajectory_id == trajectory_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_set_id": self.trajectory_set_id,
            "pinned_at": self.pinned_at,
            "streets_source": self.streets_source,
            "parameters": self.parameters,
            "manifests": [manifest.to_dict() for manifest in self.manifests],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ManifestBundle":
        version = raw.get("schema_version", MANIFEST_SCHEMA_VERSION)
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema version {version} != {MANIFEST_SCHEMA_VERSION}; "
                "rebuild the manifest rather than reading it with a mismatched reader"
            )
        return cls(
            trajectory_set_id=raw["trajectory_set_id"],
            manifests=tuple(LandmarkManifest.from_dict(item) for item in raw["manifests"]),
            parameters=raw.get("parameters") or {},
            streets_source=raw.get("streets_source"),
            pinned_at=raw.get("pinned_at", ""),
            schema_version=version,
        )

    def save(self, path: str | Path) -> Path:
        """Write the pinned bundle as JSON, creating parent directories."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ManifestBundle":
        """Read a pinned bundle previously written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def all_tiles(self) -> tuple[TileRef, ...]:
        """Every distinct imagery tile across all windows, ordered by ``level/row/col``.

        This is the tile set the ground-truth builder needs panoptic labels
        for (integration plan §4).
        """
        from csnav.trajectory.coverage import merge_tiles

        return merge_tiles(manifest.tiles for manifest in self.manifests)
