"""Folium review map for rasterized panoptic ground-truth labels.

The geographic counterpart to `csnav.viz.ground_truth_gallery`'s per-tile
paging view: this answers "does the rasterized road/intersection geometry
actually sit where the streets are, across the whole label set", the same
"did the offline build pick up the right streets?" question
`csnav.viz.map_view.manifest_map` answers for candidate-road manifests,
applied here to rasterized ground truth instead.

Road/intersection polygons are vectorized back out of each label's own
semantic/instance bands (`rasterio.features.shapes`) rather than carried as
separate stored geometry - `PanopticLabel` only stores the raster, so the map
draws exactly what a training loader would actually read, not a
reconstruction that could drift from it.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape

from csnav.data.arcgis.models import Extent
from csnav.data.ground_truth.labels import PanopticClass, PanopticLabel
from csnav.viz.map_view import base_map, save_map
from csnav.viz.style import INTERSECTION_COLOR, LANDMARK_COLOR, TILE_COLOR


def _folium():
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "csnav.viz.ground_truth_view needs folium; install the visualization extra with "
            "`uv sync --extra viz`"
        ) from exc
    return folium


def _polygon_latlon(polygon) -> list[list[list[float]]]:
    """Shapely WGS84 (lon, lat) polygon -> folium's ``[[lat, lon], ...]`` rings, exterior first."""
    rings = [list(polygon.exterior.coords)] + [list(interior.coords) for interior in polygon.interiors]
    return [[[lat, lon] for lon, lat in ring] for ring in rings]


def _label_bounds(labels: Sequence[PanopticLabel]) -> Extent:
    return Extent(
        xmin=min(label.tile.bounds.xmin for label in labels),
        ymin=min(label.tile.bounds.ymin for label in labels),
        xmax=max(label.tile.bounds.xmax for label in labels),
        ymax=max(label.tile.bounds.ymax for label in labels),
        wkid=4326,
    )


def _segments_by_instance(label: PanopticLabel) -> dict[int, Any]:
    return {segment.instance_id: segment for segment in label.segments}


def _add_label(fmap, roads_group, intersections_group, tiles_group, label: PanopticLabel) -> tuple[int, int]:
    folium = _folium()
    bounds = label.tile.bounds
    folium.Rectangle(
        bounds=[[bounds.ymin, bounds.xmin], [bounds.ymax, bounds.xmax]],
        color=TILE_COLOR,
        weight=0.8,
        opacity=0.7,
        fill=False,
        tooltip=f"tile {label.tile.key} - {len(label.segments)} instance(s)",
    ).add_to(tiles_group)

    segments = _segments_by_instance(label)
    road_count = 0
    intersection_count = 0
    mask = label.semantic != int(PanopticClass.BACKGROUND)
    if not mask.any():
        return road_count, intersection_count

    # rasterio's shapes() doesn't accept uint32 (only a fixed set of GDAL
    # dtypes); instance ids are small positive counters, so int32 is lossless.
    instance_i32 = label.instance.astype(np.int32)
    for geometry, instance_value in rio_shapes(instance_i32, mask=mask, transform=label.transform):
        segment = segments.get(int(instance_value))
        if segment is None:
            continue
        polygon = shapely_shape(geometry)
        if segment.class_id == int(PanopticClass.ROAD):
            group, color = roads_group, LANDMARK_COLOR
            tooltip = (
                f"{segment.name or segment.segment_id}<br>segment {segment.segment_id}<br>"
                f"tile {label.tile.key}"
                + (
                    f"<br>width {segment.width_m:.1f} m"
                    + (" (default)" if segment.default_width_used else "")
                    if segment.width_m
                    else ""
                )
            )
            road_count += 1
        else:
            group, color = intersections_group, INTERSECTION_COLOR
            tooltip = f"intersection of {', '.join(segment.intersection_segment_ids)}<br>tile {label.tile.key}"
            intersection_count += 1
        for part in _polygons(polygon):
            folium.Polygon(
                locations=_polygon_latlon(part),
                color=color,
                weight=1.0,
                opacity=0.9,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                tooltip=tooltip,
            ).add_to(group)
    return road_count, intersection_count


def _polygons(geometry) -> list[Any]:
    from shapely.geometry import Polygon

    if isinstance(geometry, Polygon):
        return [geometry]
    return [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]


def ground_truth_review_map(labels: Iterable[PanopticLabel]):
    """A folium map of a rasterized label set: tile footprints, roads, intersections.

    ``labels`` is typically every `PanopticLabel` under one
    ``scripts/build_ground_truth.py`` output directory (load with
    `csnav.data.ground_truth.checks.check_label_directory` and pull
    ``.tile``/reload, or iterate the directory directly with
    :meth:`PanopticLabel.load`). Each kind of geometry is one map-wide layer
    (not per-tile), toggled via folium's own layer control - unlike
    `csnav.viz.map_view.manifest_map`'s per-window selector, ground truth has
    no window structure to browse, only "how much is here and does it look
    right", which a flat toggle answers fine even for hundreds of tiles.
    """
    folium = _folium()
    labels = list(labels)
    if not labels:
        raise ValueError("no labels supplied")

    bounds = _label_bounds(labels)
    fmap = base_map(((bounds.ymin + bounds.ymax) / 2.0, (bounds.xmin + bounds.xmax) / 2.0), zoom=15)

    tiles_group = folium.FeatureGroup(name=f"tiles ({len(labels)})", show=True)
    roads_group = folium.FeatureGroup(name="roads", show=True)
    intersections_group = folium.FeatureGroup(name="intersections", show=True)

    total_roads = 0
    total_intersections = 0
    for label in labels:
        roads, intersections = _add_label(fmap, roads_group, intersections_group, tiles_group, label)
        total_roads += roads
        total_intersections += intersections

    tiles_group.add_to(fmap)
    roads_group.layer_name = f"roads ({total_roads})"
    roads_group.add_to(fmap)
    intersections_group.layer_name = f"intersections ({total_intersections})"
    intersections_group.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)
    fmap.fit_bounds([[bounds.ymin, bounds.xmin], [bounds.ymax, bounds.xmax]])
    return fmap


def save_ground_truth_map(fmap: Any, path):
    """Write a ground-truth review map to a self-contained HTML file, creating parent directories."""
    return save_map(fmap, path)
