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

Every kind of geometry (tiles, roads, intersections) is drawn as **one**
``folium.GeoJson`` layer holding a whole ``FeatureCollection``, not one
Python object per shape. A real full-AOI label set is hundreds of tiles over
a dense city street network - one road can even vectorize into several
disjoint polygon pieces where an intersection cuts through it (see
:mod:`csnav.data.ground_truth.rasterize`) - so the shape count for even a
few hundred tiles can run into the thousands. One ``folium.Polygon`` /
``CircleMarker`` per shape means that many heavyweight Python/Jinja objects
alive at once before ``.render()`` ever runs, which is enough to OOM-kill the
process on a memory-constrained devcontainer well before it produces an
error. Three ``GeoJson`` layers - each just a plain dict serialized once -
avoid that no matter how many tiles are in the set.

``ground_truth_review_map`` also only needs one `PanopticLabel` - full
semantic/instance rasters included - alive **at a time**: it consumes
``labels`` as a single-pass iterator, extracting a label's (much smaller)
vectorized features before moving to the next, rather than materializing the
whole label set (every tile's full-resolution arrays, simultaneously) as a
list up front. A generator that loads one label from disk per step (see
``scripts/visualize_ground_truth.py``) keeps this at O(1) raster memory
regardless of how many tiles are in the set - the accumulated feature dicts
are far smaller than the rasters they came from.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from rasterio.features import shapes as rio_shapes

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


class _BoundsAccumulator:
    """Running min/max over tile bounds seen so far - avoids holding every label just to compute this."""

    def __init__(self) -> None:
        self.xmin = self.ymin = float("inf")
        self.xmax = self.ymax = float("-inf")

    def add(self, bounds) -> None:
        self.xmin = min(self.xmin, bounds.xmin)
        self.ymin = min(self.ymin, bounds.ymin)
        self.xmax = max(self.xmax, bounds.xmax)
        self.ymax = max(self.ymax, bounds.ymax)

    @property
    def center(self) -> tuple[float, float]:
        return (self.ymin + self.ymax) / 2.0, (self.xmin + self.xmax) / 2.0


def _segments_by_instance(label: PanopticLabel) -> dict[int, Any]:
    return {segment.instance_id: segment for segment in label.segments}


def _tile_feature(label: PanopticLabel) -> dict[str, Any]:
    bounds = label.tile.bounds
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [bounds.xmin, bounds.ymin],
                    [bounds.xmax, bounds.ymin],
                    [bounds.xmax, bounds.ymax],
                    [bounds.xmin, bounds.ymax],
                    [bounds.xmin, bounds.ymin],
                ]
            ],
        },
        "properties": {"tile": label.tile.key, "instances": len(label.segments)},
    }


def _label_shape_features(label: PanopticLabel) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Road/intersection polygons vectorized out of one label, as GeoJSON Feature dicts.

    ``rasterio.features.shapes`` already yields each polygon's geometry as a
    plain GeoJSON-shaped mapping in the raster's own CRS (EPSG:4326 here), so
    it is used directly as a Feature's ``geometry`` with no shapely round
    trip - one fewer object created per shape, which matters at this scale.
    """
    segments = _segments_by_instance(label)
    mask = label.semantic != int(PanopticClass.BACKGROUND)
    road_features: list[dict[str, Any]] = []
    intersection_features: list[dict[str, Any]] = []
    if not mask.any():
        return road_features, intersection_features

    # rasterio's shapes() doesn't accept uint32 (only a fixed set of GDAL
    # dtypes); instance ids are small positive counters, so int32 is lossless.
    instance_i32 = label.instance.astype(np.int32)
    for geometry, instance_value in rio_shapes(instance_i32, mask=mask, transform=label.transform):
        segment = segments.get(int(instance_value))
        if segment is None:
            continue
        if segment.class_id == int(PanopticClass.ROAD):
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "tile": label.tile.key,
                    "segment_id": segment.segment_id or "",
                    "name": segment.name or "",
                    "width_m": f"{segment.width_m:.1f}" if segment.width_m else "",
                    "default_width": "yes" if segment.default_width_used else "no",
                },
            }
            road_features.append(feature)
        else:
            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "tile": label.tile.key,
                    "intersection_of": ", ".join(segment.intersection_segment_ids),
                },
            }
            intersection_features.append(feature)
    return road_features, intersection_features


def ground_truth_review_map(labels: Iterable[PanopticLabel]):
    """A folium map of a rasterized label set: tile footprints, roads, intersections.

    ``labels`` is typically every `PanopticLabel` under one
    ``scripts/build_ground_truth.py`` output directory. It is consumed as a
    single-pass iterator - pass a generator that loads one label from disk at
    a time (see ``scripts/visualize_ground_truth.py``) rather than a
    pre-built list, so only one tile's full-resolution rasters are ever alive
    at once; a plain list works too, just without that memory benefit. Each
    kind of geometry is one map-wide ``GeoJson`` layer (not per-tile),
    toggled via folium's own layer control - unlike
    `csnav.viz.map_view.manifest_map`'s per-window selector, ground truth has
    no window structure to browse, only "how much is here and does it look
    right", which a flat toggle answers fine even for a full-AOI label set.
    """
    folium = _folium()

    bounds = _BoundsAccumulator()
    tile_features: list[dict[str, Any]] = []
    road_features: list[dict[str, Any]] = []
    intersection_features: list[dict[str, Any]] = []
    for label in labels:
        bounds.add(label.tile.bounds)
        tile_features.append(_tile_feature(label))
        roads, intersections = _label_shape_features(label)
        road_features.extend(roads)
        intersection_features.extend(intersections)

    if not tile_features:
        raise ValueError("no labels supplied")

    fmap = base_map(bounds.center, zoom=15)

    def _geojson_layer(features, name, color, fields, aliases, fill_opacity=0.45):
        # GeoJsonTooltip asserts its fields exist among the data's own
        # properties keys at render time - which fails outright against an
        # empty FeatureCollection (no properties keys to check against at
        # all), so a layer with nothing in it gets no tooltip rather than a
        # crash at .render()/.save() time.
        kwargs: dict[str, Any] = {
            "name": f"{name} ({len(features)})",
            "style_function": lambda _f, color=color, fill_opacity=fill_opacity: {
                "color": color, "weight": 1.0, "fillColor": color, "fillOpacity": fill_opacity,
            },
        }
        if features:
            kwargs["tooltip"] = folium.GeoJsonTooltip(fields=fields, aliases=aliases)
        return folium.GeoJson({"type": "FeatureCollection", "features": features}, **kwargs)

    _geojson_layer(
        tile_features, "tiles", TILE_COLOR, ["tile", "instances"], ["tile", "instance(s)"], fill_opacity=0.0
    ).add_to(fmap)
    _geojson_layer(
        road_features, "roads", LANDMARK_COLOR,
        ["name", "segment_id", "width_m", "default_width", "tile"],
        ["name", "segment", "width (m)", "default width?", "tile"],
    ).add_to(fmap)
    _geojson_layer(
        intersection_features, "intersections", INTERSECTION_COLOR,
        ["intersection_of", "tile"], ["intersection of", "tile"], fill_opacity=0.6,
    ).add_to(fmap)

    folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)
    fmap.fit_bounds([[bounds.ymin, bounds.xmin], [bounds.ymax, bounds.xmax]])
    return fmap


def save_ground_truth_map(fmap: Any, path):
    """Write a ground-truth review map to a self-contained HTML file, creating parent directories."""
    return save_map(fmap, path)
