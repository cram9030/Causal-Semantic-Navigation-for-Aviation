"""Coordinate-frame and metric geometry utilities.

Everything that turns angles or degrees into meters lives here, so the rest of
the pipeline can stay in WGS84:

* :mod:`csnav.geometry.local_frame` - WGS84 <-> local ENU tangent plane, for points.
* :mod:`csnav.geometry.shapes` - the same conversion for whole shapely geometries.
* :mod:`csnav.geometry.fov` - angular field of view -> ground footprint distance.

See `docs/INTEGRATION_PLAN.md` §2.
"""
