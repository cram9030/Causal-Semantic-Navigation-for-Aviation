# Phase 0: local ENU tangent-plane utilities

Phase 0's last item, per `docs/INTEGRATION_PLAN.md` §5: stand up the WGS84
<-> local East-North-Up (ENU) tangent-plane conversion used by every
downstream metric geometry step (RNP tube containment, street buffers, FOV
projection - §2, §3.3). Implemented as `csnav.geometry.local_frame`,
matching the module `LocalFrame` sketched in the integration plan's UML
diagram (§7).

## Why a geocentric (cart + topocentric) pipeline, not a flat-earth approximation

`LocalFrame` builds a PROJ pipeline per origin:

```
+proj=pipeline
  +step +proj=cart +ellps=WGS84
  +step +proj=topocentric +ellps=WGS84 +lat_0=<origin_lat> +lon_0=<origin_lon> +h_0=<origin_height>
```

`cart` converts the WGS84 geodetic point to geocentric (ECEF) coordinates,
and `topocentric` rotates/translates that into East-North-Up meters
relative to the origin. This is exact for the WGS84 ellipsoid (not a
locally-flat or spherical-earth approximation like an equirectangular or
azimuthal-equidistant projection would be), and it's a two-line addition on
top of `pyproj`, already a project dependency for the EPSG:3857<->4326 work
in `csnav.data.arcgis.projections`. It also naturally carries a height/Up
component, needed later for AGL-aware FOV occlusion work, not just 2D
buffer/containment math.

Cross-checked in `tests/geometry/test_local_frame.py` against an
independent code path - `pyproj.Geod.inv`'s direct ellipsoidal
geodesic distance/azimuth computation - rather than only asserting a
round-trip: for a ~2.7km offset near the San Jose AOI, the two agree to
within millimeters.

## Re-anchoring, not one global frame

`LocalFrame` takes its origin in the constructor rather than being a
singleton/global - distortion from the tangent-plane approximation grows
with distance from the origin, so per §2/§3.2 of the integration plan, the
manifest builder (Phase 1) is expected to construct a fresh `LocalFrame`
per trajectory window rather than reuse one anchor for an entire flight.

## Module layout

```
src/csnav/geometry/
└── local_frame.py   # LocalFrame (WGS84 <-> local ENU meters), Point, LatLon
```

Lives under `src/csnav/` alongside `data/`, rather than a separate
top-level `geometry/` package, for the same reason `data/acquisition`
became `src/csnav/data/arcgis` in Phase 0 (see `docs/INTEGRATION_PLAN.md`
§6's "Implementation note"): one installable package, not a split source
tree.

## Running the tests

```bash
uv sync --extra dev
uv run pytest tests/geometry/test_local_frame.py
```
