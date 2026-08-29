# Phase 0: CSJ Streets + LIDAR elevation clients

Phase 0 also needs two more San Jose datasets, both normalized to EPSG:4326
per `docs/INTEGRATION_PLAN.md` §2:

- **CSJ `Streets`** - road centerlines with width/lane attributes, the
  Phase 1 manifest builder's source for candidate landmark geometry (§3.3)
  and, later, buffer widths for ground-truth label rasterization.
- **Ground elevation**, used for AGL correction and FOV occlusion modeling
  (§2), not just visualization. San Jose's own "Imagery & Elevation" LIDAR
  product (Valley Water) turned out, on inspection, not to serve this need -
  see "Ground elevation: San Jose's LIDAR product turned out to be contour
  lines" below - so this ended up sourced from USGS 3DEP instead, via a
  different module (`csnav.data.lidar`, not `csnav.data.arcgis`) and a
  different design than everything else in this doc.

## Why discovery instead of a hardcoded service/layer name

`docs/phase0_arcgis_tile_client.md` already made this call for imagery
services, and it applies here too, with one extra wrinkle: **`Streets` isn't
its own top-level service** - it's one layer (`name: "Streets"`) inside a
shared, generically-named service (at the time of writing,
`OPN/OPN_OpenDataService/MapServer`), alongside dozens of unrelated layers
(parcels, zoning, etc.). Service-level name matching
(`ArcGISCatalog.discover_services`, the same building block
`discover_imagery_services` now sits on top of) isn't enough to find it.

`ArcGISCatalog.find_layer()` handles this: it walks every service matching a
name substring under a root folder, fetches each one's own layer list, and
returns the REST URL of the first layer whose *name* (not the service's)
matches. This keeps `CSJStreetsClient` from hardcoding a service name that
has already changed once and could again - only the substrings passed to
`find_layer` need updating if San Jose reorganizes its catalog.
`fetch_csj_streets.py` defaults `--root` to `OPN` (where `Streets` lives at
the time of writing) rather than the whole services directory, both to keep
discovery fast and because the wider directory contains folders (e.g.
`Internal`) that 403/404 when actually listed despite being advertised in
the parent folder's JSON; pass `--root ''` to search everything if San
Jose's layout has moved `Streets` elsewhere.

`ArcGISCatalog.walk()` tolerates that kind of folder on its own too - a
folder that 403/404s when listed is logged and skipped rather than aborting
the whole walk, since one bad or restricted folder shouldn't hide every
service elsewhere in the tree (this matters most when searching a wide or
whole-catalog root, e.g. for `--root ''` above). An ArcGIS *error payload*
(a 200 response with an `error` body, as opposed to a plain HTTP error)
still propagates - that typically means something is wrong with the request
itself, not just "this folder isn't here".

`scripts/fetch_csj_streets.py` also accepts an explicit `--layer-url` to
skip discovery entirely once the real endpoint is known for a given
environment, since a live discovery request isn't always possible (e.g.
sandboxed/offline dev).

## Ground elevation: San Jose's LIDAR product turned out to be contour lines

Two earlier versions of this client tried to source ground elevation from
San Jose's own "Imagery & Elevation" LIDAR product, and both were wrong
about what that product actually *is*:

1. First attempt: assumed it was an ArcGIS `ImageServer` on
   `geo.sanjoseca.gov`, discovered the same way `Streets`/imagery are. Wrong
   - no such service exists in that catalog.
2. Second attempt: it's actually published by Valley Water (Santa Clara
   Valley Water District) as two static, whole-county ZIP downloads with no
   ArcGIS Server or query endpoint at all -
   `https://gis.valleywater.org/Download/LiDAR{1,5}FT.zip` - and this
   client downloaded + extracted one, assuming the archive held a plain
   raster file. Also wrong: a real `LiDAR5FT.zip` extracts to an Esri File
   Geodatabase (`LiDAR5FT.gdb/`) holding exactly one layer,
   ```
   $ ogrinfo -so LiDAR5FT.gdb
   Layer: LiDAR5FT (Multi Line String)
   ```
   confirmed (via `gdalinfo` too, not just `rasterio`'s bundled GDAL) to
   have no raster/mosaic dataset inside it anywhere. It's **contour lines**
   (matching the accompanying `5ft_contours.txt`'s own FGDC metadata title,
   "LiDAR Contour Shapefile Grid") - `docs/INTEGRATION_PLAN.md`'s original
   data-source table already hedged this ("Ground elevation / **contour
   data**"), which earlier revisions of this client didn't take seriously
   enough.

Getting an elevation *surface* value at an arbitrary `(lon, lat)` out of
contour lines needs interpolation (TIN, IDW, ...) - real, unbuilt work with
real accuracy tradeoffs for the 200-4000 ft AGL operating envelope this
project targets, and a new vector-geometry dependency this codebase doesn't
otherwise need. Rather than build that silently, this was a real decision
point, not a design detail - see the project owner's call below.

## Ground elevation, actually: USGS 3DEP

The chosen path is to source ground elevation from USGS's 3D Elevation
Program (3DEP) national elevation mosaic instead of San Jose's own data -
a live ArcGIS ImageServer USGS maintains specifically for this kind of
programmatic per-AOI/per-point access:

```
https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer
```

Unlike `geo.sanjoseca.gov`'s catalog, there's no discovery step here - it's
a fixed, publicly documented federal endpoint, not something that moves
around inside a generic catalog folder the way `Streets` does. And unlike
the Valley Water archive, there's no local download/extract/cache step
either: `LidarElevationClient.read_window(bbox)`/`identify(lon, lat)` are
live per-request queries against `/exportImage`/`/identify`, requesting
output directly in EPSG:4326 (`bboxSR`/`imageSR`/`sr`) so the service
reprojects server-side - the same "the source reprojects, we don't need
`rasterio.warp`" shape `CSJStreetsClient`'s `outSR=4326` already uses, and
architecturally close to what the very first (San Jose ImageServer) attempt
above looked like - just pointed at a real, verified endpoint instead of a
guessed one.

This codebase's own sandbox can't reach `elevation.nationalmap.gov` (the
same as `geo.sanjoseca.gov` and `gis.valleywater.org`), so end-to-end
verification happened from a real devcontainer instead. `identify()` is
confirmed working there - `--identify -121.9 37.3` returns `41.5276`, a
plausible elevation for that spot (the Los Gatos/Almaden foothills south of
San Jose). Getting there took one real bug: the ImageServer `identify`
operation silently ignores a bare `sr` query parameter alongside a plain
`"lon,lat"` geometry string - the live response echoed the point back under
the service's *native* Web Mercator spatial reference (wkid 102100) instead
of the requested EPSG:4326, so `(-121.9, 37.3)` degrees got reinterpreted as
`(-121.9, 37.3)` **meters** near the Web Mercator origin (off the coast of
Africa) - zero catalog items there, which surfaced as a misleading `NoData`
rather than a loud error. `identify()` now embeds the spatial reference
directly in the `geometry` JSON object (the standard ArcGIS REST
convention) instead of relying on a separate `sr` param, which the service
does honor.

`read_window()`/`--bbox` uses `bboxSR`/`imageSR` for `exportImage` - a
different (correctly-documented, unambiguous) parameter pair than
`identify`'s bare `sr` - and is confirmed working too: `--bbox -121.95
37.30 -121.85 37.36 --output ...` wrote a real 512x512 GeoTIFF against the
live service.

## Module layout

```
src/csnav/data/
├── arcgis/
│   ├── catalog.py    # + discover_services() (generic) and find_layer() (sublayer-by-name resolution)
│   └── streets.py     # CSJStreetsClient: paginated /query against one Streets layer, GeoJSON in EPSG:4326
└── lidar.py             # LidarElevationClient: live USGS 3DEP ImageServer queries (read_window/identify)
```

`streets.py` sits alongside the existing tile client (`client.py`) under
`csnav.data.arcgis` rather than a separate `data/acquisition/` package -
they're all ArcGIS Server clients for the same `geo.sanjoseca.gov` catalog
and share its models/catalog/projections utilities, so keeping them in one
package avoids duplicating that plumbing. `lidar.py` sits outside that
package instead, since it isn't an ArcGIS client at all - it still reuses
`arcgis.models.Extent` and `arcgis.reproject.ReprojectedTile` rather than
duplicating those small shared types. See `docs/INTEGRATION_PLAN.md` §6 for
how this maps onto the originally-sketched module layout.

## `CSJStreetsClient`

- `query(bbox=None, where="1=1", out_fields="*")` - paginates via
  `resultOffset`/`resultRecordCount` until the service stops reporting
  `exceededTransferLimit` (falling back to a page-size-based heuristic if
  that field is absent), so it's safe to call unbounded against a layer with
  more centerlines than one page holds. `outSR` is always forced to 4326;
  `bbox`, if given, must already be in EPSG:4326 - this client does no
  metric geometry of its own (see `geometry/local_frame.py` for tube-envelope
  math that produces such a bbox).
- Returns `StreetSegment` objects (`object_id`, `parts` - a tuple of
  `(lon, lat)` polyline parts, mirroring GeoJSON LineString/MultiLineString -
  and a raw `attributes` dict) rather than a GeoDataFrame, so this module
  doesn't pull in a geospatial-geometry dependency (shapely/geopandas) the
  rest of the pipeline doesn't otherwise need. `StreetSegment.to_geojson_feature()`
  round-trips back to GeoJSON for output/interop.
- Field names for width/lane counts are read generically via `attributes`
  rather than named explicitly in `StreetSegment`, since the exact schema is
  a property of the live service, not this client.

## `LidarElevationClient`

- `identify(lon, lat) -> float | None` - single-point elevation via the
  ImageServer `/identify` operation; `None` on `NoData`.
- `read_window(bbox, width=512, height=512, pixel_type="F32") -> ReprojectedTile`
  - AOI raster pull via `/exportImage`, requesting `bboxSR=imageSR=4326`
  directly. Returns a `ReprojectedTile` (reused from `arcgis/reproject.py` -
  same data/transform/crs/`to_geotiff()` shape), with its transform built
  from the *requested* bbox/size rather than trusted from whatever
  georeferencing the returned TIFF embeds - the same "always supply our own
  transform" approach `reproject_tile_to_4326` uses for imagery tiles.
- `get_metadata()` - the service's own reported extent/pixel size/pixel
  type, for a lightweight reachability check.
- No `cache_dir`/download step - every call is live.

## Running the tests

```bash
uv sync --extra dev
uv run pytest tests/data/arcgis/test_streets.py tests/data/arcgis/test_catalog.py tests/data/test_lidar.py \
       tests/scripts/test_fetch_csj_streets.py tests/scripts/test_fetch_lidar_elevation.py
```

All tests mock HTTP responses with `responses`; none of them make live
network calls, since this codebase's own sandbox can't reach either
`geo.sanjoseca.gov` or `elevation.nationalmap.gov`. Both the exact CSJ
Streets layer location and USGS 3DEP's `identify()` have since been
confirmed against the live services from a real devcontainer - see "Ground
elevation, actually: USGS 3DEP" above for the one real bug that surfaced
doing so.
