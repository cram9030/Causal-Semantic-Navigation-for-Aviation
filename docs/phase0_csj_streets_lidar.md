# Phase 0: CSJ Streets + LIDAR elevation clients

Phase 0 also needs two more San Jose datasets, both normalized to EPSG:4326
per `docs/INTEGRATION_PLAN.md` §2:

- **CSJ `Streets`** - road centerlines with width/lane attributes, the
  Phase 1 manifest builder's source for candidate landmark geometry (§3.3)
  and, later, buffer widths for ground-truth label rasterization.
- **San Jose's Imagery & Elevation LIDAR product** - ground elevation, used
  for AGL correction and FOV occlusion modeling (§2), not just visualization.

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

The LIDAR elevation product, by contrast, is published as its own
`ImageServer`, so it only needs the existing service-level discovery
(`ArcGISCatalog.discover_services(..., service_types=("ImageServer",))`) -
no sublayer resolution required.

Both `scripts/fetch_csj_streets.py` and `scripts/fetch_lidar_elevation.py`
also accept an explicit `--layer-url`/`--service-url` to skip discovery
entirely once the real endpoint is known for a given environment, since a
live discovery request isn't always possible (e.g. sandboxed/offline dev).

## Module layout

```
src/csnav/data/arcgis/
├── catalog.py     # + discover_services() (generic) and find_layer() (sublayer-by-name resolution)
├── streets.py      # CSJStreetsClient: paginated /query against one Streets layer, GeoJSON in EPSG:4326
└── elevation.py      # LidarElevationClient: point identify() + AOI export via /exportImage, EPSG:4326
```

`streets.py` and `elevation.py` sit alongside the existing tile client
(`client.py`) rather than under a separate `data/acquisition/` package -
they're all ArcGIS Server clients for the same `geo.sanjoseca.gov` catalog
and share its models/catalog/projections utilities, so keeping them in one
package avoids duplicating that plumbing. See `docs/INTEGRATION_PLAN.md` §6
for how this maps onto the originally-sketched module layout.

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
- `export_elevation(extent, width, height, pixel_type="F32") -> bytes` +
  `load_elevation_tile(...)` - AOI raster pull via `/exportImage`, requesting
  `bboxSR=imageSR=4326` directly (the service reprojects server-side, unlike
  the Web Mercator-only cached imagery tiles in `client.py`/`reproject.py`,
  so no client-side `rasterio.warp` step is needed here). `load_elevation_tile`
  still builds its own transform from the *requested* extent/width/height
  rather than trusting whatever georeferencing the returned TIFF embeds -
  the same "always supply our own transform" approach `reproject_tile_to_4326`
  uses for imagery tiles.
- Returns a `ReprojectedTile` (reused from `reproject.py` - same
  data/transform/crs/`to_geotiff()` shape applies whether or not a CRS warp
  actually happened).

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/data/arcgis/test_streets.py tests/data/arcgis/test_elevation.py \
       tests/data/arcgis/test_catalog.py \
       tests/scripts/test_fetch_csj_streets.py tests/scripts/test_fetch_lidar_elevation.py
```

All tests mock ArcGIS responses with `responses`; none of them make live
network calls, since the exact CSJ Streets/LIDAR service and layer names
should be confirmed against the live catalog for a given environment before
relying on the discovery defaults in a real pull.
