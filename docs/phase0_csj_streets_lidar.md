# Phase 0: CSJ Streets + LIDAR elevation clients

Phase 0 also needs two more San Jose datasets, both normalized to EPSG:4326
per `docs/INTEGRATION_PLAN.md` §2:

- **CSJ `Streets`** - road centerlines with width/lane attributes, the
  Phase 1 manifest builder's source for candidate landmark geometry (§3.3)
  and, later, buffer widths for ground-truth label rasterization.
- **San Jose's Imagery & Elevation LIDAR product** - ground elevation, used
  for AGL correction and FOV occlusion modeling (§2), not just visualization.
  Despite the name, this one turns out **not** to be an ArcGIS service at
  all - see "LIDAR: a static download, not an ArcGIS service" below - so it
  gets a different module (`csnav.data.lidar`, not `csnav.data.arcgis`) and a
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

## LIDAR: a static download, not an ArcGIS service

The "discover it, don't hardcode it" approach above assumes the dataset is
actually served through `geo.sanjoseca.gov`'s ArcGIS catalog. San Jose's
LIDAR elevation product isn't: it turns out to be published by Valley Water
(Santa Clara Valley Water District) as two static, whole-county ZIP
downloads, with no ArcGIS Server, catalog, or per-request query endpoint in
front of them at all -

```
https://gis.valleywater.org/Download/LiDAR1FT.zip   (1 ft/px)
https://gis.valleywater.org/Download/LiDAR5FT.zip   (5 ft/px)
```

(An earlier version of this client tried to *discover* an ArcGIS
`ImageServer` for this product, the way `Streets`/imagery are discovered -
that was the wrong model entirely: there's nothing to discover when the URL
is fixed and already known. `csnav.data.lidar` replaces that attempt.)

Because there's no bounding-box query support at the source, "for the AOI"
scoping happens client-side, after the fact - and it only ever scopes a
*read*, never the *fetch*: **the entire chosen whole-county product always
downloads**, whether or not a caller ever asks for a `bbox`. There is no way
to fetch less; Valley Water doesn't offer one. `LidarElevationClient`
downloads + extracts the product once (cached under a `cache_dir`, skipped
on a later call unless `overwrite=True` - these are large, whole-county
archives, worth caching), and every subsequent `read_window(bbox)`/
`identify(lon, lat)` call is a local read against whatever raster(s) the
extracted archive contains, via `rasterio` - no live network call per query,
but also no reduction in what was already downloaded. `--bbox` on
`fetch_lidar_elevation.py` exists to keep the *output* file small (an AOI
GeoTIFF, not the whole county's worth of pixels), not to keep the *download*
small; running the script with neither `--bbox` nor `--identify` still
triggers the (one-time) download and just reports the raster source(s)
found, for exactly this reason - there's nothing left to scope without one
of those two.

The source raster's CRS is read from the file itself (not assumed), and
`read_window`/`identify` reproject to EPSG:4326 on read if it isn't already,
the same "never assume a source is already in EPSG:4326" rule the rest of
this codebase follows (`CLAUDE.md` §2). If a raster source turns out to be
many tiles rather than one seamless raster, `read_window` mosaics whichever
tiles intersect `bbox` via `rasterio.merge.merge` before reprojecting.

**What's actually inside the archive, confirmed against a real download:**
this codebase's own dev/CI environment can't reach `gis.valleywater.org`, so
the design above was originally written assuming a plain raster file
(`.tif`/`.img`/etc.) straight in the ZIP - wrong. A real `LiDAR5FT.zip`
extracts to an Esri **File Geodatabase** (`LiDAR5FT.gdb/`) plus a
`5ft_contours.txt`, not a bare raster file. Reading raster/mosaic data out
of a File Geodatabase with open-source GDAL is version/build-dependent - it
normally needs Esri's proprietary FileGDB SDK compiled in (this repo's own
sandbox has neither the `FileGDB` nor even the vector-only `OpenFileGDB`
GDAL driver registered, so it can't be tested here at all). `discover_rasters()`
now also looks inside any `.gdb` directory found under the extracted archive
and lists its raster subdatasets (`list_gdb_raster_subdatasets`, via
`rasterio.open(gdb_path).subdatasets`) - so this *should* work automatically
in an environment whose GDAL build supports it, without needing anything
further. If it doesn't - `ensure_local()` raises `LidarElevationError` with
a message naming the `.gdb` path(s) found and every other file extension
present (e.g. `.txt`) - that error message is the signal to check what GDAL
actually reports for that `.gdb` (`gdalinfo path/to/LiDAR5FT.gdb`, or `python
-c "import rasterio; ...` per the error text) and, if it turns out the
geodatabase holds only vector contour lines rather than a raster/mosaic
dataset, revisit this design entirely - a queryable elevation *surface* from
contour *lines* needs interpolation (TIN, IDW, etc.), which is a materially
different and larger piece of work than a raster windowed read, not
implemented here.

## Module layout

```
src/csnav/data/
├── arcgis/
│   ├── catalog.py    # + discover_services() (generic) and find_layer() (sublayer-by-name resolution)
│   └── streets.py     # CSJStreetsClient: paginated /query against one Streets layer, GeoJSON in EPSG:4326
└── lidar.py             # LidarElevationClient: download/extract Valley Water's ZIP, then local windowed reads
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

- `ensure_local(overwrite=False) -> list[Path | str]` - download
  (`download_archive`, streamed with a `tqdm` progress bar, resumable-skip if
  the archive is already on disk) + extract (`extract_archive`) the chosen
  `product` (`"1ft"` or `"5ft"`) - **always the whole archive**, regardless
  of any `bbox` a caller might use afterwards; there is no scoped-fetch
  option at the source. Returns the raster *sources* found via
  `discover_rasters()`: plain files with a recognized extension
  (`RASTER_EXTENSIONS`), plus, for each `.gdb` directory found
  (`find_gdb_dirs`), any raster subdataset URIs GDAL reports inside it
  (`list_gdb_raster_subdatasets` - see the File Geodatabase caveat above).
  Called automatically by `read_window`/`identify` on first use; call it
  explicitly first to control when the (potentially large) download happens,
  or to just prefetch + inspect what was found without reading anything.
- `read_window(bbox) -> ReprojectedTile` - the AOI raster covering `bbox`
  (must be EPSG:4326), mosaicked from whichever raster source(s) intersect
  it and reprojected to EPSG:4326 if the source CRS differs. Returns a
  `ReprojectedTile` (reused from `arcgis/reproject.py` - same
  data/transform/crs/`to_geotiff()` shape).
- `identify(lon, lat) -> float | None` - single-point elevation via a direct
  1x1-pixel index-and-read against the source raster (not a tiny
  `read_window` bbox - a bbox epsilon small enough to read as "a point" can
  still be narrower than this DEM's own pixel size, which would make
  `rasterio.merge.merge` round the output window to zero pixels). `None`
  where no raster source covers the point, or the pixel is nodata.

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/data/arcgis/test_streets.py tests/data/arcgis/test_catalog.py tests/data/test_lidar.py \
       tests/scripts/test_fetch_csj_streets.py tests/scripts/test_fetch_lidar_elevation.py
```

All tests mock HTTP responses with `responses`; none of them make live
network calls, since neither the exact CSJ Streets layer location nor the
Valley Water LIDAR archives' internal layout have been confirmed against
the live sources from this codebase's own dev/CI environment.
