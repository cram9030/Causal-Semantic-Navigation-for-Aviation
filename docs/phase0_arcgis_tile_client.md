# Phase 0: ArcGIS tile client for DPW imagery

Phase 0 data collection needs aerial imagery tiles from San Jose's ArcGIS
Server (`geo.sanjoseca.gov`), specifically the `DPW_ImageryCached` service
and its historic counterparts, reprojected from the service's native
EPSG:3857 (Web Mercator) into EPSG:4326 for the rest of the training-data
pipeline.

## Why discovery instead of a hardcoded service name

`DPW_ImageryCached` is the current/default cached basemap, but San Jose
publishes each imagery capture as its own service under the `Imagery`
folder (e.g. a service per flown year). Training data should draw from the
**full historic archive**, not just the newest layer, so
`csnav.data.arcgis.catalog.ArcGISCatalog.discover_imagery_services()` walks
the REST services directory recursively and returns every service matching
a name substring (default `DPW_Imagery`) instead of assuming one fixed URL.
That keeps the pipeline correct even if the exact set of historic service
names isn't known ahead of time, and it will keep picking up newly
published vintages without a code change.

## Module layout

```
src/csnav/data/arcgis/
├── models.py        # ServiceRef, TileInfo, LevelOfDetail, Extent, ServiceMetadata
├── projections.py    # EPSG:4326 <-> EPSG:3857 helpers (pyproj)
├── catalog.py         # ArcGISCatalog: recursive service discovery + historic-year extraction
├── tiles.py            # ArcGIS tileInfo-based tile bounds / row-col-for-extent math
├── client.py            # ArcGISTileClient: WMTS / /tile / /export, picks the best transport
└── reproject.py          # warp fetched tile bytes from EPSG:3857 to EPSG:4326 (rasterio)
```

`scripts/fetch_historic_imagery.py` ties these together: discover every
matching service, fetch the tiles covering a bounding box for each one, warp
to EPSG:4326, and write one GeoTIFF per tile under
`<output-dir>/<service-name>/`.

## Transport selection

`ArcGISTileClient.best_transport()` prefers, in order:

1. **`/tile/{level}/{row}/{col}`** - ArcGIS's native cached-tile resource,
   used whenever the service metadata includes a `tileInfo` block (i.e. it's
   a cached/tiled MapServer). This is preferred over WMTS because it
   addresses tiles with the *exact* level/row/col grid computed from the
   service's own `tileInfo` (see `tiles.tile_bounds`), so there's no risk of
   the WMTS `TileMatrix` identifiers not lining up with that grid.
2. **WMTS** - used when the service exposes `WMTS/1.0.0/WMTSCapabilities.xml`
   (checked with a lightweight probe request) but has no native tile cache,
   e.g. a dynamic MapServer with the WMTS capability enabled.
3. **`/export`** - dynamic image export, used when the service isn't
   pre-tiled and doesn't expose WMTS, or as a fallback for an arbitrary
   (non tile-aligned) bounding box via `fetch_export()` directly.

Earlier revisions of this client preferred WMTS first. In practice,
geo.sanjoseca.gov's WMTS `ResourceURL` template returned a 400 for tile
requests built from the MapServer's own tileInfo level/row/col - the WMTS
`TileMatrix` identifiers for a given `TileMatrixSet` aren't guaranteed to
equal the cache's level numbers, and this client doesn't parse the
`TileMatrixSet`'s own matrix definitions to translate between the two. The
native `/tile` resource has no such ambiguity, so it's tried first whenever
it's available.

## Reprojection

Tiles come back as raw PNG/JPEG bytes with no embedded georeferencing.
`reproject_tile_to_4326()` combines the tile's *known* bounds (computed from
the service's `tileInfo`, see `tiles.tile_bounds`) with the pixel data to
build a source raster in EPSG:3857, then warps it to EPSG:4326 with
`rasterio.warp.reproject`.

## 404s at the finest zoom level are expected, not a bug

ArcGIS cache generation only creates tiles that intersect actual source
imagery, especially at the deepest LODs - `tileInfo.lods` can list a level
(e.g. `DPW_ImageryCached2025`'s level 23, ~1.9cm/pixel) without every tile
in that level's theoretical row/col grid having been generated across the
whole service extent. Requesting one of those un-generated tiles returns a
plain `404`, not an error condition. `fetch_historic_imagery.py` treats a
404 from `/tile` as "not cached here" and skips it (logged at `DEBUG`,
counted separately from real failures); if *every* tile in an AOI comes
back 404 at the default (finest) level, that's a sign the service's
deep-zoom coverage doesn't reach that area - rerun with a coarser
`--level` to confirm.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```
