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

1. **WMTS** - if the service exposes `WMTS/1.0.0/WMTSCapabilities.xml`
   (checked with a lightweight probe request).
2. **`/tile/{level}/{row}/{col}`** - ArcGIS's native cached-tile resource,
   used whenever the service metadata includes a `tileInfo` block (i.e. it's
   a cached/tiled MapServer).
3. **`/export`** - dynamic image export, used when the service isn't
   pre-tiled, or as a fallback for an arbitrary (non tile-aligned) bounding
   box via `fetch_export()` directly.

## Reprojection

Tiles come back as raw PNG/JPEG bytes with no embedded georeferencing.
`reproject_tile_to_4326()` combines the tile's *known* bounds (computed from
the service's `tileInfo`, see `tiles.tile_bounds`) with the pixel data to
build a source raster in EPSG:3857, then warps it to EPSG:4326 with
`rasterio.warp.reproject`.

## What couldn't be verified in this environment

This sandbox's egress policy blocks direct network access to
`geo.sanjoseca.gov` (confirmed via the proxy status endpoint - a 403 policy
denial, not a transient failure), so the exact set of historic service names
and their `tileInfo`/capabilities could not be inspected live. The client
and catalog are built against the documented ArcGIS Server REST API
contract and are deliberately discovery-driven (no hardcoded service list)
so they should work unmodified once run somewhere with access - but running
`scripts/fetch_historic_imagery.py` against the real service the first time
is the remaining validation step. All modules are covered by unit tests
against mocked ArcGIS REST responses (`tests/data/arcgis/`) so the request
building, tile-grid math, and reprojection logic are verified independent
of live connectivity.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```
