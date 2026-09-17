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

## Running it

Normally run via DVC (`dvc repro fetch_imagery` - see the top-level
README's "Running the pipeline" section); the direct CLI is useful for a
one-off AOI or debugging:

```bash
uv run python scripts/fetch_historic_imagery.py \
    --bbox -121.95 37.30 -121.85 37.36 \
    --output-dir data/raw/dpw_imagery
```

This discovers every `DPW_Imagery*` service under the `Imagery` folder
(current + all historic vintages), fetches the tiles covering the bounding
box for each one, reprojects them from EPSG:3857 to EPSG:4326, and writes
one GeoTIFF per tile under `data/raw/dpw_imagery/<service-name>/`.

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--bbox MINLON MINLAT MAXLON MAXLAT` | yes | - | Area of interest, as four floats in EPSG:4326 (lon/lat degrees). |
| `--output-dir PATH` | yes | - | Directory to write GeoTIFFs into; one subfolder per discovered service, created if missing. |
| `--base-url URL` | no | `https://geo.sanjoseca.gov/server/rest/services` | Root of the ArcGIS REST services directory to search. |
| `--name-contains TEXT` | no | `DPW_Imagery` | Substring used to match service names under `Imagery` - matches every historic vintage whose name contains it (e.g. also matches `DPW_ImageryCached2025`), not just one exact name. |
| `--level N` | no | auto-detected (see below) | Tile LOD level to fetch, per that service's own `tileInfo`. |
| `--overwrite` | no | off | Re-fetch a tile even if its output GeoTIFF already exists. Without it, a tile already on disk is skipped - see "Resuming a run" below. |
| `--coverage-sample-size N` | no | 25 | Tiles to sample when checking a level actually has cached coverage for the AOI before committing to a full run - see "Auto-detected level" below. |
| `--skip-coverage-check` | no | off | Skip that sample check entirely. With `--level`, fetches it unconditionally; without `--level`, falls back to the naive finest-level default, unchecked. |
| `-v`, `--verbose` | no | off | Enable DEBUG-level logging, including per-tile "not cached at this level" messages that are otherwise suppressed. |

### Auto-detected level

Without `--level`, the script does **not** just use the finest level - some
ArcGIS caches (San Jose's included) only generate tiles for part of an AOI
at their finest zoom, or none of it at all. Before committing to a full run,
it samples `--coverage-sample-size` tiles spread across the AOI at each
level, from finest to coarsest, and uses the first level with any sampled
coverage. This check is deliberately cheap (a handful of requests per
level, not the whole grid) so a level with zero coverage is skipped in
seconds instead of grinding through possibly millions of individual 404s
over several hours. Passing an explicit `--level` runs the same check
against just that level and stops with an error - rather than quietly
running for hours - if it finds no coverage; the error lists the levels the
service actually has, in case you meant a different one. This is a sample,
not an exhaustive check, so it's possible (if rare in practice) for it to
miss extremely sparse coverage - pass `--skip-coverage-check` to force that
`--level` through anyway if you suspect a false negative.

### Resuming a run

Re-running the same command skips any tile whose output GeoTIFF
(`<level>_<row>_<col>.tif`) already exists in `<output-dir>/<service-name>/`,
instead of re-downloading and re-warping it - safe to interrupt (Ctrl-C) and
restart, or to rerun with a larger/overlapping `--bbox`. Each tile is written
to a temp file and renamed into place only once it's complete, so an
interrupted write never leaves a partial file that a later run would
mistake for a finished download. Pass `--overwrite` to force re-fetching
everything instead.

Even at the auto-detected (or explicitly requested) level, ArcGIS only
generates cache tiles where source imagery actually exists, so it's normal
for *some* individual tiles within the AOI to still come back "not cached" -
that's different from the whole-level "no coverage at all" case the
coverage check screens out above. The script logs a per-service summary of
tiles written vs. not cached vs. failed either way.

Progress is shown live via a `tqdm` bar (services overall, plus a per-service
tile bar with running written/missing/failed counts) - useful since a large
AOI at a fine `--level` can mean fetching thousands of tiles.

## Running the tests

```bash
uv sync --extra dev
uv run pytest
```
