# Causal Semantic Navigation for Aviation

## Phase 0: data pipeline

This phase builds the ArcGIS tile client used to collect aerial imagery
training data from San Jose's ArcGIS Server (`geo.sanjoseca.gov`), covering
the `DPW_ImageryCached` service and its historic vintages. See
[`docs/phase0_arcgis_tile_client.md`](docs/phase0_arcgis_tile_client.md) for
design details.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Tests

```bash
pytest
```

### Fetching historic imagery for an area of interest

```bash
python scripts/fetch_historic_imagery.py \
    --bbox -121.95 37.30 -121.85 37.36 \
    --output-dir data/raw/dpw_imagery
```

This discovers every `DPW_Imagery*` service under the `Imagery` folder
(current + all historic vintages), fetches the tiles covering the bounding
box for each one, reprojects them from EPSG:3857 to EPSG:4326, and writes
one GeoTIFF per tile under `data/raw/dpw_imagery/<service-name>/`.

#### Options

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--bbox MINLON MINLAT MAXLON MAXLAT` | yes | - | Area of interest, as four floats in EPSG:4326 (lon/lat degrees). |
| `--output-dir PATH` | yes | - | Directory to write GeoTIFFs into; one subfolder per discovered service, created if missing. |
| `--base-url URL` | no | `https://geo.sanjoseca.gov/server/rest/services` | Root of the ArcGIS REST services directory to search. |
| `--name-contains TEXT` | no | `DPW_Imagery` | Substring used to match service names under `Imagery` - matches every historic vintage whose name contains it (e.g. also matches `DPW_ImageryCached2025`), not just one exact name. |
| `--level N` | no | each service's finest cached level | Tile LOD level to fetch, per that service's own `tileInfo`. The finest level is often only cached for part of the AOI (see note below) - pass a coarser (smaller) level for full-AOI coverage. |
| `-v`, `--verbose` | no | off | Enable DEBUG-level logging, including per-tile "not cached at this level" messages that are otherwise suppressed. |

Note on `--level` and missing tiles: ArcGIS only generates cache tiles where
source imagery actually exists, especially at the finest levels, so it's
normal for some (or, at the deepest level, even all) tiles in an AOI to come
back "not cached" rather than written. The script logs a per-service summary
of tiles written vs. not cached vs. failed, and warns if a whole run came
back empty so you know to retry with a coarser `--level`.

Progress is shown live via a `tqdm` bar (services overall, plus a per-service
tile bar with running written/missing/failed counts) - useful since a large
AOI at a fine `--level` can mean fetching thousands of tiles.
