# Causal Semantic Navigation for Aviation

## Phase 0: data pipeline

This phase builds the ArcGIS clients used to collect San Jose's data sources
from `geo.sanjoseca.gov`'s ArcGIS Server:

- Aerial imagery (`DPW_ImageryCached` and its historic vintages) - see
  [`docs/phase0_arcgis_tile_client.md`](docs/phase0_arcgis_tile_client.md).
- CSJ `Streets` centerlines and the Imagery & Elevation LIDAR product - see
  [`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md).
- The local ENU tangent-plane conversion utilities every downstream metric
  geometry step builds on - see
  [`docs/phase0_local_frame.md`](docs/phase0_local_frame.md).

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

#### Dev container (GPU-ready)

`.devcontainer/` defines a CUDA 12.4 dev container (VS Code Dev Containers /
GitHub Codespaces / any [Dev Containers spec](https://containers.dev/)
tool), with the [Claude Code CLI](https://code.claude.com/docs/en/devcontainer)
preinstalled. It picks up a host GPU automatically when one is present
(`hostRequirements.gpu: "optional"`) and installs the `ml` extra (torch,
torchvision, transformers, scikit-learn) needed for fine-tuning Mask2Former
in Phase 2 — see `docs/INTEGRATION_PLAN.md` §5. No GPU is required for
Phase 0/1 work; the container just runs CPU-only in that case.

Open the repo in VS Code and choose **Dev Containers: Reopen in Container**,
or run `devcontainer up` from the [Dev Containers CLI](https://github.com/devcontainers/cli).
On first build, `.devcontainer/post-create.sh` installs the project
(`pip install -e ".[dev,ml]"`) and prints whether a GPU is visible.

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
| `--level N` | no | auto-detected (see below) | Tile LOD level to fetch, per that service's own `tileInfo`. |
| `--overwrite` | no | off | Re-fetch a tile even if its output GeoTIFF already exists. Without it, a tile already on disk is skipped - see "Resuming a run" below. |
| `--coverage-sample-size N` | no | 25 | Tiles to sample when checking a level actually has cached coverage for the AOI before committing to a full run - see "Auto-detected level" below. |
| `--skip-coverage-check` | no | off | Skip that sample check entirely. With `--level`, fetches it unconditionally; without `--level`, falls back to the naive finest-level default, unchecked. |
| `-v`, `--verbose` | no | off | Enable DEBUG-level logging, including per-tile "not cached at this level" messages that are otherwise suppressed. |

#### Auto-detected level

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

#### Resuming a run

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

### Fetching CSJ Streets for an area of interest

```bash
python scripts/fetch_csj_streets.py \
    --bbox -121.95 37.30 -121.85 37.36 \
    --output data/raw/csj_streets/downtown.geojson
```

Resolves the `Streets` layer by name (it lives inside a shared, generically
named service rather than as its own top-level service - see
[`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md)),
queries it restricted to `--bbox` (in EPSG:4326; omit `--bbox` to pull the
whole layer), and writes the results as a GeoJSON `FeatureCollection`.
Pass `--layer-url` to skip discovery and query a known layer URL directly.

### Fetching ground elevation

Ground elevation ends up sourced from USGS's 3D Elevation Program (3DEP)
national elevation ImageServer, not San Jose's own "Imagery & Elevation"
LIDAR product - that turned out, on inspection of a real download, to be
contour lines (an Esri File Geodatabase holding one `MultiLineString`
layer), not a raster DEM. Getting a queryable elevation surface out of
contour lines needs interpolation (TIN/IDW), which isn't built here; see
[`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md) for
that investigation. USGS 3DEP is a fixed, publicly documented federal
ArcGIS ImageServer - no discovery needed, and every call is a live
per-request query (no local download/cache, unlike the Valley Water
approach this replaced).

```bash
python scripts/fetch_lidar_elevation.py \
    --bbox -121.95 37.30 -121.85 37.36 \
    --output data/raw/lidar/downtown_dem.tif
```

Both paths are confirmed working against the live service: `--identify -121.9
37.3` returns `41.5276`, and the `--bbox` example above writes a real
512x512 GeoTIFF. Pass `--identify LON LAT` instead of `--bbox`/`--output` to
print a single point's elevation rather than fetching a raster. See
[`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md) for
the full story of how this data source was chosen.

## Phase 1: trajectory set, tubes, and precomputed manifests

This phase defines the candidate trajectory set `T`, the primary trajectory
`t_p`, the known start state `x_0`, and the RNP-style containment tube; builds
the offline per-window landmark manifests from those; and provides the
visualization tools for reviewing all of it. See
[`docs/phase1_trajectory_manifests.md`](docs/phase1_trajectory_manifests.md).

The pilot trajectory set and its CONOPS parameters live in
[`configs/scenarios/san_jose_downtown.yaml`](configs/scenarios/san_jose_downtown.yaml).
The tube radius is a config value with no default anywhere in code - it is
swept across experiments (`--tube-radius`, or `ConopsConfig.with_tube_radius`).

### Visualizing the trajectory set

```bash
pip install -e ".[dev,viz]"

python scripts/visualize_trajectories.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --output-dir out/viz
```

Writes, into `--output-dir`:

| File | What it shows |
| --- | --- |
| `trajectory_graph.png` | The graph of `T`: candidates as nodes, transition corridors as edges, `x_0` as the entry node - plus one arc-length profile per candidate showing height above ground, the tube radius, the tube+FOV outer radius, and the manifest window boundaries. |
| `trajectory_set.html` | Interactive map of every trajectory with its own tube corridor and visible footprint, over San Jose imagery. |
| `trajectory_<id>.html` | One map per trajectory: its tube at the configured radius, the per-window visible footprints, and the imagery tiles those footprints cover. |

#### Options

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--scenario PATH` | yes | - | Scenario YAML defining `T`, `t_p`, `x_0`, and the CONOPS parameters. |
| `--output-dir PATH` | yes | - | Directory to write the figure and maps into; created if missing. |
| `--tube-radius M` | no | the scenario's `conops.tube_radius_m` | Override the tube radius, in meters - the sweep entry point. |
| `--tile-level N` | no | the scenario's `conops.tile_level` | Imagery cache level for the "tiles in view" layer. |
| `--no-tiles` | no | off | Skip the imagery-tile layer on the per-trajectory maps. |
| `--no-imagery` | no | off | Omit the San Jose DPW imagery basemap layer, for maps reviewed without network access. |
| `--dpi N` | no | 140 | Raster DPI for the PNG figure. |
| `-v`, `--verbose` | no | off | DEBUG-level logging. |

The maps are self-contained HTML - open them in a browser. Every element is a
toggleable layer, and hovering a corridor, window footprint, or tile shows the
numbers behind it (window id, arc-length span, max AGL, FOV ground radius,
tile `level/row/col`).

### Building the landmark manifests

```bash
python scripts/build_manifests.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --output data/manifests/san_jose_downtown_r250.json \
    --map out/viz/manifests.html
```

For each window of each trajectory, this grows the tube by the sensor's ground
field of view, queries CSJ Streets against that envelope, clips the returned
centerlines to it, derives their intersections, and records the imagery tiles
the window covers. The result is one pinned JSON bundle - the runtime "possible
roads" lookup reads it and never re-queries CSJ Streets.

Pass `--streets-geojson` to build from an archived pull written by
`scripts/fetch_csj_streets.py` instead of the live layer; prefer that when
rebuilding a manifest that has to match an earlier flight-planning cycle, since
the live layer refreshes weekly. Pass `--elevation` to derive AGL from USGS
3DEP ground elevation rather than treating waypoint height as AGL.

### Converting between WGS84 and a local ENU tangent plane

```python
from csnav.geometry.local_frame import LocalFrame

frame = LocalFrame(origin_lat=37.3382, origin_lon=-121.8863)

point = frame.to_enu(lat=37.3562, lon=-121.8663)  # Point(east=..., north=..., up=...) meters
back = frame.to_wgs84(point.east, point.north, point.up)  # LatLon(lat=..., lon=..., height=...)
```

Every metric geometry operation (RNP tube containment, street buffers, FOV
projection) should go through this conversion rather than doing
distance/area math directly on raw WGS84 degrees - see
[`docs/phase0_local_frame.md`](docs/phase0_local_frame.md).
