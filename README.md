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

This project uses [uv](https://docs.astral.sh/uv/) for dependency
management - it reads `pyproject.toml` and installs the exact versions
pinned in the committed `uv.lock`, rather than resolving fresh against
whatever's newest on PyPI (torch in particular moves fast enough that an
unpinned install can silently jump to a much larger/newer CUDA stack
between one setup and the next). [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
if you don't have it, then:

```bash
uv sync --extra dev
```

This creates a `.venv/` and installs the project in editable mode plus the
`dev` extra (pytest, responses). Run commands inside it with `uv run`
(e.g. `uv run pytest`), or `source .venv/bin/activate` to work in it
directly. Add `--extra ml` and/or `--extra dvc` (or `--all-extras` for
everything) as needed - see "Dev container" and "Data versioning (DVC)"
below.

#### Dev container (GPU-ready)

`.devcontainer/` defines a CUDA 12.4 dev container (VS Code Dev Containers /
GitHub Codespaces / any [Dev Containers spec](https://containers.dev/)
tool), with [uv](https://docs.astral.sh/uv/) and the
[Claude Code CLI](https://code.claude.com/docs/en/devcontainer) preinstalled.
It picks up a host GPU automatically when one is present
(`hostRequirements.gpu: "optional"`) and installs the `ml` extra (torch,
torchvision, transformers, scikit-learn) needed for fine-tuning Mask2Former
in Phase 2 — see `docs/INTEGRATION_PLAN.md` §5. No GPU is required for
Phase 0/1 work; the container just runs CPU-only in that case.

Open the repo in VS Code and choose **Dev Containers: Reopen in Container**,
or run `devcontainer up` from the [Dev Containers CLI](https://github.com/devcontainers/cli).
On first build, `.devcontainer/post-create.sh` installs the project
(`uv sync --all-extras`, from `uv.lock`) and prints whether a GPU is visible.

### Tests

```bash
uv run pytest
```

### Fetching historic imagery for an area of interest

```bash
uv run python scripts/fetch_historic_imagery.py \
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
uv run python scripts/fetch_csj_streets.py \
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
uv run python scripts/fetch_lidar_elevation.py \
    --bbox -121.95 37.30 -121.85 37.36 \
    --output data/raw/lidar/downtown_dem.tif
```

Both paths are confirmed working against the live service: `--identify -121.9
37.3` returns `41.5276`, and the `--bbox` example above writes a real
512x512 GeoTIFF. Pass `--identify LON LAT` instead of `--bbox`/`--output` to
print a single point's elevation rather than fetching a raster. See
[`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md) for
the full story of how this data source was chosen.

### Data versioning (DVC)

The three fetch scripts above are also wired up as a [DVC](https://dvc.org)
pipeline (`dvc.yaml` + `params.yaml`), so the AOI-scoped data pulls are
reproducible and their large outputs (imagery GeoTIFFs, the streets
GeoJSON, the LIDAR DEM) are versioned outside of git rather than just
gitignored-and-hoped-for. `data/raw/` and `*.tif` stay gitignored as
before - that's unaffected by and compatible with DVC, which tracks the
actual bytes separately via its own content-addressed cache, referenced
from git only through `dvc.yaml`/`dvc.lock`.

```bash
uv sync --extra dev --extra dvc
uv run dvc repro          # (re)run any stage whose script/deps/params changed
uv run dvc dag             # show the pipeline graph
uv run dvc push / uv run dvc pull # sync tracked data with the configured remote
```

`--extra dvc` only needs to be passed to `uv sync` once - the venv keeps
it installed for subsequent `uv run` calls, dvc included.

Stage parameters (AOI bbox, service URLs, output paths) live in
`params.yaml`, not hardcoded in the scripts - edit a value and `dvc repro`
reruns only the affected stage(s). For a one-off run without touching the
file, use `uv run dvc exp run --set-param aoi.min_lon=-121.90 ...`.

The `.dvc/config` checked in here points the default remote at a
**local placeholder directory** (`../csnav-dvc-storage`, a sibling of the
repo) so a solo checkout works with zero setup. Before this is used by
more than one machine/collaborator, or at San Jose-imagery scale, swap it
for real object storage, e.g.:

```bash
uv run dvc remote add -d storage s3://<bucket>/csnav-dvc     # or gs://, azure://, etc.
```

(and add the matching extra - `dvc[s3]`, `dvc[gs]`, `dvc[azure]` - to
`pyproject.toml`'s `dvc` group in place of the plain `dvc` pin.)

Unimplemented past Phase 0 (see `docs/INTEGRATION_PLAN.md` ss6): a
`build_manifest` stage for `trajectory/ManifestBuilder` once it exists,
parameterized by `manifest.tube_radius_m` in `params.yaml` (placeholder
left commented there) so the CONOPS/altitude tube-radius sweep CLAUDE.md
calls for is a `dvc exp run --set-param manifest.tube_radius_m=<value>`
away rather than a code change; and, once `segmentation/` lands, stages
for Mask2Former training/checkpoints and `dvc.yaml` `metrics:`/`plots:`
entries for the Phase 4 Integrity Risk / Time-to-Alert / Availability
comparison.

## Phase 1: trajectory set, transitions, tubes, and precomputed manifests

This phase defines the candidate trajectory set `T`, the primary trajectory
`t_p`, the known start state `x_0`, the transitions permitted between routes,
and the RNP-style containment tube; builds the offline per-window landmark
manifests from those; and provides the visualization tools for reviewing all of
it. See [`docs/phase1_trajectory_manifests.md`](docs/phase1_trajectory_manifests.md).

The pilot trajectory set and its CONOPS parameters live in
[`configs/scenarios/san_jose_downtown.yaml`](configs/scenarios/san_jose_downtown.yaml).
The tube radius is a config value with no default anywhere in code - it is
swept across experiments (`--tube-radius`, or `ConopsConfig.with_tube_radius`).

### Transitions are generated, not authored

A transition between two routes is **not known before flight**: it may begin
anywhere along the route being flown, at any arc length, not just at a
waypoint. So the scenario file declares only which hand-offs are *permitted* -
a `TransitionRule` - and the geometry is generated:

- **Initiation** is any arc-length position on the source route. A rule can
  narrow that window (`initiate_from_m` / `initiate_to_m`) for a target that
  only becomes valid past some point.
- **Arrival** is the first target waypoint ahead of where the initiation point
  projects onto the target's ground track, which is what guarantees the
  transition makes forward progress rather than doubling back.
- **Geometry** is a cubic Hermite spline matching the source's heading at the
  initiation point and the target's at the arrival waypoint. This is a
  placeholder for a real dynamics model - smooth and heading-continuous, but
  ignorant of turn rate, bank limits, and airspeed.
- **The family is the object**, not any one path: initiation is continuous, so
  the aircraft may be anywhere in the region the family sweeps while a hand-off
  is under way. That region is what the maps shade.

A return to `x_0` is an ordinary candidate route whose last waypoint is `x_0`,
one per outbound route, reached by the same machinery. Composite routes then
need no declaration at all - "fly `t_p`, divert to `t_alt_north`, then take the
northern return" is a path through the transition graph, and
`TrajectorySet.route_paths()` enumerates them.

### Visualizing the trajectory set

```bash
uv sync --extra dev --extra viz

uv run python scripts/visualize_trajectories.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --output-dir out/viz
```

Writes, into `--output-dir` - all of it self-contained, interactive HTML:

| File | What it shows |
| --- | --- |
| `trajectory_graph.html` | The **structural** view: `T` as a transition graph (routes as nodes, permitted hand-offs as edges, `x_0` as the entry), the routes those rules permit, and one arc-length profile per route showing height above ground, the tube radius, the camera's ground reach, and the manifest window boundaries. Node positions are graph layers, deliberately not geography - that is the map's job. |
| `trajectory_set.html` | The spatial view: every route with its tube and visible footprint, and every transition family with the region it can reach, over San Jose imagery. |
| `trajectory_<id>.html` | One map per route: its tube at the configured radius, the per-window visible footprints, and the imagery tiles those footprints cover. Windows overlap, so each is its own layer with a **window selector** panel - see below. |
| `transition_<source>__<target>.html` | One map per transition rule: every sampled hand-off, where each initiates on the source, the waypoint it rejoins at, and the region the family sweeps. |

#### Options

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--scenario PATH` | yes | - | Scenario YAML defining `T`, `t_p`, `x_0`, the transition rules, and the CONOPS parameters. |
| `--output-dir PATH` | yes | - | Directory to write the report and maps into; created if missing. |
| `--tube-radius M` | no | the scenario's `conops.tube_radius_m` | Override the tube radius, in meters - the sweep entry point. |
| `--tile-level N` | no | the scenario's `conops.tile_level` | Imagery cache level for the "tiles in view" layer. |
| `--transition-samples N` | no | the scenario's `conops.transition.samples` | Initiation points sampled per transition rule. A fidelity knob on how finely the continuous family is stood in for - denser to inspect, sparser to draw. |
| `--no-tiles` | no | off | Skip the imagery-tile layer on the per-route maps. |
| `--no-transitions` | no | off | Skip the transition-family layers and the per-rule maps. |
| `--no-imagery` | no | off | Omit the San Jose DPW imagery basemap layer, for maps reviewed without network access. |
| `-v`, `--verbose` | no | off | DEBUG-level logging. |

Every element on the maps is a toggleable layer, and hovering a corridor,
transition path, window footprint, or tile shows the numbers behind it (window
id, arc-length span, max AGL, camera ground reach, initiation arc length,
arrival waypoint, turn angles, tile `level/row/col`).

#### Isolating manifest windows

A trajectory's manifest windows overlap - adjacent ones share a boundary and
each corridor is round-capped - so all of them at once reads as a chain of
blobs. Any map that draws windows therefore puts each on its own layer and adds
a **window selector** panel (top right) rather than listing them in folium's
flat layer control:

- expand a trajectory to see its windows, each labelled with its index and
  arc-length span;
- tick individual windows, or use `all` / `none` per trajectory;
- hit `solo` on a window to isolate it and hide every other window.

Where a window carries several kinds of geometry - its footprint, its candidate
roads, its intersections, its imagery tiles - those appear as **category**
checkboxes across the top of the panel and apply to every window at once. A
layer is drawn when its window is selected *and* its category is enabled, so
"every window's roads with no footprints" and "just window 3, everything about
it" are both a click or two. Consecutive windows also alternate fill and dash so
the sequence stays countable when they are all shown.

### What the CONOPS parameters mean

`conops.window_length_m` is the target arc length of one **manifest window**. A
trajectory is chopped into windows along its arc length and §3.3 builds one
precomputed landmark manifest per window, so this trades manifest count against
how much ground each manifest covers: halve it and you get twice as many
manifests, each with a tighter footprint and a shorter candidate-road list for
the runtime lookup to sift. It is a target rather than an exact stride - a
trajectory is divided into `round(length / window_length_m)` *equal* windows,
so a 2 km route at a 1200 m window length gives two 1000 m windows rather than
1200 m plus an 800 m tail, whose manifest would nearly duplicate its
neighbour's.

`conops.camera` carries the field of view, the **sensor pose** (mounting
relative to the body frame - all zeros is the nadir camera the first prototype
flies), and an **attitude margin**: how far off level the aircraft is allowed
to be when a footprint is sized, with a larger allowance applying near
waypoints where the turns are. The margin defaults to zero, so the first proof
of concept sizes footprints as if the aircraft were level; the mechanism is
there to switch on.

`conops.transition` carries the Hermite `tangent_gain` (how far the curve
bulges, scaled by the endpoint separation so the shape is scale-free), the
`max_turn_deg` feasibility screen, and the sampling density.

### Building the landmark manifests

```bash
uv run python scripts/build_manifests.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --output data/manifests/san_jose_downtown_r250.json \
    --map out/viz/manifests.html
```

For each window of each candidate route, this grows the tube by how far the
camera can see, queries CSJ Streets against that envelope, clips the returned
centerlines to it, derives their intersections, and records the imagery tiles
the window covers. The result is one pinned JSON bundle - the runtime "possible
roads" lookup reads it and never re-queries CSJ Streets.

**Every transition rule is covered the same way, by default.** A transition
may begin anywhere along its source, so the aircraft can legitimately be
anywhere the sampled family sweeps while a hand-off is under way - the
manifest has to say what could be seen from there too, not just from the
candidate routes. `build_set` generates each rule's family and builds windows
over every sampled path, querying streets once per family rather than once per
path. Pass `--no-transitions` to build candidate-route manifests only.

`--map` writes a review map of the built bundle, with the window selector
described above: expand a route or a transition rule to get its windows, solo
one, and see exactly what it covers - transition windows are labelled by which
sampled path they belong to and where it initiates, since a transition has no
single arc-length origin the way a route does. Pass `--map-landmarks` to
include each window's candidate roads and intersections as further categories
(off by default - across a whole bundle that is a lot of geometry, and the
per-route `manifest_map` view is the one for inspecting landmarks closely).

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
