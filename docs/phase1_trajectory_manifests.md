# Phase 1: trajectory set, RNP tube, manifests, and visualization

Phase 1 of `docs/INTEGRATION_PLAN.md` §5: *"Define `T`, `t_p`, `x_0`, and the
RNP-style tube model"* and *"implement the offline manifest builder: trajectory
windows → tube envelope → CSJ Streets query → per-window landmark manifest"*,
plus the visualization tools needed to actually review what that builder
produced.

## What got built

```
src/csnav/trajectory/
├── waypoints.py         # Waypoint (4D, WGS84), TrajectoryRole
├── trajectory.py        # Trajectory, TrajectorySet, Transition, TrajectoryWindow
├── tube.py              # TubeModel: containment + corridor/envelope geometry
├── coverage.py          # visible footprint (tube + FOV), TileRef, AGL providers
├── manifest.py          # LandmarkManifest, ManifestBundle, JSON pinning
├── manifest_builder.py  # the offline builder, and StaticStreetsSource
└── config.py            # Scenario / ConopsConfig, versioned YAML loading

src/csnav/geometry/
├── fov.py               # FieldOfView -> ground footprint radius
└── shapes.py            # WGS84 <-> ENU conversion for whole shapely geometries

src/csnav/viz/
├── style.py             # shared palette so both views agree on colours
├── graph_view.py        # matplotlib: the graph of T, per-trajectory profiles
└── map_view.py          # folium: corridors, window footprints, tiles, manifests

configs/scenarios/san_jose_downtown.yaml   # the pilot T / t_p / x_0 / CONOPS
scripts/visualize_trajectories.py          # render the figures and maps
scripts/build_manifests.py                 # build and pin a manifest bundle
```

As in Phase 0, these live under `src/csnav/` rather than as the top-level
`trajectory/` tree sketched in §6 of the integration plan — one installable
package, not a split source tree (see §6's "Implementation note").

## Design decisions worth knowing

### Arc length is the state, not lat/lon

`Trajectory` tracks progress as an explicit arc-length value in meters
(`cumulative_distances`, `point_at`, `distance_at_time`), computed as the 3D
path length in a local ENU frame anchored at the trajectory's first waypoint.
This is CLAUDE.md core decision 5: arc length is what decides which window's
manifest applies, and it is what the deterministic `Predict x(t)` mechanism
will advance in Phase 3. It is *not* re-derived from lat/lon on each use.

Being a 3D path length, a leg flown at 300 m is ~9 cm longer over 2 km than its
ground track — small, but it is the distance actually flown, which is what a
constant-velocity predictor should consume.

### Windows never emit slivers

`Trajectory.windows(window_length)` divides the trajectory into
`round(length / window_length)` equal windows rather than walking a fixed
stride and leaving a remainder. A 2 km trajectory at a 1200 m window length
gives two 1000 m windows, not 1200 m + a 800 m tail — a sliver window's
manifest would be nearly identical to its neighbour's while still costing a
build and a lookup entry. Window ids (`"<trajectory_id>:0000"`) are stable
across rebuilds of the same trajectory and window length, because they are the
key a pinned manifest is looked up under at runtime.

### Transition corridors are trajectories; the graph is over candidates

A transition corridor carries `role: transition` and declares the pair it
`connects`. `TrajectorySet.to_networkx()` then puts *candidates* on the nodes
and transitions on the edges (with the corridor as an edge attribute), plus the
reserved node id `x0` so the "return path to `x_0`" of §3.2 has somewhere to
point. `networkx.DiGraph` per CLAUDE.md, matching what the Phase 3 slice DAG
spec will use.

### Tube radius: an input, everywhere

`TubeModel(radius=...)` has no default, `ConopsConfig` refuses to load a
scenario without `conops.tube_radius_m`, and nothing in the tube or manifest
code derives a radius from landmark geometry. Sweeping is
`ConopsConfig.with_tube_radius(r)` — or `--tube-radius` on either script —
which is the whole of what §8's "sensitivity study on integrity risk vs. tube
radius" needs from this layer.

Containment (`TubeModel.contains`) is **lateral**: horizontal cross-track
distance from the ground track, as RNP defines it. Height is not part of the
test; a vertical containment bound is a later addition, not an assumption baked
in now.

Transition corridors can take their own radius
(`conops.transition_tube_radius_m`) or share the primary's by setting it to
`null`. §8 lists that choice as still open, so both are expressible and the
decision is recorded in the config rather than hardcoded.

### "Possible roads" means visible, not just contained

§3.3 scopes a manifest to what could be seen *from any state inside the tube*,
so the search footprint is the tube corridor grown by the sensor's ground FOV
radius at that window's **maximum** AGL:

```
footprint = buffer(window centerline, tube_radius + fov.ground_radius(max_agl))
```

`FieldOfView.ground_radius` is the half-diagonal of the nadir rectangular
footprint, so it bounds the footprint at any aircraft heading — heading inside
the tube is not constrained a priori. Taking the window's maximum AGL (not its
mean) keeps the manifest a superset of what any state in the window could see.

AGL itself comes from an `AglProvider`. The default treats waypoint height as
AGL, which is only right for plans authored in AGL; `agl_from_elevation` wraps
`LidarElevationClient.identify` (USGS 3DEP) for real terrain, per §2's point
that GPS-derived height is ellipsoidal/MSL.

### One street query per trajectory, clipped per window

`ManifestBuilder.build_trajectory` issues **one** CSJ Streets query for the
whole route's envelope and clips it per window in memory, rather than one query
per window. `--per-window-query` switches to per-window envelopes, which is
preferable for a long or strongly curved route where the whole-route bounding
box would pull in far more of the city than the corridor touches.

Clipping and the recorded off-track `offset` both happen in the trajectory's
local ENU frame, so `offset` is a real distance in meters, then the result is
converted back to WGS84 for storage.

Intersections are derived from the manifest's own clipped centerlines (STRtree
pairwise, snapped at 2 m), so road and intersection landmarks — the two classes
Mask2Former will detect separately in §3.4 — both come out of the same build.

### Pinning is a file, not a convention

`ManifestBundle.save` writes plain JSON with GeoJSON geometry: schema version,
`pinned_at`, the streets source it was built from, and the full parameter set
(tube radii, window length, FOV, tile level). §3.3 says a manifest is pinned to
its flight-planning cycle and not rebuilt on CSJ Streets' weekly refresh — that
is only auditable if the artifact says what it was built from, and only
reproducible if the build can be re-run from an archived streets pull, which is
what `StaticStreetsSource` and `--streets-geojson` are for.

At runtime, `LandmarkManifest.query(fov_footprint)` is a pure in-memory filter
over that pinned geometry. There is a regression test asserting it never
touches the street source.

## Visualization

Two views, deliberately different in kind.

**`csnav.viz.graph_view`** (matplotlib, no network): the structural graph of
`T`, and one arc-length profile per candidate showing height above ground, the
tube radius as a band, the tube+FOV outer radius, and the window boundaries.
The default `geographic` layout puts each node at its trajectory's midpoint, so
the structural graph also reads spatially.

**`csnav.viz.map_view`** (folium/Leaflet, self-contained HTML): the spatial
view, over real imagery.

* `trajectory_set_map` — all of `T` at once, one toggleable layer per
  trajectory, each with its own tube at whatever radius the CONOPS assigns it.
* `trajectory_map` — one trajectory in detail: its tube, the per-window visible
  footprints (each tooltipped with window id, arc span, AGL, and FOV radius),
  and the imagery tiles those footprints cover.
* `manifest_map` / `bundle_map` — a built manifest drawn where it actually
  sits: candidate roads with their off-track offsets, intersections, tiles.

The San Jose DPW 2025 imagery cache is available as a basemap layer (off by
default in the layer control), addressed through the same `/tile/{z}/{y}/{x}`
endpoint `ArcGISTileClient` fetches from — so what you see under a corridor is
what the pipeline will see.

### Tile scheme

`web_mercator_tile_info()` supplies the standard EPSG:3857 tiling scheme
(level 0 = one 256×256 world tile, halving each level), which is what San
Jose's caches are published against, so the tile layers work without a live
service request. Prefer a service's own `tileInfo` from `ArcGISTileClient`
where you have it — a service is free to publish a custom scheme.

`tiles_for_footprint` filters by real polygon intersection, not just the
bounding box: a corridor is a thin diagonal, and its bounding box holds several
times as many tiles as it actually covers. It refuses (rather than silently
truncating) a level that would enumerate more than `max_tiles` — at ~1.9 cm/px,
a whole route at the finest level is not something to materialize by accident.

## Running it

```bash
pip install -e ".[dev,viz]"
```

Render the figures and maps for the pilot scenario:

```bash
python scripts/visualize_trajectories.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --output-dir out/viz
```

Sweep a different tube radius without touching the config:

```bash
python scripts/visualize_trajectories.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --tube-radius 500 --output-dir out/viz_r500
```

Build and pin the manifests — from the live CSJ Streets layer, or from an
archived pull:

```bash
python scripts/build_manifests.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --output data/manifests/san_jose_downtown_r250.json \
    --map out/viz/manifests.html

python scripts/build_manifests.py \
    --scenario configs/scenarios/san_jose_downtown.yaml \
    --streets-geojson data/raw/csj_streets/downtown.geojson \
    --elevation \
    --output data/manifests/san_jose_downtown_r250.json
```

## Using it from Python

```python
from csnav.trajectory import load_scenario, ManifestBuilder, StaticStreetsSource

scenario = load_scenario("configs/scenarios/san_jose_downtown.yaml")
t_p = scenario.trajectory_set.primary

# Deterministic progress along the trajectory - the Phase 3 predict step's input.
state = t_p.point_at(1500.0)                       # 1500 m of arc length in
window = t_p.window_for_distance(1500.0, scenario.conops.window_length)

# The runtime lookup: pinned manifest in, candidate roads out. No live query.
from csnav.trajectory.manifest import ManifestBundle
bundle = ManifestBundle.load("data/manifests/san_jose_downtown_r250.json")
roads = bundle.by_window_id(window.window_id).query(fov_footprint)
```

## Tests

`tests/trajectory/`, `tests/geometry/test_fov.py`, `tests/geometry/test_shapes.py`,
`tests/viz/`, and `tests/scripts/test_visualize_trajectories.py`.

Arc length, cross-track distance, and FOV footprints are checked against
independent computations (`pyproj.Geod`, closed-form trigonometry) rather than
round trips through the code under test — the same approach
`tests/geometry/test_local_frame.py` takes for the ENU frame these build on.

The manifest-builder tests answer CLAUDE.md's testing priority 2 directly: the
fixture streets are laid out in the trajectory's own ENU frame at exact north
offsets (0, 150, 199, 260, 5000 m) from a due-east centerline, so which ones
belong in a 200 m tube is decidable by reading that list. (They are laid out in
ENU on purpose: a two-vertex line between two equal-*latitude* points is not a
constant-offset line — it sags a meter or two toward the equator over a few
kilometres, which is more than enough to make a 199 m street look like a 200 m
one.)

## What Phase 1 does not do

* **No FOV occlusion.** `FieldOfView` assumes flat ground and a straight-down
  camera. §2 wants terrain/building occlusion modelled eventually; that needs
  the 3DEP surface and a line-of-sight test, and it belongs with the AGL work,
  not the tube.
* **No variable-radius tubes.** §8 defers segment-dependent radii. The radius
  is per trajectory (with a per-trajectory override), constant along it.
* **No slice DAG.** `Predict x(t)`, the manifest ∩ FOV(t) lookup at a specific
  slice, and the filter loop are Phase 3. What Phase 1 provides for them is the
  arc-length state, the window lookup, and `LandmarkManifest.query`.
