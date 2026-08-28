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
├── trajectory.py        # Trajectory, TrajectorySet, TransitionRule, TrajectoryWindow
├── transition.py        # TransitionModel: generates the family a rule admits
├── tube.py              # TubeModel: containment + corridor/envelope geometry
├── coverage.py          # visible footprint (tube + camera reach), TileRef, AGL providers
├── manifest.py          # LandmarkManifest, ManifestBundle, JSON pinning
├── manifest_builder.py  # the offline builder, and StaticStreetsSource
└── config.py            # Scenario / ConopsConfig, versioned YAML loading

src/csnav/geometry/
├── fov.py               # FieldOfView -> ground footprint extents
├── camera.py            # SensorPose, AttitudeMargin, Camera -> ground reach
└── shapes.py            # WGS84 <-> ENU conversion for whole shapely geometries

src/csnav/viz/
├── style.py             # shared palette so both views agree on colours
├── graph_view.py        # Plotly: the transition graph, routes, profiles
├── map_view.py          # folium: corridors, transition families, tiles, manifests
├── window_selector.py   # the per-window Leaflet control (see "Isolating windows")
└── static/
    └── window_selector.js   # its behaviour, linted and unit-tested under node

configs/scenarios/san_jose_downtown.yaml   # the pilot T / t_p / x_0 / CONOPS
scripts/visualize_trajectories.py          # render the report and maps
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

### What `window_length_m` is, and why windows never come out as slivers

A trajectory is chopped into **windows** along its arc length, and §3.3 builds
one precomputed landmark manifest per window. `window_length_m` is the target
length of one such window, so it trades manifest count against how much ground
each manifest has to cover: halve it and you get twice as many manifests, each
with a tighter footprint and a shorter candidate-road list for the runtime
lookup to sift.

`Trajectory.windows(window_length)` divides a trajectory into
`round(length / window_length)` **equal** windows rather than walking a fixed
stride and leaving a remainder. A 2 km trajectory at a 1200 m window length
gives two 1000 m windows, not 1200 m + an 800 m tail — a sliver window's
manifest would be nearly identical to its neighbour's while still costing a
build and a lookup entry. So the configured value is a target, not an exact
stride.

Window ids (`"<trajectory_id>:0000"`) are stable across rebuilds of the same
trajectory and window length, because they are the key a pinned manifest is
looked up under at runtime.

### Transitions are generated, not authored

This is the load-bearing change from the first cut of Phase 1, and it follows
from what is actually known before flight. A transition **is not** known in
advance: it may begin anywhere along the route being flown, at any arc length,
not only at a waypoint. What *is* known is that it is permitted, and roughly
where it could lead. So the flight plan carries a `TransitionRule` — "`t_p` may
hand off to `t_alt_north`" — and `TransitionModel` generates the family of
paths that rule admits.

* **Initiation** is any arc length `s` on the source. A rule may narrow the
  domain (`initiate_from_m` / `initiate_to_m`) for a target that only becomes
  valid past some point.
* **Arrival** is the first target waypoint *ahead* of where the initiation
  point projects onto the target's ground track (`Trajectory.project`). Using
  the projection rather than raw proximity is what guarantees forward progress:
  nearest-by-distance would happily pick a waypoint the aircraft has already
  passed and fly backwards to it.
* **Geometry** is a cubic Hermite spline matching the source's direction of
  travel at `s` and the target's at the arrival waypoint, with endpoint
  tangents of `tangent_gain x` the straight-line distance between them — so the
  curve's shape is scale-free across short and long transitions. This is a
  placeholder for a dynamics model: it is smooth and heading-continuous, and it
  knows nothing about turn rate, bank limits, or airspeed.
* **The family, not the path, is the object.** Initiation is continuous in `s`,
  so the aircraft may be anywhere in the region the family sweeps, not only on
  one of the sampled curves. `TransitionFamily.reachable_footprint` is that
  region — the union of the sampled paths' tubes — and it is what the map
  shades. `conops.transition.samples` is a fidelity knob on how finely the
  continuum is stood in for, not a modelling parameter.

`TrajectorySet` therefore holds only candidate routes and rejects an authored
`role: transition` trajectory outright, with a message pointing at the rule.
`to_networkx()` puts routes on the nodes and rules on the edges, plus the
reserved node `x0` for the start state. `networkx.DiGraph` per CLAUDE.md,
matching what the Phase 3 slice DAG spec will use.

### Returns are routes, and composite routes fall out of the graph

A return to `x_0` is not a special kind of edge — it is a candidate trajectory
whose last waypoint is `x_0`, one per outbound route
(`t_return_via_p`, `t_return_via_north`, `t_return_via_east` in the pilot). The
ordinary transition machinery gets the aircraft onto one, so an abort part way
out needs nothing declared: the hand-off may initiate anywhere along the
outbound leg.

The payoff is that **composite routes need no declaration at all**. "Fly `t_p`,
divert to `t_alt_north`, then take the northern return" is a path through the
transition graph, and `TrajectorySet.route_paths()` enumerates every such path
from an entry to a terminal. The pilot scenario declares six routes and eight
rules; the five routes that implies — including the three
outbound-divert-return compositions — are derived, not written down.

### The turn screen cannot tell a wanted reversal from an unwanted one

`max_turn_deg` drops initiations demanding a sharper heading change than the
limit at either end. It is what keeps a near-orthogonal alternate from
generating diverts no aircraft could fly.

It has a limit worth knowing, found while implementing it: **it measures
heading change, so a deliberate turn-around looks exactly like a mistake.**
Diverting onto a return route is a reversal — departure turns of 150-180° are
the normal case there, not a defect — and the same number comes out when the
target waypoint has effectively been passed. In the pilot scenario the
conops-level screen sits at 120° and the three return edges override it to
180°, which puts the tension in the config where it can be seen. A screen that
actually separated the two cases would have to test the generated curve's
curvature against a turn radius, not test an angle. That is the natural next
step if the screen needs to do real work.

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

### The camera has a pose, and an allowance for not being level

§3.3 scopes a manifest to what could be seen *from any state inside the tube*,
so the search footprint is the tube corridor grown by how far the camera can
actually see at that window's worst case:

```
footprint = buffer(window centerline, tube_radius + max_ground_reach(window))
```

`Camera` is `FieldOfView` plus two things the FOV alone cannot express:

* **`SensorPose`** — where the sensor is mounted relative to the body frame
  (X forward, Y right, Z down). All zeros is nadir, which is what the first
  prototype flies, and `Camera.ground_reach` then reduces *exactly* to
  `FieldOfView.ground_radius`. A forward- or side-looking mounting is a config
  change, not a rewrite: the corner rays are rotated by the pose and
  intersected with flat ground, so an off-nadir camera's reach comes out right.
  Watch the signs — nose-up pitch swings a belly camera's view forward, and
  right-wing-down roll swings it *left*, because the belly turns away from the
  dropped wing. Both are what the right-hand rule gives and what a real
  aircraft does.
* **`AttitudeMargin`** — how far off level the aircraft is allowed to be when
  the footprint is sized. Real vehicles bank and pitch, most of all around
  waypoints where the turns are, and that swings the footprint well off nadir.
  The margin carries a cruise allowance and a larger `maneuver_*` allowance
  that applies within `maneuver_radius_m` of a waypoint, selected via
  `Trajectory.distance_to_nearest_waypoint`. Reach is maximized over every
  ±roll, ±pitch combination, so it bounds the footprint rather than estimating
  it. **Everything defaults to zero**, so the first proof of concept sizes
  footprints as if the aircraft were level; the mechanism is there to switch
  on.

`max_ground_reach` samples across a window and takes the maximum, because both
inputs vary along it — height above ground, and the margin, which widens near
waypoints. Taking the maximum (not the mean) keeps the manifest a superset of
what any state in the window could see. The reach is a heading-free circular
bound: it is the largest distance from the nadir point to any FOV corner, so it
holds whatever the aircraft's heading is, which matters because heading inside
the tube is not constrained a priori.

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

Two views, deliberately different in kind, and both interactive HTML.

**`csnav.viz.graph_view`** (Plotly, self-contained, no network): the
**structural** view.

* `transition_graph_figure` — routes as nodes, permitted hand-offs as edges,
  `x_0` as the entry. Positions come from `layered_layout`: the x coordinate is
  how many transitions it takes to reach a node, the y spreads a layer. Hover
  an edge for its initiation window, how many paths it admits, and the turns
  those demand.
* `route_table_figure` — the routes the rules permit, derived not declared.
* `route_profile_figure` — one row per route: height above ground, the tube
  radius as a band, the tube + camera ground reach, and the window boundaries.

Node position here is a **graph layer, not a latitude**. The first cut plotted
each route's midpoint at its own lat/lon, which produced something that was
neither a map nor a graph: it looked geographic but showed only three points,
and the edges said nothing about where a transition actually goes. Geography is
the map's job.

**`csnav.viz.map_view`** (folium/Leaflet, self-contained HTML): the spatial
view, over real imagery.

* `trajectory_set_map` — all of `T` at once, one toggleable layer per route
  with its own tube, plus one layer per transition rule holding the family it
  admits: a dashed curve per sampled initiation, a marker where each begins on
  the source, and the shaded union of their tubes.
* `transition_map` — one transition family in detail, over its source and
  target, with each path's initiation arc length, arrival waypoint, and turn
  angles in the tooltip.
* `trajectory_map` — one route in detail: its tube, the per-window visible
  footprints (each tooltipped with window id, arc span, AGL, and camera ground
  reach), and the imagery tiles those footprints cover.
* `manifest_map` / `bundle_map` — a built manifest drawn where it actually
  sits: candidate roads with their off-track offsets, intersections, tiles.

### Isolating windows

Windows overlap by design — adjacent ones meet at a shared boundary and every
corridor is round-capped — so drawing a trajectory's whole run of them at once
in one colour produces a chain of blobs rather than a readable sequence. Two
things address that.

**Each window is its own layer, driven by a `WindowSelector` panel.** Folium's
own layer control is flat, so listing 36 window layers there would only move the
problem; the selector is a collapsible tree instead. Expand a trajectory to get
its windows, each labelled with its index and arc-length span; tick the ones you
want, use `all`/`none` per trajectory, or hit `solo` on a window to see it
alone. Window layers are created with `control=False` so they stay out of
folium's control, which keeps the tube, centerline, and basemap toggles there
uncluttered.

Where a window carries more than one kind of geometry, those are **categories** —
footprints, candidate roads, intersections, imagery tiles — toggled across all
windows at once from a row at the top of the panel. A layer is drawn when its
window is selected *and* its category is enabled, so "every window's roads, no
footprints" and "window 3, everything about it" are both one or two clicks.
`manifest_map` groups a window's landmarks with its footprint for exactly this
reason: soloing a window isolates what that window's manifest actually contains,
not just where it sits.

**Consecutive windows alternate their styling.** Even with everything shown, odd
windows are drawn with a lighter fill and a dashed outline (`_window_style`), and
where colour is not already carrying "which trajectory" they also alternate
between two shades (`viz.style.window_shade`). That makes a run of windows
countable rather than blobby.

The control's behaviour is a real `.js` file rather than a template string, so
it can be linted and unit-tested: `tests/viz/test_window_selector.js` covers
solo, `all`/`none`, and the window x category mask under node, and pytest runs
that suite (skipping if node is absent). Its spec is emitted as a JavaScript
object literal rather than JSON, because layer references have to come out as
the bare variable names folium declared — every string in it is escaped so a
label can break out neither of the literal nor of the `<script>` element.

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

The visualization extra is Plotly and folium; both emit self-contained HTML
that opens without network access (the maps still fetch basemap tiles when
you open them, and `--no-imagery` drops that layer too).

Render the report and maps for the pilot scenario:

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
from csnav.trajectory import load_scenario

scenario = load_scenario("configs/scenarios/san_jose_downtown.yaml")
trajectory_set, conops = scenario.trajectory_set, scenario.conops
t_p = trajectory_set.primary

# Deterministic progress along the trajectory - the Phase 3 predict step's input.
state = t_p.point_at(1500.0)                       # 1500 m of arc length in
window = t_p.window_for_distance(1500.0, conops.window_length)

# What the rules permit, without declaring any of it.
trajectory_set.route_paths()
# ('t_p', 't_alt_north', 't_return_via_north'), ...

# Divert to the northern alternate from 1500 m along t_p.
rule = next(r for r in trajectory_set.transitions if r.key == ("t_p", "t_alt_north"))
path = conops.transition.path(t_p, trajectory_set.by_id("t_alt_north"), 1500.0, rule=rule)
path.arrival_index, path.departure_turn      # where it rejoins, and how hard it turns

# The region the aircraft may occupy during any hand-off on that edge.
family = conops.transition.family(t_p, trajectory_set.by_id("t_alt_north"), rule)
reachable = family.reachable_footprint(conops.tube_for(path.trajectory))
```

## Tests

`tests/trajectory/`, `tests/geometry/test_fov.py`,
`tests/geometry/test_camera.py`, `tests/geometry/test_shapes.py`,
`tests/viz/`, and `tests/scripts/test_visualize_trajectories.py`.

Arc length, cross-track distance, and camera footprints are checked against
independent computations (`pyproj.Geod`, closed-form trigonometry) rather than
round trips through the code under test — the same approach
`tests/geometry/test_local_frame.py` takes for the ENU frame these build on.

The transition tests pin the formalization's claims rather than its output: the
arrival waypoint never moves backwards as initiation advances; the generated
curve's departure and arrival headings converge on the source's and target's as
resolution rises (a sampled curve only approaches its own endpoint tangents, so
the test checks the error shrinks and lands under a degree); a transition onto
where the aircraft already is produces nothing; and the reachable footprint
covers every sampled path with room to spare.

The manifest-builder tests answer CLAUDE.md's testing priority 2 directly: the
fixture streets are laid out in the trajectory's own ENU frame at exact north
offsets (0, 150, 199, 260, 5000 m) from a due-east centerline, so which ones
belong in a 200 m tube is decidable by reading that list. (They are laid out in
ENU on purpose: a two-vertex line between two equal-*latitude* points is not a
constant-offset line — it sags a meter or two toward the equator over a few
kilometres, which is more than enough to make a 199 m street look like a 200 m
one.)

## What Phase 1 does not do

* **No FOV occlusion.** `Camera` assumes flat ground below the aircraft. §2
  wants terrain/building occlusion modelled eventually; that needs the 3DEP
  surface and a line-of-sight test, and it belongs with the AGL work, not the
  tube.
* **No dynamics in the transition model.** The Hermite spline is smooth and
  heading-continuous, and that is all. Turn radius, bank limits, and airspeed
  are not modelled, and the turn-angle screen is a stand-in for them, with the
  limitation noted above.
* **No manifests over transition families.** `ManifestBuilder.build_set` covers
  the candidate routes. Building manifests over the region a family sweeps is
  the obvious next step and is deliberately not done yet.
* **No variable-radius tubes.** §8 defers segment-dependent radii. The radius
  is per trajectory (with a per-trajectory override), constant along it.
* **No slice DAG.** `Predict x(t)`, the manifest ∩ FOV(t) lookup at a specific
  slice, and the filter loop are Phase 3. What Phase 1 provides for them is the
  arc-length state, the window lookup, and `LandmarkManifest.query`.
