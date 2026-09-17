# CLAUDE.md

Guidance for Claude Code (or any agentic coding session) working in this repository.

## Project

Causal Semantic Navigation for Aviation — a research prototype exploring whether Structural
Causal Models can improve the robustness and explainability of place-recognition-based visual
localization for low-altitude (200–4000 ft AGL) autonomous aircraft, compared to standard
landmark-based / SLAM approaches.

Full architecture and rationale: see `docs/INTEGRATION_PLAN.md` in this repo — it is the source
of truth for design decisions and should be read before making structural changes. It includes
the UML class diagram for the module layout.

**Read the integration plan before changing module boundaries, DAG structure, or the filter
loop.** Most "why is it built this way" questions are answered there, not in code comments.

## Core design decisions (do not silently relitigate these)

1. **Estimation target:** the causal model estimates the aircraft's current state
   `x_t = (lat_t, lon_t, height_t)`. Roads and intersections are **evidence/measurement-model
   nodes**, not the thing being estimated. If a change starts treating segmentation output as
   the final answer rather than an input to a state estimate, stop and re-read §3.1 of the
   integration plan.
2. **Coordinates:** WGS84 (EPSG:4326) for all storage, interop, and reported values. Metric
   operations (buffers, tube containment, FOV projection) happen in a local ENU tangent-plane
   frame (`geometry/local_frame.py`) and convert back to WGS84 before being stored or returned.
   Never do distance/area math directly on raw lat/lon degrees.
3. **Trajectory-conditioned DAG:** the causal graph is chain-structured across time slices, not
   one global graph solved end to end. Each slice's Markov blanket is `x(t-1)` and `x(t+1)`.
   Landmark manifests are precomputed offline per trajectory window and **pinned** for the
   flight-planning cycle — the runtime "possible roads" step is a lookup, never a live spatial
   query against the full street network.
4. **Tube radius is a configurable input, not a constant.** It will be swept across CONOPS/
   altitude scenarios. Never hardcode a radius value inside the manifest builder or containment
   check — it must be a parameter passed in from trajectory/CONOPS configuration.
5. **`Predict x(t)` is a deterministic mechanism**, not a fitted/statistical one. The first
   implementation selects a valid point along the active trajectory (or transition path) toward
   the next waypoint under a constant-velocity assumption. Track arc-length progress along the
   trajectory as an explicit value (not just derived from lat/lon) — it's what decides which
   trajectory window's precomputed manifest applies next.
6. **DoWhy-GCM fits mechanisms within a slice; it does not chain slices itself.** The
   slice-to-slice filter loop (`causal_model/filter_loop.py`) is custom code that feeds the
   previous posterior in as evidence and reads the MAP query result back out. Don't try to make
   GCM hold the whole flight's DAG at once.
7. **Transitions are generated, never authored.** A transition is not known before flight: it may
   initiate at any arc length along the route being flown, not only at a waypoint. A scenario
   declares a `TransitionRule` (which hand-offs are permitted, and optionally where they may
   begin); `trajectory/transition.py` generates the family of Hermite paths it admits. The family
   — and the region it sweeps — is the object of interest, not any single path: any point between
   two trajectories is a valid state mid-transition. A return to `x_0` is an ordinary candidate
   route ending at `x_0`, not a special edge, and composite routes are paths through the
   transition graph rather than separate declarations.

## Repository layout

```
data/acquisition/     San Jose imagery (DPW_ImageryCached2025), CSJ Streets, LIDAR clients — all normalize output to EPSG:4326
data/ground_truth/    Rasterize CSJ street geometry (+ widths) over imagery tiles into panoptic training labels
geometry/             WGS84 <-> local ENU conversions; the only place metric geometry math should live
trajectory/           Trajectory / TrajectorySet / Waypoint definitions, TubeModel, generated TransitionModel, offline ManifestBuilder, scenario config
viz/                  Interactive transition-graph + profile figures (Plotly) and tube/transition/tile/manifest maps (folium)
segmentation/         Mask2Former training + inference, confusion-matrix uncertainty quantification
scene_graph/          Per-slice builder: predict -> possible-roads lookup -> Mask2Former match
causal_model/         Slice DAG spec, DoWhy-GCM mechanism fitting, the slice-chaining filter loop
baseline_slam/        Modified SLAM baseline (RANSAC removed) for the integrity-risk comparison
eval/                 Integrity Risk / Time-to-Alert / Availability comparison between the two approaches
docs/                 Integration plan, UML diagram, and other design references
```

## Data sources (pilot AOI: City of San José, CA)

- **Imagery:** `DPW_ImageryCached2025` — ArcGIS cached tile service, native EPSG:3857
  (reproject to 4326 at fetch time; the projection is datum-free, just a formula), ~1.9 cm/px
  at max zoom.
- **Streets:** CSJ `Streets` FeatureServer/Hub dataset — centerlines with width/lane attributes,
  refreshed weekly. Manifests built from this are pinned per flight-planning cycle and are not
  rebuilt mid-cycle on the weekly refresh.
- **Elevation:** San Jose's Imagery & Elevation LIDAR product — used for AGL correction and FOV
  occlusion modeling, not just visualization.

## Coding conventions

- Python, type-hinted throughout. Every public function/class gets a docstring stating units
  (meters vs. degrees, seconds vs. arbitrary time index) — this codebase mixes coordinate
  frames and unit ambiguity is the most likely source of silent bugs.
- Any function doing geometric math must state in its docstring which frame it operates in
  (WGS84 vs. local ENU) and must not silently accept the wrong one.
- Config (tube radius, trajectory sets, AOI bounds, model checkpoints) lives in versioned config
  files, not hardcoded constants — several of these (notably tube radius) are explicitly meant
  to be swept across experiments.
- Prefer `networkx.DiGraph` for both trajectory graphs and the slice DAG spec — matches
  DoWhy-GCM's expected graph structure and avoids a translation layer.
- **Never materialize the full per-tile dataset into memory or into one output artifact.**
  `data/ground_truth/` (and any future per-tile output — segmentation predictions, confusion-matrix
  inputs, etc.) is one file pair per imagery tile at full-AOI scale, which means hundreds of
  thousands of files, not hundreds. Any function that iterates such a dataset — building a review
  map, a QA gallery, a training manifest, anything — must consume it as a single-pass
  iterator/generator (load one tile, use it, let it be garbage-collected, load the next), never
  `list(...)`/`sorted(...)` the whole thing into memory first "to make it easier to compute a
  summary" or "because a helper needed `len()`". This already caused a real incident: an
  eager `[PanopticLabel.load(...) for ... in labels_dir.glob(...)]` in the ground-truth
  visualizer got the process SIGKILL'd (exit 137, no traceback — a killed process can't raise
  one) on a real ~260,000-tile label set, on top of a *separate* problem in the same code
  (one heavyweight rendering object per shape instead of one batched layer) that had already
  been "fixed" without anyone noticing the eager-list loop underneath it. See
  `docs/phase2_ground_truth_rasterization.md`'s "Memory" section for the full story and the
  before/after numbers.
  A batch view over the whole set (a map, a gallery page) is a second, distinct limit even
  once loading is properly lazy: it still has to hold *something* per tile (a vectorized
  feature, a DOM thumbnail) to draw them all together, and that stops being practical — for a
  human to review, or for a browser to render — long before Python runs out of memory. Give
  large-N entry points an explicit, cheap way to select a bounded subset (e.g. `--limit`/
  `--sample`, or scoping to a smaller precomputed set like one trajectory's manifest) rather
  than assuming "the whole label set" is ever the right amount of data to hand to one view.

## Testing priorities

Given where the risk actually is (per the integration plan's open-items list):
1. CRS/reprojection correctness (EPSG:3857 -> 4326, WGS84 <-> local ENU round-trips) — test this
   before anything downstream.
2. Manifest builder: given a trajectory + tube radius, does the candidate landmark set match a
   hand-checked expectation for a small test trajectory?
3. Slice chaining: does `x(t-1) -> predict -> match -> posterior -> x(t+1)` actually decouple
   from other slices as intended (i.e., does changing slice `t+5`'s inputs leave slice `t`'s
   output unchanged)? This is the load-bearing tractability assumption — worth a regression test.
4. Confusion-matrix-derived noise priors actually flow into the GCM mechanism fitting, not just
   computed and discarded.
5. Any function that processes `data/ground_truth/` (or another per-tile output at that scale):
   does it actually run in ~O(1) memory per tile, or does it just work at whatever tile count the
   test fixture happened to use? A test against a handful of synthetic tiles will not catch an
   eager `list(...)`/`[... for ... in glob(...)]` load — this has already happened once. Prove it
   with a generator that would raise if consumed more than once, or a synthetic set large enough
   (thousands, not tens) to make an accidental full-materialization show up in timing/memory.

## What not to do

- Don't reintroduce a live/global spatial query in the runtime path — possible-roads lookups
  must hit the precomputed manifest, not CSJ Streets directly, at inference time.
- Don't collapse the state-estimation framing back into "predict the correct label" — that was
  an earlier misunderstanding this design deliberately corrected.
- Don't fit `Predict x(t)` statistically — it's specified as deterministic.
- Don't hardcode a tube radius anywhere; it's a swept experimental parameter.
- Don't author transition geometry into a scenario or a `TrajectorySet` — declare the rule and let
  the model generate it (`TrajectorySet` rejects a `role: transition` trajectory outright).
- Don't plot graph structure in lat/lon. The transition graph is a structural view; geography
  belongs on the folium maps.
- Don't load a whole per-tile dataset (`data/ground_truth/` or similar) into a `list` before
  processing it, and don't hand a whole such dataset to one rendering/aggregation call without a
  way to bound how much of it that call actually consumes. Both have already caused a real
  `SIGKILL` on real data — see the "coding conventions" bullet above.
