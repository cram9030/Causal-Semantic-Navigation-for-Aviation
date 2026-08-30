# Phase 2: ground-truth panoptic label rasterization, and its QA tooling

Phase 2 of `docs/INTEGRATION_PLAN.md` §5: *"Rasterize ground-truth panoptic
labels using CSJ street geometry/widths over San Jose imagery tiles"*, plus
the visualization and automated-check tooling needed to actually trust what
that rasterizer produced across a large tile set. Fine-tuning Mask2Former
and building its confusion matrix (§5's other Phase 2 bullet) are not part of
this - this is the ground-truth half only.

## What got built

```
src/csnav/data/ground_truth/
├── labels.py       # PanopticClass, SegmentInfo, PanopticLabel (2-band GeoTIFF + JSON sidecar)
├── rasterize.py    # GroundTruthBuilder.rasterize(streets, tile, width, height, transform) -> PanopticLabel
└── checks.py       # check_label / check_label_directory: structural + statistical sanity checks

src/csnav/viz/
├── ground_truth_view.py     # folium review map: tile footprints, vectorized roads/intersections
└── ground_truth_gallery.py  # static paged HTML QA gallery: imagery/label PNGs + opacity slider, flagging

scripts/build_ground_truth.py       # rasterize a label set (full AOI grid, or scoped to a pinned manifest)
scripts/check_ground_truth.py       # run the automated checks over a label set, exit non-zero on error
scripts/visualize_ground_truth.py   # render the review map and/or the QA gallery for a label set
```

As with Phases 0/1, this lives under `src/csnav/` as `csnav.data.ground_truth`
rather than a separate top-level `data/ground_truth/` tree - one installable
package (see integration plan §6's "Implementation note").

## Design decisions worth knowing

### Two classes, matching the manifest's own split

A label's semantic band has exactly three values: background, road, and
intersection (`PanopticClass`). This mirrors
`csnav.trajectory.manifest.ManifestLandmark` / `ManifestIntersection`, which
already split candidate roads from their junctions for the same reason
integration plan §3.4 gives: the Mask2Former match step detects road and
intersection instances separately. Intersections are *derived* from the
rasterized centerlines (a junction-clustering pass, the same algorithm
`ManifestBuilder._intersections` uses, adapted to one tile instead of one
trajectory window) rather than sourced from a separate CSJ dataset - CSJ
Streets only publishes centerlines.

### Two deliberate departures from the integration plan's UML sketch

§7's UML gives `GroundTruthBuilder.rasterize(streets, tile) -> PanopticLabel`
with edges to both `CSJStreetsClient` and `ArcGISTileClient`. The actual
signature is `rasterize(streets, tile, width, height, transform, ...)`, and
neither client is called at rasterization time:

1. **The pixel grid is supplied, not fetched.** `width`/`height`/`transform`
   are read by the caller from an already-fetched, already-reprojected
   imagery GeoTIFF (`scripts/fetch_historic_imagery.py`'s output). This keeps
   `rasterize()` a pure function of geometry - testable with a synthetic
   transform and no raster file at all - and *guarantees* pixel-for-pixel
   alignment with whatever imagery a training loader actually reads, rather
   than reconstructing a transform that could drift from it.
2. **Streets come from an in-memory list, not a live query.** Exactly the
   `StaticStreetsSource` reasoning `csnav.trajectory.manifest_builder`
   already applies: CSJ Streets refreshes weekly, and ground truth for a
   given imagery vintage should stay pinned to the street network as it
   stood for that vintage, not drift with the live layer. `streets` is
   loaded once from an archived GeoJSON pull and spatially indexed
   (`shapely.strtree.STRtree`) so `scripts/build_ground_truth.py` can filter
   per tile cheaply across a whole AOI.

### Tile scope: full AOI grid is the default; a manifest is an optional filter

Two ways to choose which tiles get labeled, both producing the same
`PanopticLabel` format:

- **Default - every tile under `--imagery-dir`.** Scans the
  `{level}_{row}_{col}.tif` files `fetch_historic_imagery.py` already writes.
  This is the one Mask2Former training actually needs: full AOI coverage,
  independent of any particular trajectory set.
- **`--manifest`** - restrict to `ManifestBundle.all_tiles()` for one pinned
  manifest. This is a secondary, narrower path: a quick regional-sensitivity
  check (does this CONOPS/tube-radius combination's actual coverage look
  right?) or a smaller smoke-test run, without rasterizing the whole AOI.
  Tile addressing (`level/row/col`) is shared between the two paths by
  construction - a manifest's `TileRef`s come from the same tile scheme
  `fetch_historic_imagery.py` addressed its files with, provided the same
  `tile_level` was used for both - so a manifest-scoped run resolves to
  imagery files already on disk under the same `--imagery-dir`.

### Pairing imagery vintages with a matching street network

CSJ Streets is a live, weekly-refreshed layer with no confirmed historic
archive. Training data spanning multiple imagery vintages (older captures
alongside the current cache) raises a real risk: pairing a historic image
with *today's* street network mislabels any road that has since changed.
Two things address this, with one caveat:

- **`params.yaml`'s `ground_truth.vintages`** is a map, not a single path:
  each entry pairs one imagery vintage's directory with the street-network
  snapshot that should label it. `dvc.yaml`'s `build_ground_truth` stage is a
  `foreach` over this map, so adding another vintage - imagery directory plus
  its matching streets snapshot - is a config edit, not a new stage.
- **`scripts/fetch_csj_streets.py --historic-moment`** forwards ArcGIS's
  `historicMoment` query parameter, the standard way to read an
  *archiving-enabled* layer as of a past edit moment.
  **This is not confirmed to do anything for CSJ Streets specifically** -
  whether that layer has server-side archiving enabled hasn't been checked
  (this codebase's own sandbox can't reach `geo.sanjoseca.gov`; see
  `docs/phase0_csj_streets_lidar.md` for the same live-service-verification
  gap on the LIDAR client). Check `CSJStreetsClient.get_metadata()`'s
  `archivingInfo` field against the real service before relying on it; absent
  archiving support, the flag has no effect and the current network comes
  back regardless of the moment requested.
- **Fallback if archiving isn't supported**: source a historic snapshot some
  other way (an external historic CSJ Streets export, or hand-edited
  geometry for a known road change) and point a `ground_truth.vintages` entry
  at that GeoJSON file directly - the builder doesn't care where a streets
  snapshot came from, only that it's a `StreetSegment` GeoJSON pull.

### Fallback roadway width

CSJ's width attribute isn't published for every segment
(`csnav.data.arcgis.streets.WIDTH_FIELD_CANDIDATES`). Where it's absent,
`GroundTruthBuilder` falls back to `default_width_m` (6 m: one travel lane
each way, a reasonable default for an urban local street) rather than
skipping the segment or leaving a hole in the label. Every `SegmentInfo`
records whether its width came from CSJ or the fallback
(`default_width_used`), so:

- `check_label`/`check_label_directory` warn when a tile leans heavily on the
  fallback (over 50% of its segments by default,
  `DEFAULT_WIDTH_WARN_FRACTION`) - a real data-quality signal (CSJ's width
  coverage is thin in that area), not a bug.
- The gallery surfaces the same count per tile, and the folium map's road
  tooltips mark a fallback-width road explicitly.

Like tube radius (CLAUDE.md core decision 4), this is a swept/versioned input
(`params.yaml`'s `ground_truth.default_width_m`), never a constant baked into
`rasterize()`'s call site.

### Storage format

One 2-band `uint32` GeoTIFF per tile (band 1 semantic class id, band 2
instance id), pixel-aligned to that tile's source imagery, plus a JSON
sidecar (`SegmentInfo` per instance id, plus provenance: which streets file
and which imagery file produced it). This is deliberately close to COCO
panoptic's own "id-encoded raster + segments_info" shape, so a later
Mask2Former training script can convert to that format without this module
reimplementing PNG id-packing or RLE encoding itself.

## Visualization and QA: two views, plus automated checks

Rasterizing thousands of tiles across a full AOI needs both a systematic
check (did anything break structurally?) and a way for a person to actually
look at a lot of them quickly:

1. **`scripts/check_ground_truth.py`** - automated, no human in the loop.
   Verifies each label is internally consistent with its own sidecar (shape
   match, every rasterized instance id has a `segments_info` entry and vice
   versa, no background pixel carries an instance id and no foreground pixel
   lacks one), and reports per-tile road/intersection pixel coverage and the
   default-width fallback rate. Exits non-zero on any `"error"`-severity
   issue, so it doubles as a CI gate on `build_ground_truth`'s output.
2. **`csnav.viz.ground_truth_view.ground_truth_review_map`** - the
   geographic view, one folium map. Road/intersection polygons are
   vectorized straight back out of each label's own semantic/instance bands
   (`rasterio.features.shapes`) rather than carried as separate stored
   geometry, so the map draws exactly what a training loader would read, not
   a reconstruction that could drift from it. Answers "does the rasterized
   geometry actually sit on the streets, across the whole set" - the same
   question `csnav.viz.map_view.manifest_map` answers for candidate-road
   manifests.
3. **`csnav.viz.ground_truth_gallery`** - the exhaustive per-tile view, a
   self-contained static HTML page (no server). Two pixel-aligned PNGs per
   tile (imagery, and a transparent-background label overlay) are stacked
   with a CSS-adjustable opacity slider in the browser, rather than one
   pre-baked blend - "imagery only" / "labels only" / "overlay at any
   strength" are all the same slider, no extra images to render. A
   thumbnail grid (one fixed-alpha blend each) drives which tile is open in
   the large viewer; arrow keys or prev/next buttons step through the whole
   set, and a per-tile "flag" checkbox persists to the page's own
   `localStorage` with a one-click export of the flagged list - built
   specifically so a reviewer can move through and exhaustively verify a
   large label set quickly, not just spot-check a handful.

`scripts/visualize_ground_truth.py` wires both views up from one labels
directory + its paired imagery directory.

## Running the tests

```bash
uv sync --extra dev --extra viz
uv run pytest tests/data/ground_truth tests/viz/test_ground_truth_view.py \
       tests/viz/test_ground_truth_gallery.py tests/scripts/test_build_ground_truth.py \
       tests/scripts/test_check_ground_truth.py tests/scripts/test_visualize_ground_truth.py
```

All of them run against synthetic tiles/streets (small, hand-checkable
geometry - a known-width road, a known crossing) - none needs real San Jose
data or network access.
