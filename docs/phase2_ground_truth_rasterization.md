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

## Running it

### The default: DVC

```bash
uv sync --extra dev --extra dvc --extra viz
uv run dvc repro build_ground_truth check_ground_truth visualize_ground_truth
```

`params.yaml`'s `streets.layer_url`/`streets.where` pin the correct CSJ
Streets layer and filter (see "The CSJ Streets layer and filter" below);
`ground_truth.vintages` pins each imagery vintage to the street-network
snapshot that should label it, `foreach`-expanded into one
`build_ground_truth`/`check_ground_truth`/`visualize_ground_truth` stage
triple per vintage. `dvc repro` (with no stage names) reruns every stage
whose script/deps/params changed since the last run, in the right order -
this is the normal way to run the whole pipeline, including after a
`params.yaml` edit (a new vintage, a swept `default_width_m`, a corrected
streets filter). See the top-level README's "Running the pipeline" section
for `dvc dag`/`dvc exp run`/remote-storage setup shared across every phase.

### Running the scripts directly

Useful for a one-off run outside `--imagery-dir`/`--streets-geojson`'s
`params.yaml`-pinned paths, or while developing/debugging one stage. These
three run in this order, every time all three are needed - none of them
reruns an earlier one for you, so rerunning only the last step just
re-renders whatever the earlier steps already produced.

**1. Fetch the streets GeoJSON** (skip if reusing an existing, still-current
pull - see "Keeping a labels directory in sync" below for when it's stale):

```bash
uv run python scripts/fetch_csj_streets.py \
    --bbox -121.95 37.30 -121.85 37.36 \
    --output data/raw/csj_streets/downtown.geojson
```

No other flags are needed for CSJ San Jose - `--layer-url`/`--where`
default to the correct, pinned layer and filter (see below).

**2. Rasterize the label set:**

```bash
uv run python scripts/build_ground_truth.py \
    --imagery-dir data/raw/dpw_imagery/DPW_ImageryCached2025 \
    --streets-geojson data/raw/csj_streets/downtown.geojson \
    --output-dir data/ground_truth/current
```

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--imagery-dir PATH` | yes | - | Directory of `{level}_{row}_{col}.tif` tiles to label. |
| `--streets-geojson PATH` | yes | - | An archived pull from `fetch_csj_streets.py`, never a live query - see "Pairing imagery vintages with a matching street network" below. |
| `--output-dir PATH` | yes | - | Where to write each tile's 2-band GeoTIFF + JSON sidecar. |
| `--manifest PATH` | no | off | Restrict to one pinned `ManifestBundle`'s tiles instead of every tile under `--imagery-dir` - a smaller regional-sensitivity run rather than the full AOI. |
| `--default-width-m M` | no | 6.0 | Fallback width for a segment CSJ doesn't publish one for. |
| `--intersection-radius-m M` | no | 3.0 | Radius a derived intersection is rasterized as. |
| `--intersection-snap-m M` | no | 2.0 | Clustering tolerance for merging nearby junction points into one intersection instance. |
| `--overwrite` | no | off | Re-rasterize a tile whose label file already exists. |

**3. Check and visualize:**

```bash
uv sync --extra dev --extra viz

uv run python scripts/check_ground_truth.py --labels-dir data/ground_truth/current

uv run python scripts/visualize_ground_truth.py \
    --labels-dir data/ground_truth/current \
    --imagery-dir data/raw/dpw_imagery/DPW_ImageryCached2025 \
    --map out/viz_ground_truth/current_map.html \
    --gallery-dir out/viz_ground_truth/current_gallery
```

`check_ground_truth.py` (base install only) verifies every label's raster
matches its own JSON sidecar (shape, every instance id accounted for, no
orphan pixels) and reports the default-width fallback rate per tile, exiting
non-zero on any structural error - usable as a CI gate on
`build_ground_truth`'s output.

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--labels-dir PATH` | yes | - | A `build_ground_truth.py` output directory. |
| `--default-width-fraction F` | no | 0.5 | Warn on a tile where more than this fraction of its road segments fell back to the default width. |
| `--default-width-report PATH` | no | off | Stream a CSV of every OBJECTID/name/raw-CSJ-attributes that fell back to the default, across the whole label set. |
| `--report PATH` | no | off | Write the full structured check report as JSON. |

`visualize_ground_truth.py` (needs the `viz` extra) renders:

- **the review map** (`--map`) - every tile's footprint plus its
  road/intersection polygons, vectorized straight back out of the label
  rasters (so it shows exactly what a training loader would read), over San
  Jose imagery. Road tooltips show the OBJECTID, name, computed width, and
  every raw CSJ attribute for that segment.
- **the QA gallery** (`--gallery-dir`) - a self-contained static HTML page
  for paging through every tile quickly: a thumbnail grid drives a large
  viewer with two pixel-aligned images (imagery, and a transparent label
  overlay) under one opacity slider - "imagery only" / "overlay" / "labels
  only" are all the same slider, nothing extra to render - with arrow-key
  navigation and a per-tile "flag" checkbox (persisted in the page's own
  `localStorage`, exportable as a plain text list). The info panel lists
  every road instance in the current tile (OBJECTID, name, width, whether it
  fell back to the default) so going from "this tile looks off" to "here's
  the OBJECTID to check" never needs the map open too.

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--labels-dir PATH` | yes | - | A `build_ground_truth.py` output directory. |
| `--imagery-dir PATH` | yes | - | The same imagery directory the labels were built against. |
| `--map PATH` | no | off | Write the review map here. |
| `--gallery-dir PATH` | no | off | Write the QA gallery here. At least one of `--map`/`--gallery-dir` is required. |
| `--limit N` | no | off | Only process the first N tiles by tile key (see "Tile selection" below). |
| `--sample N` | no | off | Only process N tiles, evenly spaced across the sorted set - mutually exclusive with `--limit`. |
| `--overwrite` | no | off | Force a full re-render even where output files already exist - see "Keeping a labels directory in sync" below for when this matters. |

### Keeping a labels directory in sync

Fetching, building, and visualizing are three independent steps, each of
which reuses whatever it already has on disk unless told otherwise - so
each has its own condition for when a fresh run is actually needed:

- **Re-fetch streets** (`fetch_csj_streets.py`) whenever the live CSJ
  network has changed since the last pull, or the pinned
  layer/filter/schema itself changed. Nothing downstream re-fetches this
  for you.
- **Re-rasterize with `--overwrite`** (`build_ground_truth.py`) whenever
  `--streets-geojson`'s contents changed - without it, a tile whose label
  file already exists is left as-is.
- **Re-render with `--overwrite`** (`visualize_ground_truth.py`) whenever
  `--labels-dir`'s contents changed - the gallery's resumability (skip a
  tile whose 3 PNGs already exist, needed for a run spanning hundreds of
  thousands of tiles) has no way to tell that the underlying label data is
  different, only that a same-named file is already there. Reusing a
  `--gallery-dir` without `--overwrite` after rebuilding labels will keep
  showing the *old* images; this script logs a warning when it detects
  that situation.

**A `--labels-dir` and its `--imagery-dir` must stay scoped together.**
`build_ground_truth.py` only ever enumerates tiles from the *current*
`--imagery-dir` and writes/overwrites those - it never deletes a label file
for a tile outside that enumeration. Pointing it at a narrower or different
`--imagery-dir` than a previous run used (a different bbox, a different
zoom level) leaves the earlier run's tiles sitting in `--output-dir`
indefinitely, orphaned relative to the current imagery. That has two real
consequences:

- `check_ground_truth.py` audits the *entire* `--labels-dir`, orphaned
  tiles included, so its warning counts reflect however much of the
  directory is actually stale - not just the current imagery scope.
- `visualize_ground_truth.py`'s `--limit`/`--sample` draw only from labels
  that currently have a matching imagery file (see "Tile selection" below),
  so a limited/sampled run stays correct even with orphans present - but
  `--map` with no `--limit`/`--sample` still processes every label,
  orphaned or not.

There's no automatic pruning for this (deliberately not built
speculatively). If `--imagery-dir`'s scope has changed since a
`--labels-dir` was last built, delete and rebuild `--output-dir` from
scratch rather than reusing it across genuinely different imagery scopes.

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

### `rasterize()` is a pure function of geometry, not a live-fetching step

§7's UML gives `GroundTruthBuilder.rasterize(streets, tile, width, height,
transform) -> PanopticLabel`, with an edge to `LocalFrame` only - neither
`CSJStreetsClient` nor `ArcGISTileClient` is called at rasterization time:

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
  whether that layer has server-side archiving enabled hasn't been checked.
  Check `CSJStreetsClient.get_metadata()`'s `archivingInfo` field against the
  real service before relying on it; absent archiving support, the flag has
  no effect and the current network comes back regardless of the moment
  requested.
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
- `scripts/check_ground_truth.py --default-width-report out.csv` streams a
  CSV of every OBJECTID/name/raw-CSJ-attributes that fell back to the
  default, across the whole label set (one pass, one row per segment - safe
  at full-AOI scale) - the tool to reach for if a schema change ever makes
  `WIDTH_FIELD_CANDIDATES` stop matching again (its `attributes` column
  shows exactly what CSJ published for that segment, independent of whether
  any candidate field name matched it).

Like tube radius (CLAUDE.md core decision 4), this is a swept/versioned input
(`params.yaml`'s `ground_truth.default_width_m`), never a constant baked into
`rasterize()`'s call site.

**CSJ's actual field is `FOCWIDTH`** ("face-of-curb width", i.e.
curb-to-curb), first in `WIDTH_FIELD_CANDIDATES`. The other candidate names
(`WIDTH`, `ROADWIDTH`, etc.) are kept as fallbacks in case a
differently-sourced streets layer uses one of them instead.

### The CSJ Streets layer and filter

`scripts/fetch_csj_streets.py`'s `DEFAULT_LAYER_URL`/`DEFAULT_WHERE` (and
`params.yaml`'s matching `streets.layer_url`/`streets.where`, for the DVC
pipeline) pin this pipeline to:

```
layer:  https://geo.sanjoseca.gov/server/rest/services/OPN/OPN_OpenDataService/MapServer/60
where:  FEATURECLASS='StreetCenterline'
```

**"Streets" is an ambiguous substring in CSJ's catalog** - two other layers
under `OPN/OPN_OpenDataService` also match it, and neither is valid for this
pipeline: `Underground Designated Streets` (`MapServer/522`) has no width
field or `FEATURECLASS` at all, and `Paving Moratorium Streets`
(`MapServer/423`) isn't street centerlines either. That's why `--layer-url`
is pinned to an explicit URL by default rather than left to name-based
discovery, which has no way to prefer the right one of three
similarly-named matches.

**`MapServer/60` isn't only street centerlines, either** - its
`FEATURECLASS` field is a domain shared across many feature types on this
one layer (sanitary-sewer and storm-water infrastructure lines, parcels,
address points; see the schema reference below), not street-specific.
`STREETCLASS`/`FUNCTCLASS` are a different axis - they classify *within*
`StreetCenterline` rows (`AL`: Alley, `DW`: Driveway, `RA`: Ramp are all
still `StreetCenterline`), not between street and non-street features.
`FEATURECLASS='StreetCenterline'` is the filter that actually excludes the
non-street features, which is what "Overlapping/occluding segments" below
depends on.

If the catalog ever reorganizes again (a query error, a suspiciously sparse
pull, a schema change), three flags on `scripts/fetch_csj_streets.py`
re-derive the correct layer/field/value without guessing:

- `--list-layers` - print every layer whose name matches
  `--layer-name-contains` (not just the first discovery would pick), to
  compare every "Streets"-ish candidate.
- `--list-fields` (against a layer pinned with `--layer-url`) - print that
  layer's fields: name, type, alias, and every stored code + display label
  for a coded-value domain field, read from the untruncated `?f=json`
  metadata (CSJ's own REST HTML directory page truncates a long coded-value
  list as `...N more...`).
- `--distinct-values FIELD` - print every value a plain string field (no
  coded-value domain to read off from metadata alone, e.g. `DESIGNATION`/
  `DESCRIPTION`) actually contains.

Once a new correct layer/filter is confirmed, update
`DEFAULT_LAYER_URL`/`DEFAULT_WHERE` in `scripts/fetch_csj_streets.py` and
`params.yaml`'s `streets.layer_url`/`streets.where` together - both need to
carry the same values, since the script is also run directly outside the
DVC pipeline.

### Reference: the `Streets` layer schema (MapServer/60)

Recorded here so nobody has to re-derive it from the ArcGIS REST directory
by hand. Source:
`https://geo.sanjoseca.gov/server/rest/services/OPN/OPN_OpenDataService/MapServer/60`,
cross-checked against the City's own
[Open Data page](https://gisdata-csj.opendata.arcgis.com/datasets/CSJ::streets)
(35,811 records both places). Display field: `FULLNAME`. Geometry:
`esriGeometryPolyline`. `MaxRecordCount`: 2000 (why `CSJStreetsClient`
paginates). Supports `returnDistinctValues`/`orderByFields`/pagination -
what `--distinct-values`/`query_distinct_values` rely on.

Long municipality/zip-code lists are summarized rather than spelled out in
full below, since they aren't relevant to any filtering decision this
pipeline makes - get the full list with `--list-fields` if needed.

| Field | Type | Alias | Coded values |
| --- | --- | --- | --- |
| `OBJECTID` | esriFieldTypeOID | OBJECTID | |
| `FACILITYID` | esriFieldTypeString(20) | Centerline ID | |
| `INTID` | esriFieldTypeInteger | Integer ID | |
| `STREETMASTERID` | esriFieldTypeInteger | Street Master ID | |
| `FROMINTERID` / `TOINTERID` | esriFieldTypeInteger | From/To Intersection ID | |
| `FROMLEFT`/`TOLEFT`/`FROMRIGHT`/`TORIGHT` | esriFieldTypeInteger | Left/Right From/To Address | |
| `ADDRNUMTYPE` | esriFieldTypeString(20) | Address Number Type | `CONTIGUOUS`, `STD EVEN ODD`, `OTHER` |
| `FULLNAME` | esriFieldTypeString(125) | Full Street Name | |
| `ONEWAYDIR` | esriFieldTypeString(10) | One Way Indicator | `P`: From-To, `N`: To-From, `B`: Both |
| `MODELFLAG` | esriFieldTypeString(1) | Model Flag | `S`: Single, `M`: Median, `D`: Divided |
| `STREETCLASS` | esriFieldTypeString(20) | Street Class | `FY`: Freeway, `HY`: Highway, `EX`: Expressway, `RA`: Ramp, `MA`: Major Arterial, `MI`: Minor Arterial, `CO`: Collector, `RE`: Residential, `EA`: Easement, `DW`: Driveway, `PA`: Path, `RU`: Rural Access, `AL`: Alley - all 13 are legitimate `StreetCenterline` rows, not something to filter out |
| `FUNCTCLASS` | esriFieldTypeString(20) | Functional Class | `AR`: Freeway, `CA`: Highway, `LO`: Residential, `MA`: Major Arterial, `NC`: Neighborhood Collector |
| `SPEEDLIMIT` | esriFieldTypeSmallInteger | Speed Limit | |
| `PRIVATE` / `OFFICIAL` / `INCORPORATED` | esriFieldTypeString(3) | Private / Official / Incorporated | `Yes`, `No` |
| `MUNILEFT` / `MUNIRIGHT` | esriFieldTypeString(10) | Municipality on Left/Right | 20 South Bay cities/agencies (`SJ`: San Jose, `SC`: Santa Clara, `CO`: County, ...) |
| `ZIPLEFT` / `ZIPRIGHT` | esriFieldTypeString(5) | Zip on Left/Right | 67 South Bay ZIP codes |
| **`FOCWIDTH`** | esriFieldTypeDouble | **FOC Width** | the width field this pipeline reads (feet; face-of-curb, i.e. curb-to-curb) - see "Fallback roadway width" above |
| `ROWWIDTH` | esriFieldTypeDouble | ROW Width | right-of-way width (feet) - wider than `FOCWIDTH`, not currently read by `street_width_m` |
| **`FEATURECLASS`** | esriFieldTypeString(50) | **Feature Class** | 36 values total - a domain shared citywide across many feature types on this layer, not street-specific: `AddressPoint`, `CondoParcel`, `Parcel`, 14 sanitary-sewer infrastructure values (`ssCasing`, `ssGravityMain`, `ssManhole`, `ssPressurizedMain`, ...), 18 storm-water infrastructure values (`swCulvert`, `swGravityMain`, `swManhole`, `swPressurizedMain`, ...), and **`StreetCenterline`** - the value this pipeline filters to |
| `PLANCRT` / `PLANMOD` | esriFieldTypeString(25) | Plan Created/Modified | |
| `LASTUPDATE` / `CREATIONDATE` | esriFieldTypeDate | Last Update/Creation Date | |
| `NOTES` | esriFieldTypeString(255) | Notes | |
| `Shape` / `Shape_Length` | esriFieldTypeGeometry / Double | SHAPE / SHAPE_Length | |
| `RSN` | esriFieldTypeString(10) | Street RSN | |
| `PARCELID` | esriFieldTypeString(20) | PARCELID | |
| `ESNLEFT` / `ESNRIGHT` | esriFieldTypeString(5) | ESNLEFT / ESNRIGHT | |
| `FHWAFUNCTCLASS` | esriFieldTypeSmallInteger | FHWA Functional Class | `1`: Interstate, `2`: Other Freeway or Expressway, `3`: Other Principal Arterial, `4`: Minor Arterial, `5`: Major Collector, `6`: Minor Collector, `7`: Local |
| `RESPONSIBILITY` | esriFieldTypeString(10) | Responsible Agency | `SJ`: San Jose, `SC`: Santa Clara, `CO`: County, `ST`: State of California, `US`: Federal, `PR`: Private, ... 8 more agencies |

### Overlapping/occluding segments

Independent of *which* layer is queried, CSJ's Streets data can mix
non-street features (ramps, alleys, driveways, or - on `MapServer/60`
specifically - sanitary-sewer/storm-water infrastructure lines) in with
real street centerlines, and those can sit close enough to a real street to
spatially overlap it once buffered. `rasterize()` burns overlapping
polygons in a fixed order (currently: alphabetical by `segment_id`/
OBJECTID), and whichever one is drawn last **completely overwrites** the
earlier one's pixels in both the semantic and instance bands - so a real
street segment can end up with zero pixels of its own, entirely replaced by
an unrelated feature that happened to overlap it and draw afterward. The
symptom looks like a data-corruption/indexing bug from the map or gallery
alone: a visibly-distinct road segment reporting an unrelated OBJECTID, at
the default width. It is not an indexing bug - the OBJECTID pairing itself
stays correct throughout `rasterize()`; the overlap and one-sided overwrite
is real geometry.

**The fix is filtering at the query, not after rasterizing**:
`FEATURECLASS='StreetCenterline'` (see above) excludes non-street features
before they ever reach `rasterize()`, so nothing is left to occlude a real
street with. This applies to Phase 1 landmark manifests too, not just
ground truth: `scripts/build_manifests.py --streets-geojson` reads the same
pinned export, so a manifest built from an unfiltered pull carries the same
contamination in its candidate-road set. `check_label`'s "OBJECTID(s) never
rasterized, occluded by an overlapping segment" warning (and the matching
entry in `--report`'s JSON) is a safety net either way - a resurgence of it
is a signal the filter or layer pin needs re-checking.

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
   manifests. Every kind of geometry (tiles, roads, intersections) is one
   `folium.GeoJson` layer holding a whole `FeatureCollection`, never one
   Python object per shape - see "Memory-safe by design" below for why that
   matters at scale.
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

### Memory-safe by design

A full-AOI label set is easily hundreds of thousands of tiles, each
carrying two full-resolution rasters - per CLAUDE.md's "never materialize
the full per-tile dataset" convention, nothing in this pipeline holds more
than one tile's rasters, or more than one rendering object per whole
`FeatureCollection`, in memory at once:

- `ground_truth_review_map`/`build_gallery` consume their label sets as a
  **single-pass iterator**, not a pre-built list -
  `scripts/visualize_ground_truth.py`'s `_iter_labels` generator loads one
  label from disk, lets the caller extract what it needs (vectorized
  features for the map; rendered PNGs, written straight to disk, for the
  gallery), and only then loads the next.
- Every kind of geometry on the review map (tiles, roads, intersections) is
  one `folium.GeoJson` layer holding a whole `FeatureCollection`, never one
  heavyweight `folium.Polygon`/`Rectangle`/`CircleMarker` Python/Jinja
  object per shape - a full-AOI label set's shape count runs into the
  thousands (a road can vectorize into several disjoint pieces where an
  intersection cuts through it), and per-shape objects don't scale.

The gallery scales to any label-set size this way with no further changes
needed - each tile's images are written to disk and released before the
next tile is even loaded. The review map is different: it draws everything
in *one* map, so it still needs every tile's vectorized features (much
smaller than the rasters they came from, but not free) resident at once.
`scripts/visualize_ground_truth.py --limit N` / `--sample N` scope a run
down to N tiles (respectively: the first N by tile key, or N evenly spread
across the whole sorted set) when even that doesn't fit.

### Tile selection

Tile keys sort numerically by `(level, row, col)`, not as plain strings - a
mix of zoom levels in one `--labels-dir` would otherwise put a run of tiles
first for reasons unrelated to their actual level/position.

The map and the gallery apply `--limit`/`--sample` to two different pools,
though: the map's `N` comes from every label in `--labels-dir`, but the
gallery's `N` comes only from labels that currently have a matching file
under `--imagery-dir` (`_paths_with_imagery`, applied *before* the
limit/sample cut) - see "Keeping a labels directory in sync" above for why
a labels directory can otherwise carry tiles with no matching imagery at
all, which would otherwise make a limited/sampled gallery run pick tiles
that can never render.

### Gallery persistence across runs

`write_gallery` keeps a `tiles.json` manifest sidecar next to `index.html`:
every call reads whatever the gallery directory already lists, merges in
the tiles from the current call (keyed by `stem` - a fresh render for an
existing stem replaces its old entry), and writes the union back to both
files. A later, smaller/differently-scoped selection (a smaller `--limit`,
a different `--manifest`, even a call with zero matched tiles) only *adds
to* or *refreshes* the page, never shrinks it. `--overwrite` still forces a
fresh render of a given tile's images; it does not remove any other tile's
entry from the manifest.

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
