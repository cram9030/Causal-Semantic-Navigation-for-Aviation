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

**`WIDTH_FIELD_CANDIDATES` started as a guessed lookup list, not a verified
CSJ schema contract** - none of its original entries were confirmed against
the live service (this codebase's sandbox can't reach `geo.sanjoseca.gov`).
While it was missing the real field name, *every* road fell back to the
flat `default_width_m`, and the visible symptom was roads that all rendered
at roughly the same width - uniformly too narrow (specifically, like the
default's "one travel lane each way, no parking", since that's exactly what
the default is) rather than varying with each street's actual pavement
width. This happened in practice against a real full-AOI label set, and was
tracked down to a real, confirmed answer:

**CSJ's actual field is `FOCWIDTH`** ("face-of-curb width" - the curb-to-curb
width), now listed first in `WIDTH_FIELD_CANDIDATES`. The earlier guessed
names (`WIDTH`, `ROADWIDTH`, etc.) are kept as fallbacks in case a
differently-sourced streets layer uses one of them, but for the live CSJ
`Streets` layer, `FOCWIDTH` is the one that actually matches. Three tools
exist for catching and diagnosing a mismatch like this in the future
(a schema change, or a new source layer with yet another field name):

- Every `SegmentInfo` carries the source segment's full raw CSJ
  `attributes` dict, not just this module's `width_m`/`name` interpretation
  of it - so it's always possible to see exactly what CSJ published for a
  given `OBJECTID`, independent of whether `WIDTH_FIELD_CANDIDATES`
  happened to match it.
- The folium review map's road tooltips show the OBJECTID and every raw CSJ
  attribute for that segment (`csnav.viz.ground_truth_view`), and the
  gallery's per-tile info panel lists every road instance's OBJECTID, name,
  width, and default-width flag as a small table
  (`csnav.viz.ground_truth_gallery`) - both let a reviewer go straight from
  "this looks wrong" to "here's the exact OBJECTID and what CSJ says about
  it" without leaving the page.
- `scripts/check_ground_truth.py --default-width-report out.csv` streams a
  CSV of every OBJECTID/name/raw-attributes that fell back to the default,
  across the whole label set (one pass, one row per segment - safe at
  full-AOI scale) - the fastest way to confirm the hypothesis: if that CSV
  is nearly the whole dataset, and its `attributes` column consistently
  shows *some* width-shaped field under a name not in
  `WIDTH_FIELD_CANDIDATES`, that's the fix - add the real field name to the
  candidates list in `csnav/data/arcgis/streets.py` and rebuild.

**Resolved**: CSJ's real field is `FOCWIDTH` ("face-of-curb width", i.e.
curb-to-curb), now first in `WIDTH_FIELD_CANDIDATES` - confirmed against a
layer that does carry it. See the caveat about *which* layer that is, right
below.

### "Streets" is an ambiguous name in CSJ's catalog

`scripts/fetch_csj_streets.py` resolves the layer to query by a name
substring match (`ArcGISCatalog.find_layer`, first match wins) rather than a
hardcoded service/layer id, per `docs/phase0_csj_streets_lidar.md`'s
discovery-over-hardcoding rationale - CSJ's catalog has reorganized this
before. In practice, more than one layer's name under
`OPN/OPN_OpenDataService` contains "Streets", and which one `find_layer`'s
substring match resolves to has already changed between sessions with no
code change on this side.

Three flags on `scripts/fetch_csj_streets.py` exist specifically to pin this
down deliberately instead of guessing:

- `--list-layers` prints every layer whose name matches
  `--layer-name-contains` (not just the first), so every "Streets"-ish
  candidate is visible at once.
- `--list-fields` (against a layer pinned with `--layer-url`) prints that
  layer's actual fields - name, type, alias, and every stored code + display
  label for a coded-value domain field, in full (not truncated the way
  CSJ's own REST HTML directory page abbreviates a long coded-value list as
  `...N more...`).
- `--distinct-values FIELD` prints every value a plain string field (no
  coded-value domain to read off from metadata alone - e.g. `DESIGNATION`/
  `DESCRIPTION`) actually contains.

**Resolved**: `--list-layers` finds exactly 3 matches under
`OPN/OPN_OpenDataService`:

| Layer name | URL | What it is |
| --- | --- | --- |
| `Streets` | `.../MapServer/60` | **The correct one.** Full-attribute street centerlines - has `FOCWIDTH`, `FEATURECLASS`, `STREETCLASS`/`FUNCTCLASS`. 35,811 records, matching the City's own [Open Data page for this layer](https://gisdata-csj.opendata.arcgis.com/datasets/CSJ::streets) record count exactly. |
| `Underground Designated Streets` | `.../MapServer/522` | The sparser layer a prior session's substring search drifted onto - no width field, no `FEATURECLASS`, fields limited to `OBJECTID`/`FACILITYID`/`INTID`/`CENTERLINEID`/`FULLNAME`/`DESIGNATION`/`DESCRIPTION`/`NOTES`/`LASTUPDATE`/`CREATIONDATE`. Not street centerlines in the sense this pipeline needs. |
| `Paving Moratorium Streets` | `.../MapServer/423` | Uncharacterized beyond its name - almost certainly not centerlines either. |

`params.yaml`'s `streets.layer_url` now pins `.../MapServer/60` explicitly,
so a future catalog reorganization can't silently resolve to one of the
other two again.

### Reference: the `Streets` layer schema (MapServer/60)

Recorded here so nobody has to re-derive it from the ArcGIS REST directory
by hand again. Source:
`https://geo.sanjoseca.gov/server/rest/services/OPN/OPN_OpenDataService/MapServer/60`,
cross-checked against the City's own
[Open Data page](https://gisdata-csj.opendata.arcgis.com/datasets/CSJ::streets)
(35,811 records both places; data last updated 2025-04-28, published
2020-08-27 as of this writing). Display field: `FULLNAME`. Geometry:
`esriGeometryPolyline`. `MaxRecordCount`: 2000 (why `CSJStreetsClient`
paginates). Supports `returnDistinctValues`/`orderByFields`/pagination -
what `--distinct-values`/`query_distinct_values` rely on.

Coded-value domains below are copied from the REST *HTML* directory view,
which truncates long lists as `...N more...` - treat the ones marked
truncated as incomplete and get the full list with
`--layer-url https://geo.sanjoseca.gov/server/rest/services/OPN/OPN_OpenDataService/MapServer/60 --list-fields`
(that tool reads the untruncated `?f=json` metadata, not this HTML page).

| Field | Type | Alias | Coded values |
| --- | --- | --- | --- |
| `OBJECTID` | esriFieldTypeOID | OBJECTID | |
| `FACILITYID` | esriFieldTypeString(20) | Centerline ID | |
| `INTID` | esriFieldTypeInteger | Integer ID | |
| `STREETMASTERID` | esriFieldTypeInteger | Street Master ID | |
| `FROMINTERID` / `TOINTERID` | esriFieldTypeInteger | From/To Intersection ID | |
| `FROMLEFT`/`TOLEFT`/`FROMRIGHT`/`TORIGHT` | esriFieldTypeInteger | Left/Right From/To Address | |
| `ADDRNUMTYPE` | esriFieldTypeString(20) | Address Number Type | `CONTIGUOUS`, `STD EVEN ODD`, `OTHER` (complete) |
| `FULLNAME` | esriFieldTypeString(125) | Full Street Name | |
| `ONEWAYDIR` | esriFieldTypeString(10) | One Way Indicator | `P`: From-To, `N`: To-From, `B`: Both (complete) |
| `MODELFLAG` | esriFieldTypeString(1) | Model Flag | `S`: Single, `M`: Median, `D`: Divided (complete) |
| `STREETCLASS` | esriFieldTypeString(20) | Street Class | `FY`: Freeway, `HY`: Highway, `EX`: Expressway, **...10 more (truncated)** - renderer symbolizes `EX`/`FY`/`MA`/`MI`/`CO`/`RE` (Expressway/Freeway/Major Arterial/Minor Arterial/Collector/Residential) distinctly, so the remaining ~10 likely include non-arterial classes (alley/ramp/driveway are plausible candidates - **not confirmed**, use `--list-fields` for the untruncated list) |
| `FUNCTCLASS` | esriFieldTypeString(20) | Functional Class | `AR`: Freeway, `CA`: Highway, `LO`: Residential, **...2 more (truncated)** |
| `SPEEDLIMIT` | esriFieldTypeSmallInteger | Speed Limit | |
| `PRIVATE` / `OFFICIAL` / `INCORPORATED` | esriFieldTypeString(3) | Private / Official / Incorporated | `Yes`, `No` (complete) |
| `MUNILEFT` / `MUNIRIGHT` | esriFieldTypeString(10) | Municipality on Left/Right | `SJ`: San Jose, `SC`: Santa Clara, `MI`: Milpitas, **...17 more (truncated)** |
| `ZIPLEFT` / `ZIPRIGHT` | esriFieldTypeString(5) | Zip on Left/Right | **...64 more (truncated)** |
| **`FOCWIDTH`** | esriFieldTypeDouble | **FOC Width** | the confirmed width field (feet; face-of-curb, i.e. curb-to-curb) - see "Fallback roadway width" above |
| `ROWWIDTH` | esriFieldTypeDouble | ROW Width | right-of-way width (feet) - wider than `FOCWIDTH`, not currently read by `street_width_m` |
| `FEATURECLASS` | esriFieldTypeString(50) | Feature Class | `AddressPoint`, `CondoParcel`, `Parcel`, **...33 more (truncated)** - a broad domain shared across many CSJ layers, not street-specific; whether a value like `StreetCenterline` is among the 33 more, and whether every row on *this* layer actually carries it (vs. some other value for a ramp/alley/driveway), is **not confirmed** - check with `--distinct-values FEATURECLASS` before filtering on it |
| `PLANCRT` / `PLANMOD` | esriFieldTypeString(25) | Plan Created/Modified | |
| `LASTUPDATE` / `CREATIONDATE` | esriFieldTypeDate | Last Update/Creation Date | |
| `NOTES` | esriFieldTypeString(255) | Notes | |
| `Shape` / `Shape_Length` | esriFieldTypeGeometry / Double | SHAPE / SHAPE_Length | |
| `RSN` | esriFieldTypeString(10) | Street RSN | |
| `PARCELID` | esriFieldTypeString(20) | PARCELID | |
| `ESNLEFT` / `ESNRIGHT` | esriFieldTypeString(5) | ESNLEFT / ESNRIGHT | |
| `FHWAFUNCTCLASS` | esriFieldTypeSmallInteger | FHWA Functional Class | `1`: Interstate, `2`: Other Freeway or Expressway, `3`: Other Principal Arterial, **...4 more (truncated)** |
| `RESPONSIBILITY` | esriFieldTypeString(10) | Responsible Agency | `SJ`: San Jose, `SC`: Santa Clara, `CO`: County, **...11 more (truncated)** |

`STREETCLASS`, `FUNCTCLASS`, and `FEATURECLASS` are the three live
candidates for the "Overlapping/occluding segments" fix below - whichever
one cleanly separates real street centerlines from ramps/alleys/driveways
(if any of them do) is what `streets.where` should filter on, once
confirmed with `--list-fields`/`--distinct-values` against the untruncated
metadata rather than this truncated table.

### Overlapping/occluding segments

Independent of *which* layer is queried, CSJ's Streets data can mix
non-street features (ramps, alleys, driveways) in with real street
centerlines, and those can sit close enough to a real street to spatially
overlap it once buffered. `rasterize()` burns overlapping polygons in a
fixed order (currently: alphabetical by `segment_id`/OBJECTID), and
whichever one is drawn last **completely overwrites** the earlier one's
pixels in both the semantic and instance bands - so a real street segment
can end up with zero pixels of its own, entirely replaced by an unrelated
feature that happened to overlap it and draw afterward. The symptom looks
exactly like a data-corruption/indexing bug from the map or gallery alone:
many visibly-distinct road segments all reporting one single, seemingly
unrelated OBJECTID, at the default width (since the occluding feature is
often something the width-field logic has no reason to have a sensible
width for). It is not an indexing bug - reproduced synthetically with many
overlapping segments and confirmed the OBJECTID pairing itself stays
correct throughout `rasterize()`; the overlap and one-sided overwrite is
real.

**The right fix is filtering at the query, not after rasterizing** - once
the correct layer and its actual classification field/values are confirmed
(see above), set `--where`/`params.yaml`'s `streets.where` to exclude
non-street features there, so nothing is left to occlude anything with.
There is deliberately no default filter beyond "every feature" until that's
confirmed - a guessed filter already caused one failed fetch (an ArcGIS
query error from a field/value that turned out not to exist on the layer
being queried), and a silently-wrong-but-not-erroring guess would be worse.
**This applies to Phase 1 landmark manifests too**, not just ground truth:
`scripts/build_manifests.py --streets-geojson` reads the exact same pinned
export, so a manifest built from a contaminated pull has the same
contamination in its candidate-road set and should be rebuilt once a clean
pull exists. `check_label`'s "OBJECTID(s) never rasterized, occluded by an
overlapping segment" warning (and the matching entry in `--report`'s JSON)
is a safety net either way - it should be rare once the right layer/filter
are pinned down, not an every-tile occurrence.

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

   Every kind of geometry (tiles, roads, intersections) is one
   `folium.GeoJson` layer holding a whole `FeatureCollection`, never one
   Python object per shape. This matters at real scale: a full-AOI label set
   is hundreds of tiles over a dense city street network, and a road can
   even vectorize into several disjoint polygon pieces where an intersection
   cuts through it (see `rasterize.py`'s docstring), so the shape count for
   a few hundred tiles can run into the thousands. An earlier version drew
   one `folium.Polygon`/`Rectangle`/`CircleMarker` per shape - that many
   heavyweight Python/Jinja objects held in memory at once (well before
   `.render()` ever runs) was enough to get the whole process SIGKILL'd by
   the OOM killer on a memory-constrained devcontainer, with no traceback at
   all to point at why. Benchmarked against a synthetic ~5,000-instance
   label set, the `GeoJson`-batched version used about 30% less peak memory
   and ran about 3x faster than the one-object-per-shape version, and the
   gap widens with instance count.
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

### Memory: labels are loaded one tile at a time, never as one big list

The GeoJson-batching fix above addressed the number of *rendering* objects
held at once, but not a second, larger problem: `visualize_ground_truth.py`
originally loaded every `PanopticLabel` in the label set - both
full-resolution rasters included - into one Python `list` before doing
anything with it, so both the map and the gallery ran with the *entire*
label set's rasters resident in memory simultaneously regardless of how
efficiently each one got rendered. On a real full-AOI label set (hundreds of
tiles) this alone was enough to get the process SIGKILL'd, independent of
the GeoJson fix.

`ground_truth_review_map` and `build_gallery` now both consume `labels`/
`labels_and_imagery` as a **single-pass iterator** rather than requiring a
pre-built list - `scripts/visualize_ground_truth.py`'s `_iter_labels`
generator loads one label from disk, lets the caller extract what it needs
(vectorized features for the map; rendered PNGs, written straight to disk,
for the gallery), and only then loads the next, so at most one tile's
rasters are ever alive at once. Benchmarked at a more realistic tile
resolution (512x512, 256 tiles): peak memory dropped from ~700 MB (the
eager-list version) to ~185 MB (the streaming version) - about a 3.8x
reduction, and the eager version's cost scales with the *whole* label set's
total pixel count, so the gap only widens for a larger or higher-resolution
AOI.

The gallery scales to any label-set size this way with no further changes
needed - each tile's images are written to disk and released before the
next tile is even loaded. The review map is different: it draws everything
in *one* map, so it still needs every tile's vectorized features (much
smaller than the rasters they came from, but not free) resident at once.
`scripts/visualize_ground_truth.py --limit N` / `--sample N` scope a run
down to N tiles (respectively: the first N by tile key, or N evenly spread
across the whole sorted set) when even that doesn't fit - the flags apply to
both the map and the gallery in one run, though the gallery rarely needs
them.

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
