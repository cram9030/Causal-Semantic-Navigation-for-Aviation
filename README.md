# Causal Semantic Navigation for Aviation

A research prototype exploring whether Structural Causal Models can improve
the robustness and explainability of place-recognition-based visual
localization for low-altitude autonomous aircraft, using San Jose, CA as the
pilot AOI. See [`docs/INTEGRATION_PLAN.md`](docs/INTEGRATION_PLAN.md) for
the full architecture and rationale.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) — installs
resolve against the committed `uv.lock`, so every contributor and the dev
container get the identical, reproducible dependency graph rather than
whatever pip's resolver picks for a loose `>=` bound that day (torch in
particular moves fast enough that an unpinned install can silently jump to
a much larger/newer CUDA stack between one setup and the next).

```bash
uv sync --extra dev --extra viz --extra dvc
```

This creates `.venv` and installs the project into it editable, matching
`pip install -e ".[dev]"`. Activate it as usual (`source .venv/bin/activate`)
or prefix commands with `uv run` (e.g. `uv run pytest`) to skip activation.
`--extra viz` is folium/Plotly for the review maps and QA gallery;
`--extra dvc` is DVC itself, for the pipeline below; add `--extra ml`
(torch, torchvision, transformers, scikit-learn) for Phase 2's Mask2Former
fine-tuning, or `--all-extras` for everything.

Don't have `uv` installed? See the
[installation docs](https://docs.astral.sh/uv/getting-started/installation/)
— it's a single static binary, no Python bootstrap required. The project's
dependency metadata is still standard `pyproject.toml`, so `pip install
-e ".[dev]"` in a regular venv continues to work if you'd rather not adopt
uv; you just lose the lockfile's pinned, reproducible versions.

### Dev container (GPU-ready)

`.devcontainer/` defines a CUDA 12.4 dev container (VS Code Dev Containers /
GitHub Codespaces / any [Dev Containers spec](https://containers.dev/)
tool), with [uv](https://docs.astral.sh/uv/) and the
[Claude Code CLI](https://code.claude.com/docs/en/devcontainer) preinstalled.
It picks up a host GPU automatically when one is present
(`hostRequirements.gpu: "optional"`) and installs the `ml` extra needed for
fine-tuning Mask2Former in Phase 2 — see `docs/INTEGRATION_PLAN.md` §5. No
GPU is required for Phase 0/1 work; the container just runs CPU-only in
that case.

Open the repo in VS Code and choose **Dev Containers: Reopen in Container**,
or run `devcontainer up` from the [Dev Containers CLI](https://github.com/devcontainers/cli).
On first build, `.devcontainer/post-create.sh` installs the project
(`uv sync --extra dev --extra viz --extra ml --extra dvc`, from `uv.lock`)
and prints whether a GPU is visible.

### Tests

```bash
uv run pytest
```

## Running the pipeline

[DVC](https://dvc.org) (`dvc.yaml` + `params.yaml`) is the default way to
run this project's data pipeline - every stage below wraps one of
`scripts/*.py` as-is, with its inputs coming from `params.yaml` rather than
hardcoded, so a change to the AOI, a service URL, a CONOPS parameter, or the
CSJ Streets filter reproduces exactly the stages it affects and nothing
else. Reach for a direct `scripts/*.py` invocation only for a one-off run
outside the pinned AOI/config, or while developing a script itself - each
phase doc linked below documents that path too.

```bash
uv sync --extra dev --extra dvc --extra viz
uv run dvc repro          # (re)run every stage whose script/deps/params changed
uv run dvc dag             # show the pipeline graph
uv run dvc repro <stage>   # (re)run just one stage and its unmet dependencies
```

```
fetch_imagery -+                                      fetch_lidar    visualize_trajectories
                +-> build_ground_truth@<vintage> -+
fetch_streets -+-> build_manifests                +-> check_ground_truth@<vintage>
                                                    +-> visualize_ground_truth@<vintage>
```

| Stage | What it does | Details |
| --- | --- | --- |
| `fetch_imagery` | Pull historic San Jose aerial imagery tiles for the AOI. | [`docs/phase0_arcgis_tile_client.md`](docs/phase0_arcgis_tile_client.md) |
| `fetch_streets` | Pull CSJ Streets centerlines for the AOI. | [`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md) |
| `fetch_lidar` | Pull a ground-elevation raster for the AOI (USGS 3DEP). | [`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md) |
| `visualize_trajectories` | Render the trajectory set's structural/spatial review maps. | [`docs/phase1_trajectory_manifests.md`](docs/phase1_trajectory_manifests.md) |
| `build_manifests` | Build and pin the per-window landmark manifests. | [`docs/phase1_trajectory_manifests.md`](docs/phase1_trajectory_manifests.md) |
| `build_ground_truth@<vintage>` | Rasterize panoptic ground-truth labels for one imagery vintage. | [`docs/phase2_ground_truth_rasterization.md`](docs/phase2_ground_truth_rasterization.md) |
| `check_ground_truth@<vintage>` | Run automated structural/statistical checks over that label set. | [`docs/phase2_ground_truth_rasterization.md`](docs/phase2_ground_truth_rasterization.md) |
| `visualize_ground_truth@<vintage>` | Render that label set's review map and QA gallery. | [`docs/phase2_ground_truth_rasterization.md`](docs/phase2_ground_truth_rasterization.md) |

### Why `dvc repro` doesn't go stale the way a manual script sequence can

Every stage's script file, input files, and `params.yaml` values it reads
are declared as that stage's `deps`/`params`; every file/directory it
writes is declared as its `outs`. That gives `dvc repro` two guarantees a
sequence of manual `scripts/*.py` invocations doesn't have:

1. **A stage reruns when anything it declared a dependency on changes** -
   not just a `params.yaml` edit, but a code change to the script itself
   (`fetch_streets`'s `deps` include `scripts/fetch_csj_streets.py`), and
   the change cascades: `dvc repro` with no stage name walks the whole DAG
   and reruns every stage downstream of whatever changed, in dependency
   order, in one call - there's no "did I remember to also rebuild the
   labels, then the gallery" to get wrong.
2. **A stage's `outs` are reconciled to exactly what that stage last
   produced, on every `dvc repro` call - including one where nothing
   changed and the stage is just "cached, checking out outputs".** A file
   left in a DVC-tracked output directory that isn't part of what the
   stage actually wrote gets removed the next time DVC touches it, and a
   stage that reruns starts from an empty directory, not whatever was
   already there. `--overwrite`-style "skip if it already exists" logic in
   the scripts themselves is what direct invocation needs instead, since it
   has neither guarantee.

Both of these were verified directly (not assumed): a minimal two-stage DVC
pipeline, editing only the upstream script's source and running plain
`dvc repro`, reran that stage *and* the downstream one automatically and
rebuilt from the new output; and a file manually dropped into a tracked
output directory was removed by the very next `dvc repro`, even one where
the stage itself was reported as cached.

**The caveat**: both guarantees are properties of `dvc repro` itself, not
of the underlying scripts - a direct `scripts/*.py` invocation (see each
phase doc) gets neither one. That's why the scripts also pin their own
correct defaults independently of `params.yaml`, and why
`visualize_ground_truth.py` warns at runtime when it's about to reuse a
stale output directory outside of DVC - two different workflows, each
needing its own protection.

`build_manifests` depends on `fetch_streets`' pinned output (not the live
CSJ Streets layer) so the manifest it builds is reproducible from the exact
street geometry it was built against - the live layer refreshes weekly and
would not reproduce it (see `docs/INTEGRATION_PLAN.md` §3.3 "Pinning").
`visualize_trajectories` only needs the scenario config, so it has no such
dependency and can run standalone. `build_ground_truth`,
`check_ground_truth`, and `visualize_ground_truth` are each a `foreach` over
`params.yaml`'s `ground_truth.vintages` map, one entry per (imagery vintage
directory, matching street-network snapshot) pairing - adding another
vintage to the training set is a `params.yaml` edit, not a new stage.

Stage parameters (AOI bbox, service URLs and the pinned CSJ Streets
layer/filter, output paths, and - the swept CONOPS/altitude parameter
CLAUDE.md rule 4 calls for - `manifest.tube_radius_m`) live in
`params.yaml`, not hardcoded in the scripts - edit a value and `dvc repro`
reruns only the affected stage(s). For a one-off run without touching the
file, use `uv run dvc exp run --set-param manifest.tube_radius_m=500 ...`
(or `aoi.min_lon=-121.90`, etc.) to sweep a value without a code change.

`data/raw/`, `data/manifests/`, `out/`, and `*.tif` stay gitignored as
before - that's unaffected by and compatible with DVC, which tracks the
actual bytes separately via its own content-addressed cache, referenced
from git only through `dvc.yaml`/`dvc.lock`.

```bash
uv run dvc push / uv run dvc pull   # sync tracked data with the configured remote
```

The `.dvc/config` checked in here points the default remote at a **local
placeholder directory** (`data/dvc-storage/`, gitignored - not one of the
DVC-tracked working outputs like `data/raw/` or `data/manifests/`, just
where the "remote" cache lives locally) so a solo checkout works with zero
setup. It's kept inside the repo tree rather than a sibling directory so it
survives a devcontainer rebuild: `.devcontainer/devcontainer.json`
bind-mounts only the repo folder itself, so anything written outside it
(e.g. a sibling of the checkout under `/workspaces/`) lives in the
container's throwaway layer and is silently lost on rebuild. Before this is
used by more than one machine/collaborator, or at San Jose-imagery scale,
swap it for real object storage, e.g.:

```bash
uv run dvc remote add -d storage s3://<bucket>/csnav-dvc     # or gs://, azure://, etc.
```

(and add the matching extra - `dvc[s3]`, `dvc[gs]`, `dvc[azure]` - to
`pyproject.toml`'s `dvc` group in place of the plain `dvc` pin.)

Unimplemented past Phase 2's ground-truth builder (see
`docs/INTEGRATION_PLAN.md` §6): once `segmentation/` lands, stages for
Mask2Former training/checkpoints (fed by `build_ground_truth`'s output); and,
once `eval/` lands, `dvc.yaml` `metrics:`/`plots:` entries for the Phase 4
Integrity Risk / Time-to-Alert / Availability comparison, so a
`manifest.tube_radius_m` sweep produces a directly comparable leaderboard
across radii via `dvc exp show`.

## Phases

### Phase 0: data pipeline

Builds the ArcGIS clients used to collect San Jose's data sources from
`geo.sanjoseca.gov`'s ArcGIS Server, and the local ENU tangent-plane
conversion utilities every downstream metric geometry step builds on:

- Aerial imagery (`DPW_ImageryCached` and its historic vintages) - see
  [`docs/phase0_arcgis_tile_client.md`](docs/phase0_arcgis_tile_client.md).
- CSJ `Streets` centerlines and ground elevation (USGS 3DEP) - see
  [`docs/phase0_csj_streets_lidar.md`](docs/phase0_csj_streets_lidar.md).
- WGS84 <-> local ENU conversion - see
  [`docs/phase0_local_frame.md`](docs/phase0_local_frame.md).

### Phase 1: trajectory set, transitions, tubes, and precomputed manifests

Defines the candidate trajectory set `T`, the primary trajectory `t_p`, the
known start state `x_0`, the transitions permitted between routes, and the
RNP-style containment tube; builds the offline per-window landmark
manifests from those; and provides the visualization tools for reviewing
all of it. The pilot trajectory set and its CONOPS parameters live in
[`configs/scenarios/san_jose_downtown.yaml`](configs/scenarios/san_jose_downtown.yaml).
See [`docs/phase1_trajectory_manifests.md`](docs/phase1_trajectory_manifests.md).

### Phase 2: ground-truth panoptic label rasterization

Rasterizes CSJ street geometry/widths into panoptic ground-truth labels
over imagery tiles - the training data Mask2Former fine-tuning (the rest of
Phase 2, not yet built) will need - plus the tooling to actually trust a
large label set: automated structural checks, a geographic review map, and
a static QA gallery for paging through tiles by eye. See
[`docs/phase2_ground_truth_rasterization.md`](docs/phase2_ground_truth_rasterization.md).
