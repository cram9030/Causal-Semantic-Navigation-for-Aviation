#!/usr/bin/env python3
"""Pin each historic imagery vintage to its OWN highest (finest) available level.

Phase 0 data collection: ``fetch_historic_imagery.py`` fetches every
``DPW_Imagery*`` vintage for an AOI, and by default auto-detects each
service's own finest LOD with real cached coverage, live, on every run (see
``docs/phase0_arcgis_tile_client.md``'s "Auto-detected level" section). That
auto-detection is exactly what should be *pinned* rather than re-probed on
every ``dvc repro``: output filenames are ``<level>_<row>_<col>.tif``, and if
a rerun's live probe happens to pick a different level than an earlier run
did - plausible across a live server over hours or days - every
already-downloaded tile becomes invisible to that script's own per-tile
resume/skip check under the new level's filename prefix, and looks like a
full re-download even though the data is still sitting on disk.

Different vintages genuinely have different coverage - e.g. in this
project's AOI, every vintage has real coverage at level 21 *except*
``DPW_ImageryCached2011``, which only reaches 19. A single pinned level
either wastes resolution on every vintage that could go finer (if pinned to
the coarsest vintage's ceiling) or silently skips whole vintages (if pinned
to the finest vintage's level - ``--level``'s per-service coverage check
rejects the ones missing it and moves on, rather than failing the run, so
the omission is easy to miss). This script instead probes every
currently-discovered vintage the same way ``fetch_historic_imagery.py``
would (see :func:`csnav.data.arcgis.client.detect_finest_covered_level`,
shared by both) and pins each one to *its own* finest-covered level - the
highest resolution each vintage actually supports - as a
``{service-name: level}`` map in ``params.yaml``'s ``imagery.levels``.

This never touches the fetch itself - re-run it manually (not via
``dvc repro``, since writing to ``params.yaml`` from a stage that reads
``params.yaml`` would be circular) whenever the AOI changes or the catalog
gains a new historic vintage, then run ``dvc repro fetch_imagery`` as usual.
A vintage no longer discovered is dropped from the map; a vintage with no
coverage at any level is left out and reported separately, since pinning a
level can't fix that.

Example::

    uv run python scripts/discover_imagery_levels.py \\
        --bbox -121.95 37.30 -121.85 37.36
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.arcgis.catalog import ArcGISCatalog, DEFAULT_BASE_URL, extract_year  # noqa: E402
from csnav.data.arcgis.client import (  # noqa: E402
    ArcGISTileClient,
    DEFAULT_COVERAGE_SAMPLE_SIZE,
    detect_finest_covered_level,
)
from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.data.arcgis.projections import extent_4326_to_3857  # noqa: E402

logger = logging.getLogger("discover_imagery_levels")

# Matches the `  levels:` mapping under the top-level `imagery:` section of
# params.yaml, plus every immediately-following 4-space-indented entry line
# (see update_params_levels below) - two-space indent on `levels:` is what
# distinguishes it from any other `levels` key, and this block is treated as
# fully machine-owned: a refresh replaces every entry, it doesn't merge.
_LEVELS_BLOCK_RE = re.compile(r"(?m)^  levels:.*(?:\n    .*)*")


def discover_levels(
    catalog: ArcGISCatalog,
    aoi_4326: Extent,
    name_contains: str,
    coverage_sample_size: int,
) -> dict[str, int | None]:
    """Probe every matching service and return ``{service_name: level_or_None}``.

    Keyed by the short ``ref.name`` (e.g. ``DPW_ImageryCached2025``) - the
    same name each vintage's output subfolder is written under
    (``<output-dir>/<service-name>/``), and what ``fetch_historic_imagery.py``
    looks up in a ``--levels-file`` map. ``None`` means the service has no
    cached coverage for this AOI at *any* level - a different problem than
    "which level to pin", which this script can't fix by choice of level
    (see the warning this produces in ``main``).
    """
    services = catalog.discover_imagery_services(name_contains=name_contains)
    if not services:
        logger.error("no imagery services matched %r under the catalog", name_contains)
        raise SystemExit(1)

    aoi_3857 = extent_4326_to_3857(aoi_4326)
    results: dict[str, int | None] = {}
    for ref in tqdm(services, desc="services", unit="service"):
        service_url = catalog.service_rest_url(ref)
        client = ArcGISTileClient(service_url)
        meta = client.get_metadata()
        if not meta.supports_tiles:
            logger.warning("skipping %s: not a cached tile service", ref.full_name)
            continue

        level = detect_finest_covered_level(
            client, meta.tile_info, aoi_3857, coverage_sample_size,
            on_level_rejected=lambda lod, ref=ref: logger.info(
                "%s: level %d has no sampled coverage, trying coarser", ref.full_name, lod.level,
            ),
        )
        results[ref.name] = level
    return results


def update_params_levels(params_file: Path, levels: dict[str, int]) -> None:
    """Rewrite the ``imagery.levels`` mapping in ``params_file``, in place.

    Deliberately a targeted text substitution rather than a
    ``yaml.safe_load``/``yaml.dump`` round-trip: this file is full of
    hand-written comments that a generic YAML dump would silently discard.
    The ``levels:`` block itself is fully machine-owned (see
    ``_LEVELS_BLOCK_RE``) - a refresh replaces every entry with the current
    discovery results, sorted by year, rather than merging, so a vintage
    that's disappeared from the catalog doesn't linger. Verifies the result
    still parses to the expected value before leaving it in place.
    """
    text = params_file.read_text()
    matches = re.findall(r"(?m)^  levels:", text)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one '  levels:' line under imagery: in {params_file}, found "
            f"{len(matches)} - refusing to guess which one to update. Edit imagery.levels by "
            "hand instead."
        )

    ordered = sorted(levels.items(), key=lambda kv: extract_year(kv[0]) or -1)
    new_block = "  levels:\n" + "".join(f"    {name}: {level}\n" for name, level in ordered)
    new_text = _LEVELS_BLOCK_RE.sub(lambda _match: new_block.rstrip("\n"), text, count=1)
    params_file.write_text(new_text)

    reloaded = yaml.safe_load(new_text)
    if reloaded.get("imagery", {}).get("levels") != levels:
        # Something about the file's structure didn't round-trip the way we
        # assumed - restore the original rather than leave a corrupt file.
        params_file.write_text(text)
        raise SystemExit(
            f"post-write check failed: imagery.levels did not come back as expected after "
            f"editing {params_file}; reverted the file - edit it by hand instead."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="ArcGIS REST services directory root"
    )
    parser.add_argument(
        "--name-contains", default="DPW_Imagery",
        help="substring used to match imagery service names (matches all historic vintages)",
    )
    parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    parser.add_argument(
        "--coverage-sample-size", type=int, default=DEFAULT_COVERAGE_SAMPLE_SIZE,
        help=f"tiles to sample per level when checking coverage (default: {DEFAULT_COVERAGE_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--params-file", type=Path, default=Path("params.yaml"),
        help="params.yaml to update in place with the discovered imagery.levels map (default: params.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the discovered per-vintage levels without writing",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    catalog = ArcGISCatalog(base_url=args.base_url)
    aoi = Extent(xmin=args.bbox[0], ymin=args.bbox[1], xmax=args.bbox[2], ymax=args.bbox[3], wkid=4326)
    results = discover_levels(catalog, aoi, args.name_contains, args.coverage_sample_size)

    covered = {name: level for name, level in results.items() if level is not None}
    uncovered = sorted(name for name, level in results.items() if level is None)

    print(f"\n{'service':<30} {'year':>6} {'level':>6}")
    for name, level in sorted(results.items(), key=lambda kv: extract_year(kv[0]) or -1):
        year = extract_year(name)
        print(f"{name:<30} {year if year is not None else '?':>6} {level if level is not None else 'NONE':>6}")

    if uncovered:
        logger.warning(
            "%d service(s) have NO cached coverage for this AOI at any level - no level "
            "pins this, they will still be skipped entirely by fetch_historic_imagery.py: %s",
            len(uncovered), ", ".join(uncovered),
        )

    if not covered:
        logger.error("no service has any coverage for this AOI - nothing to pin")
        raise SystemExit(1)

    print(f"\n{len(covered)} vintage(s) pinned to their own finest-covered level, {len(uncovered)} with no coverage.")

    if args.dry_run:
        print(f"\n--dry-run: not writing to {args.params_file}")
        return

    update_params_levels(args.params_file, covered)
    print(f"\nWrote imagery.levels ({len(covered)} entries) to {args.params_file}")


if __name__ == "__main__":
    main()
