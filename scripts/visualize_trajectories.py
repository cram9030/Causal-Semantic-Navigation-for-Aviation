#!/usr/bin/env python3
"""Render the Phase 1 trajectory visualizations for a scenario config.

Produces, from one versioned scenario YAML (see ``configs/scenarios/``):

* ``trajectory_graph.png`` - the graph of the candidate set ``T``: candidates
  as nodes, transition corridors as edges, ``x_0`` as the entry node, with a
  per-trajectory arc-length profile showing height, tube radius, and FOV ground
  radius against the manifest window boundaries.
* ``trajectory_set.html`` - an interactive map of every trajectory with its own
  tube corridor and visible footprint, over San Jose imagery.
* ``trajectory_<id>.html`` - one map per trajectory: its tube at the configured
  radius, the per-window visible footprints, and the imagery tiles those
  footprints cover.

Nothing here touches the network at build time (the maps reference basemap
tiles, which the browser fetches when the HTML is opened), so this runs against
a scenario config alone - no CSJ Streets pull and no manifest needed. To draw
built manifests instead, see ``scripts/build_manifests.py --map``.

Example::

    python scripts/visualize_trajectories.py \\
        --scenario configs/scenarios/san_jose_downtown.yaml \\
        --output-dir out/viz

Sweep a different tube radius without editing the config::

    python scripts/visualize_trajectories.py \\
        --scenario configs/scenarios/san_jose_downtown.yaml \\
        --tube-radius 500 --output-dir out/viz_r500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from csnav.data.arcgis.tiles import web_mercator_tile_info  # noqa: E402
from csnav.trajectory.config import load_scenario  # noqa: E402
from csnav.viz.graph_view import trajectory_set_figure  # noqa: E402
from csnav.viz.map_view import save_map, trajectory_map, trajectory_set_map  # noqa: E402

logger = logging.getLogger("visualize_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, type=Path, help="scenario YAML (configs/scenarios/*.yaml)")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory to write figures and maps into")
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=None,
        help="override the scenario's tube radius, in meters - the sweep entry point (§8)",
    )
    parser.add_argument(
        "--tile-level",
        type=int,
        default=None,
        help="imagery cache level for the 'tiles in view' layer; defaults to the scenario's conops.tile_level",
    )
    parser.add_argument(
        "--no-tiles", action="store_true", help="skip the imagery-tile layer on the per-trajectory maps"
    )
    parser.add_argument(
        "--no-imagery",
        action="store_true",
        help="omit the San Jose DPW imagery basemap layer (for maps reviewed without network access)",
    )
    parser.add_argument("--dpi", type=int, default=140, help="raster DPI for the PNG figure")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable DEBUG-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    scenario = load_scenario(args.scenario)
    if args.tube_radius is not None:
        scenario = scenario.with_tube_radius(args.tube_radius, label=f"sweep_{args.tube_radius:g}m")
        logger.info("tube radius overridden to %.1f m", args.tube_radius)

    trajectory_set, conops = scenario.trajectory_set, scenario.conops
    tile_level = args.tile_level if args.tile_level is not None else conops.tile_level
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = trajectory_set_figure(trajectory_set, conops)
    graph_path = output_dir / "trajectory_graph.png"
    figure.savefig(graph_path, dpi=args.dpi, bbox_inches="tight")
    logger.info("wrote %s", graph_path)

    set_map = trajectory_set_map(trajectory_set, conops, include_imagery=not args.no_imagery)
    logger.info("wrote %s", save_map(set_map, output_dir / "trajectory_set.html"))

    tile_info = None if (args.no_tiles or tile_level is None) else web_mercator_tile_info()
    for trajectory in trajectory_set.trajectories:
        tube = conops.tube_for(trajectory)
        fmap = trajectory_map(
            trajectory,
            tube,
            conops.window_length,
            field_of_view=conops.field_of_view,
            tile_info=tile_info,
            tile_level=None if tile_info is None else tile_level,
            include_imagery=not args.no_imagery,
        )
        destination = save_map(fmap, output_dir / f"trajectory_{trajectory.id}.html")
        logger.info(
            "wrote %s (tube %.0f m, %d windows)",
            destination,
            tube.radius,
            len(trajectory.windows(conops.window_length)),
        )


if __name__ == "__main__":
    main()
