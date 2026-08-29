#!/usr/bin/env python3
"""Render the Phase 1 trajectory visualizations for a scenario config.

Produces, from one versioned scenario YAML (see ``configs/scenarios/``):

* ``trajectory_graph.html`` - the structural view: the candidate set ``T`` as a
  transition graph (routes as nodes, permitted hand-offs as edges, ``x_0`` as
  the entry), the routes those rules permit, and one arc-length profile per
  route showing height above ground, the tube radius, and the camera's ground
  reach against the manifest window boundaries.
* ``trajectory_set.html`` - the spatial view: every route with its tube and
  visible footprint, and every transition family with the region it can reach,
  over San Jose imagery.
* ``trajectory_<id>.html`` - one map per route: its tube at the configured
  radius, the per-window visible footprints, and the imagery tiles those
  footprints cover.
* ``transition_<source>__<target>.html`` - one map per transition rule: every
  sampled hand-off, where each initiates on the source, and the region the
  family sweeps.

Nothing here touches the network at build time (the maps reference basemap
tiles, which the browser fetches when the HTML is opened), so this runs against
a scenario config alone - no CSJ Streets pull and no manifest needed.

Example::

    uv run python scripts/visualize_trajectories.py \\
        --scenario configs/scenarios/san_jose_downtown.yaml \\
        --output-dir out/viz

Sweep a different tube radius without editing the config::

    uv run python scripts/visualize_trajectories.py \\
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
from csnav.trajectory.trajectory import X0_NODE  # noqa: E402
from csnav.viz.graph_view import (  # noqa: E402
    route_profile_figure,
    route_table_figure,
    transition_graph_figure,
    write_report,
)
from csnav.viz.map_view import (  # noqa: E402
    save_map,
    trajectory_map,
    trajectory_set_map,
    transition_map,
)

logger = logging.getLogger("visualize_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, type=Path, help="scenario YAML (configs/scenarios/*.yaml)")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory to write the report and maps into")
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
        "--transition-samples",
        type=int,
        default=None,
        help="initiation points sampled per transition rule; defaults to the scenario's conops.transition.samples",
    )
    parser.add_argument(
        "--no-tiles", action="store_true", help="skip the imagery-tile layer on the per-route maps"
    )
    parser.add_argument(
        "--no-transitions", action="store_true", help="skip the transition-family layers and per-rule maps"
    )
    parser.add_argument(
        "--no-imagery",
        action="store_true",
        help="omit the San Jose DPW imagery basemap layer (for maps reviewed without network access)",
    )
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
    if args.transition_samples is not None:
        conops = conops.with_transition_samples(args.transition_samples)
        logger.info("transition sampling overridden to %d initiations per rule", args.transition_samples)

    tile_level = args.tile_level if args.tile_level is not None else conops.tile_level
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = write_report(
        [
            transition_graph_figure(trajectory_set, conops),
            route_table_figure(trajectory_set),
            route_profile_figure(trajectory_set, conops),
        ],
        output_dir / "trajectory_graph.html",
        title=f"{trajectory_set.id} - trajectory set",
    )
    logger.info("wrote %s (%d routes permitted)", report, len(trajectory_set.route_paths()))

    set_map = trajectory_set_map(
        trajectory_set,
        conops,
        show_transitions=not args.no_transitions,
        include_imagery=not args.no_imagery,
    )
    logger.info("wrote %s", save_map(set_map, output_dir / "trajectory_set.html"))

    tile_info = None if (args.no_tiles or tile_level is None) else web_mercator_tile_info()
    for trajectory in trajectory_set.trajectories:
        tube = conops.tube_for(trajectory)
        fmap = trajectory_map(
            trajectory,
            tube,
            conops.window_length,
            camera=conops.camera,
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

    if args.no_transitions:
        return

    for rule in trajectory_set.transitions:
        if rule.source == X0_NODE:
            continue
        family = conops.transition.family(
            trajectory_set.by_id(rule.source), trajectory_set.by_id(rule.target), rule
        )
        destination = save_map(
            transition_map(trajectory_set, conops, rule, include_imagery=not args.no_imagery),
            output_dir / f"transition_{rule.source}__{rule.target}.html",
        )
        logger.info(
            "wrote %s (%d paths, %d screened out, turns %.0f-%.0f deg)",
            destination,
            len(family),
            family.rejected,
            *family.turn_range,
        )


if __name__ == "__main__":
    main()
