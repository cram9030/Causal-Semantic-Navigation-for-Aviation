"""Static views of a trajectory set: the graph of ``T``, and per-trajectory profiles.

Two figures, both matplotlib:

* :func:`plot_trajectory_graph` draws the trajectory set as its
  ``networkx.DiGraph`` - candidate trajectories as nodes, transition corridors
  as edges, ``x_0`` as the entry node. This is the structural view of ``T``:
  what the aircraft can switch to from where, which is the reachability
  structure the tube containment assumption (integration plan §3.2) is defined
  over.
* :func:`plot_trajectory_profile` draws one trajectory against arc length -
  height, tube radius, and FOV ground radius - with the manifest window
  boundaries marked, so the discretization the manifests are built over is
  visible alongside the geometry that sizes them.

For the spatial view (corridors, tubes, tiles on a real basemap) see
:mod:`csnav.viz.map_view`.
"""

from __future__ import annotations

import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from csnav.geometry.fov import FieldOfView
from csnav.trajectory.config import ConopsConfig
from csnav.trajectory.coverage import AglProvider, height_as_agl
from csnav.trajectory.trajectory import X0_NODE, Trajectory, TrajectorySet
from csnav.trajectory.tube import TubeModel
from csnav.trajectory.waypoints import TrajectoryRole
from csnav.viz.style import PRIMARY_COLOR, TRANSITION_COLOR, X0_COLOR, color_for


def geographic_layout(trajectory_set: TrajectorySet) -> dict[str, tuple[float, float]]:
    """Node positions in ``(lon, lat)`` degrees - each node at its trajectory's midpoint.

    Gives the structural graph a spatial reading: the node order across the
    figure matches the order of the routes on the ground. Positions are in
    degrees purely as a plotting coordinate; no metric math is done on them.
    """
    positions = {X0_NODE: (trajectory_set.x0.lon, trajectory_set.x0.lat)}
    for trajectory in trajectory_set.candidates:
        midpoint = trajectory.point_at(trajectory.length / 2.0)
        positions[trajectory.id] = (midpoint.lon, midpoint.lat)
    return positions


def plot_trajectory_graph(
    trajectory_set: TrajectorySet,
    ax: Axes | None = None,
    layout: str = "geographic",
    show_edge_labels: bool = True,
) -> Axes:
    """Draw ``T`` as a directed graph of candidates and the transitions between them.

    ``layout`` is ``"geographic"`` (nodes at their trajectory midpoints, so the
    graph reads like a coarse map) or any of ``"spring"``/``"shell"``/
    ``"kamada_kawai"`` for a purely structural arrangement. Edge labels name the
    transition corridor and its length in meters.

    Returns the :class:`~matplotlib.axes.Axes` so callers can add to it or save
    the figure themselves.
    """
    import matplotlib.pyplot as plt

    graph = trajectory_set.to_networkx()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    positions = _layout(graph, trajectory_set, layout)
    order = tuple(t.id for t in trajectory_set.trajectories)

    node_colors = [
        X0_COLOR
        if node == X0_NODE
        else color_for(node, TrajectoryRole(graph.nodes[node]["role"]), order)
        for node in graph.nodes
    ]
    node_sizes = [700 if node == X0_NODE else 1100 for node in graph.nodes]

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color=[TRANSITION_COLOR for _ in graph.edges],
        width=1.8,
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
        node_size=node_sizes,
    )
    nx.draw_networkx_nodes(
        graph, positions, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="white", linewidths=1.5
    )
    # Labels sit just below their node rather than inside it: trajectory ids
    # are long enough ("t_alt_north") to overflow any readable node marker.
    label_offset = _label_offset(positions)
    nx.draw_networkx_labels(
        graph,
        {node: (x, y - label_offset) for node, (x, y) in positions.items()},
        ax=ax,
        labels={node: node for node in graph.nodes},
        font_size=9,
        font_weight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    if show_edge_labels:
        labels = {}
        for source, target, data in graph.edges(data=True):
            via = data.get("via")
            labels[(source, target)] = (
                f"{via}\n{data['length_m']:.0f} m" if via else "direct"
            )
        nx.draw_networkx_edge_labels(
            graph, positions, ax=ax, edge_labels=labels, font_size=7, label_pos=0.5, rotate=False
        )

    ax.set_title(f"Candidate trajectory set T - {trajectory_set.id}")
    if layout == "geographic":
        # networkx's draw helpers hide the axes; the geographic layout is the
        # one case where the coordinates mean something, so put them back.
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=8)
        ax.set_xlabel("longitude (deg, WGS84)")
        ax.set_ylabel("latitude (deg, WGS84)")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.margins(0.16)
    else:
        ax.set_axis_off()

    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=PRIMARY_COLOR, label=f"primary (t_p = {trajectory_set.primary_id})"),
            Line2D([], [], marker="o", linestyle="", color=color_for(_first_alternate(trajectory_set), TrajectoryRole.ALTERNATE, order), label="alternate candidate"),
            Line2D([], [], marker="o", linestyle="", color=X0_COLOR, label="x_0 (start state)"),
            Line2D([], [], linestyle="-", color=TRANSITION_COLOR, label="transition corridor"),
        ],
        loc="best",
        fontsize=8,
        frameon=True,
    )
    return ax


def _label_offset(positions: dict[str, tuple[float, float]]) -> float:
    """Vertical gap between a node and its label, as a fraction of the layout's height."""
    ys = [y for _, y in positions.values()]
    span = max(ys) - min(ys)
    return 0.075 * (span if span > 0 else 1.0)


def _first_alternate(trajectory_set: TrajectorySet) -> str:
    for trajectory in trajectory_set.candidates:
        if trajectory.role is TrajectoryRole.ALTERNATE:
            return trajectory.id
    return trajectory_set.primary_id


def _layout(graph: nx.DiGraph, trajectory_set: TrajectorySet, layout: str) -> dict:
    if layout == "geographic":
        return geographic_layout(trajectory_set)
    if layout == "spring":
        return nx.spring_layout(graph, seed=0)
    if layout == "shell":
        return nx.shell_layout(graph)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    raise ValueError(f"unknown layout {layout!r} (expected geographic, spring, shell or kamada_kawai)")


def plot_trajectory_profile(
    trajectory: Trajectory,
    tube: TubeModel,
    window_length: float,
    field_of_view: FieldOfView | None = None,
    agl_provider: AglProvider = height_as_agl,
    ax: Axes | None = None,
) -> Axes:
    """Height and lateral extents of one trajectory against arc length, with window edges.

    The upper trace is height above ground in meters (via ``agl_provider``);
    the shaded band is the tube's lateral radius, and the outer band adds the
    FOV ground radius at that height - i.e. how far to either side of the track
    the manifest for that point has to reach. Dashed verticals are the manifest
    window boundaries (``window_length`` meters of arc length each).

    Both axes are in meters: arc length along x, meters of height / lateral
    offset along y.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    samples = 200
    distances = [trajectory.length * step / samples for step in range(samples + 1)]
    points = [trajectory.point_at(distance) for distance in distances]
    agls = [agl_provider(point) for point in points]

    ax.plot(distances, agls, color=PRIMARY_COLOR, lw=2, label="height above ground (m)")

    tube_band = [tube.radius for _ in distances]
    ax.fill_between(distances, 0, tube_band, color="#0072B2", alpha=0.20, label=f"tube radius {tube.radius:.0f} m")
    if field_of_view is not None:
        outer = [tube.radius + field_of_view.ground_radius(agl) for agl in agls]
        ax.plot(distances, outer, color="#009E73", lw=1.5, ls="--", label="tube + FOV ground radius (m)")

    for window in trajectory.windows(window_length):
        ax.axvline(window.end_distance, color="#999999", lw=0.8, ls=":")
    ax.axvline(0.0, color="#999999", lw=0.8, ls=":", label=f"window edges ({window_length:.0f} m)")

    ax.set_xlim(0, trajectory.length)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("arc length along trajectory (m)")
    ax.set_ylabel("meters")
    ax.set_title(f"{trajectory.id} - {trajectory.length:.0f} m over {trajectory.duration:.0f} s")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    return ax


def trajectory_set_figure(
    trajectory_set: TrajectorySet,
    conops: ConopsConfig,
    agl_provider: AglProvider = height_as_agl,
) -> Figure:
    """One figure: the graph of ``T`` on top, a profile per candidate trajectory below.

    A convenience for ``scripts/visualize_trajectories.py`` - the structural
    view and the per-trajectory geometry side by side, all driven from the same
    :class:`~csnav.trajectory.config.ConopsConfig` so the tube radius shown is
    the one the manifests were (or will be) built with.
    """
    import matplotlib.pyplot as plt

    candidates = trajectory_set.candidates
    figure = plt.figure(figsize=(11, 4 + 3 * len(candidates)))
    grid = figure.add_gridspec(len(candidates) + 2, 1, height_ratios=[2, 2, *[1] * len(candidates)])

    graph_ax = figure.add_subplot(grid[0:2, 0])
    plot_trajectory_graph(trajectory_set, ax=graph_ax)

    for row, trajectory in enumerate(candidates):
        profile_ax = figure.add_subplot(grid[row + 2, 0])
        plot_trajectory_profile(
            trajectory,
            conops.tube_for(trajectory),
            conops.window_length,
            field_of_view=conops.field_of_view,
            agl_provider=agl_provider,
            ax=profile_ax,
        )

    figure.tight_layout()
    return figure
