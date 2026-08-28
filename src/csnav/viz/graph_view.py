"""Interactive structural views of a trajectory set: the transition graph and route profiles.

Two Plotly figures, both written as self-contained HTML so they sit alongside
the Leaflet maps rather than as a static image:

* :func:`transition_graph_figure` draws ``T`` as its ``networkx.DiGraph`` -
  candidate routes as nodes, permitted transitions as edges, ``x_0`` as the
  entry. This is a **structural** view: node positions are graph layers, not
  geography. Where the routes sit on the ground is the map's job
  (:mod:`csnav.viz.map_view`); plotting a route's midpoint at its own latitude
  and longitude would produce a picture that is neither a map nor a graph.
* :func:`route_profile_figure` draws each candidate against arc length -
  height above ground, the tube radius, and the camera's ground reach - with
  the manifest window boundaries marked.

:func:`write_report` puts them, plus the enumerated routes, into one HTML page.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from csnav.trajectory.config import ConopsConfig
from csnav.trajectory.coverage import AglProvider, height_as_agl
from csnav.trajectory.trajectory import X0_NODE, TrajectorySet
from csnav.trajectory.transition import TransitionModel
from csnav.trajectory.waypoints import TrajectoryRole
from csnav.viz.style import (
    GRID_COLOR,

    TEXT_COLOR,
    TRANSITION_COLOR,
    X0_COLOR,
    color_for,
)

#: Horizontal gap between graph layers, and vertical gap between siblings, in
#: the figure's own arbitrary layout units.
_LAYER_GAP = 1.0
_SIBLING_GAP = 1.0

#: Fraction of an edge trimmed at each end so its arrowhead does not land under
#: a node marker.
_EDGE_TRIM = 0.14

#: How far an edge bows sideways per extra layer it spans, in layout units. An
#: edge between adjacent layers is drawn straight; one that skips a layer has to
#: bow, or it runs straight through the node in between and disappears.
_EDGE_BOW = 0.55

#: Points used to draw a bowed edge.
_EDGE_STEPS = 24

def layered_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Node positions in layers by how many transitions it takes to reach each node.

    ``x`` is the longest transition count from :data:`X0_NODE` - so a route
    reachable both directly and via a divert sits at its *later* position, which
    is where it reads correctly relative to the routes that feed it. ``y``
    spreads a layer's nodes evenly and centres each layer. Positions are
    arbitrary layout units with no physical meaning.
    """
    depths: dict[str, int] = {X0_NODE: 0} if X0_NODE in graph else {}
    for node in nx.topological_sort(graph) if nx.is_directed_acyclic_graph(graph) else graph.nodes:
        if node in depths:
            continue
        predecessors = [depths[p] for p in graph.predecessors(node) if p in depths]
        depths[node] = max(predecessors) + 1 if predecessors else 1

    layers: dict[int, list[str]] = {}
    for node, depth in sorted(depths.items(), key=lambda item: (item[1], item[0])):
        layers.setdefault(depth, []).append(node)

    positions: dict[str, tuple[float, float]] = {}
    for depth, nodes in layers.items():
        offset = (len(nodes) - 1) / 2.0
        for index, node in enumerate(nodes):
            positions[node] = (depth * _LAYER_GAP, (offset - index) * _SIBLING_GAP)
    return positions

def _edge_curve(
    start: tuple[float, float], end: tuple[float, float], bow: float
) -> tuple[list[float], list[float]]:
    """Quadratic Bezier from ``start`` to ``end``, bowed perpendicular by ``bow`` layout units."""
    (x0, y0), (x1, y1) = start, end
    x0, y0 = x0 + (x1 - x0) * _EDGE_TRIM, y0 + (y1 - y0) * _EDGE_TRIM
    x1, y1 = x1 - (x1 - x0) * _EDGE_TRIM, y1 - (y1 - y0) * _EDGE_TRIM

    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy) or 1.0
    control = ((x0 + x1) / 2.0 - dy / norm * bow, (y0 + y1) / 2.0 + dx / norm * bow)

    xs, ys = [], []
    for step in range(_EDGE_STEPS + 1):
        u = step / _EDGE_STEPS
        xs.append((1 - u) ** 2 * x0 + 2 * (1 - u) * u * control[0] + u**2 * x1)
        ys.append((1 - u) ** 2 * y0 + 2 * (1 - u) * u * control[1] + u**2 * y1)
    return xs, ys

def _edge_hover(
    trajectory_set: TrajectorySet,
    source: str,
    target: str,
    data: dict[str, Any],
    model: TransitionModel | None,
) -> str:
    if data.get("is_entry"):
        return f"<b>{source} &#8594; {target}</b><br>entry: this route may be flown from x_0"

    rule = data["rule"]
    lines = [
        f"<b>{source} &#8594; {target}</b>",
        f"may initiate anywhere in {data['initiate_from_m']:.0f}-{data['initiate_to_m']:.0f} m "
        "of arc length on the source",
    ]
    limit = rule.max_turn_deg if rule.max_turn_deg is not None else (model.max_turn_deg if model else None)
    if limit is not None:
        lines.append(f"turn screen: {limit:.0f}&#176;")
    if model is not None:
        family = model.family(trajectory_set.by_id(source), trajectory_set.by_id(target), rule)
        low, high = family.turn_range
        lines.append(f"{len(family)} sampled paths ({family.rejected} screened out)")
        if family.paths:
            lines.append(f"turns demanded: {low:.0f}-{high:.0f}&#176;")
    return "<br>".join(lines)

def _node_hover(trajectory_set: TrajectorySet, node: str, data: dict[str, Any], conops: ConopsConfig | None) -> str:
    if node == X0_NODE:
        x0 = trajectory_set.x0
        return (
            f"<b>x_0</b> - known start state<br>"
            f"{x0.lat:.5f}, {x0.lon:.5f}<br>{x0.height:.0f} m, t = {x0.time:.0f} s"
        )
    trajectory = data["trajectory"]
    lines = [
        f"<b>{node}</b>",
        str(trajectory.metadata.get("name", "")),
        f"role: {trajectory.role.value}",
        f"{data['length_m']:.0f} m over {data['duration_s']:.0f} s",
    ]
    if conops is not None:
        tube = conops.tube_for(trajectory)
        windows = trajectory.windows(conops.window_length)
        lines.append(f"tube radius: {tube.radius:.0f} m")
        lines.append(f"{len(windows)} manifest windows of ~{conops.window_length:.0f} m")
    return "<br>".join(line for line in lines if line)

def transition_graph_figure(
    trajectory_set: TrajectorySet,
    conops: ConopsConfig | None = None,
    model: TransitionModel | None = None,
) -> go.Figure:
    """The candidate set ``T`` as an interactive transition graph.

    Nodes are candidate routes plus ``x_0``; edges are the permitted hand-offs.
    Positions come from :func:`layered_layout` and are structural - deliberately
    not geographic. Hovering an edge reports where along the source a transition
    may initiate and, when ``model`` is given, how many paths that admits and
    the turn angles they demand.
    """
    graph = trajectory_set.to_networkx()
    positions = layered_layout(graph)
    order = tuple(t.id for t in trajectory_set.trajectories)
    if model is None and conops is not None:
        model = conops.transition

    figure = go.Figure()
    annotations: list[dict[str, Any]] = []
    midpoints: dict[str, list[Any]] = {"x": [], "y": [], "text": [], "color": []}

    for source, target, data in graph.edges(data=True):
        start, end = positions[source], positions[target]
        span = abs(end[0] - start[0]) / _LAYER_GAP
        color = X0_COLOR if data.get("is_entry") else TRANSITION_COLOR
        xs, ys = _edge_curve(start, end, bow=_EDGE_BOW * max(0.0, span - 1.0))

        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": color, "width": 1.6},
                opacity=0.75,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        annotations.append(
            {
                "x": xs[-1],
                "y": ys[-1],
                "ax": xs[-2],
                "ay": ys[-2],
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1.4,
                "arrowwidth": 1.6,
                "arrowcolor": color,
                "opacity": 0.85,
            }
        )
        middle = len(xs) // 2
        midpoints["x"].append(xs[middle])
        midpoints["y"].append(ys[middle])
        midpoints["color"].append(color)
        midpoints["text"].append(_edge_hover(trajectory_set, source, target, data, model))

    figure.add_trace(
        go.Scatter(
            x=midpoints["x"],
            y=midpoints["y"],
            mode="markers",
            marker={"size": 13, "color": midpoints["color"], "opacity": 0.45, "symbol": "diamond"},
            hovertext=midpoints["text"],
            hoverinfo="text",
            showlegend=False,
            name="transitions",
        )
    )

    node_x, node_y, node_text, node_hover, node_color = [], [], [], [], []
    for node, data in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append("x&#8320;" if node == X0_NODE else node)
        node_hover.append(_node_hover(trajectory_set, node, data, conops))
        node_color.append(
            X0_COLOR if node == X0_NODE else color_for(node, TrajectoryRole(data["role"]), order)
        )

    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={"size": 30, "color": node_color, "line": {"width": 2, "color": "white"}},
            text=node_text,
            textposition="bottom center",
            textfont={"size": 12, "color": TEXT_COLOR},
            hovertext=node_hover,
            hoverinfo="text",
            showlegend=False,
            name="routes",
        )
    )

    for label, color in (("entry from x\u2080", X0_COLOR), ("transition family", TRANSITION_COLOR)):
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines", line={"color": color, "width": 2}, name=label
            )
        )

    figure.update_layout(
        title=(
            f"Candidate trajectory set T - {trajectory_set.id}"
            f"<br><sub>structural view: position is graph layer, not geography "
            f"(t_p = {trajectory_set.primary_id})</sub>"
        ),
        annotations=annotations,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.12, "x": 0},
        hovermode="closest",
        plot_bgcolor="white",
        margin={"l": 40, "r": 40, "t": 90, "b": 40},
        height=480,
    )
    figure.update_xaxes(visible=False, range=[-0.5, max(node_x) + 0.5])
    figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return figure

def route_profile_figure(
    trajectory_set: TrajectorySet,
    conops: ConopsConfig,
    agl_provider: AglProvider = height_as_agl,
    samples: int = 120,
) -> go.Figure:
    """Height, tube radius, and camera ground reach against arc length, one row per route.

    Every quantity on the y-axis is in meters: height above ground (via
    ``agl_provider``), the tube's lateral radius as a filled band, and - when a
    camera is configured - the tube radius plus the camera's ground reach, which
    is how far to either side of the track that point's manifest has to look.
    Dashed verticals are the manifest window boundaries.
    """
    trajectories = trajectory_set.trajectories
    figure = make_subplots(
        rows=len(trajectories),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.06, 0.9 / max(len(trajectories), 1)),
        subplot_titles=[
            f"{t.id} - {t.length:.0f} m over {t.duration:.0f} s" for t in trajectories
        ],
    )
    order = tuple(t.id for t in trajectories)

    for row, trajectory in enumerate(trajectories, start=1):
        tube = conops.tube_for(trajectory)
        color = color_for(trajectory.id, trajectory.role, order)
        distances = [trajectory.length * step / samples for step in range(samples + 1)]
        points = [trajectory.point_at(distance) for distance in distances]
        agls = [agl_provider(point) for point in points]

        figure.add_trace(
            go.Scatter(
                x=distances,
                y=[tube.radius] * len(distances),
                mode="lines",
                line={"width": 0},
                fill="tozeroy",
                fillcolor="rgba(0, 114, 178, 0.18)",
                name=f"tube {tube.radius:.0f} m",
                legendgroup="tube",
                showlegend=row == 1,
                hovertemplate="tube radius %{y:.0f} m<extra></extra>",
            ),
            row=row,
            col=1,
        )

        if conops.camera is not None:
            reach = [
                tube.radius
                + conops.camera.bounded_ground_reach(
                    agl, trajectory.distance_to_nearest_waypoint(distance)
                )
                for distance, agl in zip(distances, agls)
            ]
            figure.add_trace(
                go.Scatter(
                    x=distances,
                    y=reach,
                    mode="lines",
                    line={"color": "#009E73", "width": 1.6, "dash": "dash"},
                    name="tube + camera ground reach",
                    legendgroup="reach",
                    showlegend=row == 1,
                    hovertemplate="search radius %{y:.0f} m at %{x:.0f} m<extra></extra>",
                ),
                row=row,
                col=1,
            )

        figure.add_trace(
            go.Scatter(
                x=distances,
                y=agls,
                mode="lines",
                line={"color": color, "width": 2.5},
                name="height above ground",
                legendgroup="agl",
                showlegend=row == 1,
                hovertemplate="%{y:.0f} m AGL at %{x:.0f} m<extra></extra>",
            ),
            row=row,
            col=1,
        )

        for window in trajectory.windows(conops.window_length):
            figure.add_vline(
                x=window.end_distance,
                line={"color": GRID_COLOR, "width": 0.8, "dash": "dot"},
                row=row,
                col=1,
            )

        figure.update_xaxes(title_text="arc length (m)", range=[0, trajectory.length], row=row, col=1)
        figure.update_yaxes(title_text="meters", rangemode="tozero", row=row, col=1)

    figure.update_layout(
        title=(
            f"Route profiles - {trajectory_set.id}"
            f"<br><sub>window boundaries dotted, every {conops.window_length:.0f} m of arc length</sub>"
        ),
        height=260 * len(trajectories) + 120,
        hovermode="x unified",
        plot_bgcolor="white",
        margin={"l": 60, "r": 40, "t": 100, "b": 50},
    )
    return figure

def route_table_figure(trajectory_set: TrajectorySet) -> go.Figure:
    """The routes the transition rules permit, as a table.

    Each row is a path through the transition graph - ``t_p`` flown to its end,
    or ``t_p -> t_alt_north -> the northern return``. None of these is declared
    anywhere: they are what the rules imply, and each stands for a *family* of
    flights, since where each hand-off begins is continuous.
    """
    routes = trajectory_set.route_paths()
    lengths = [
        sum(trajectory_set.by_id(step).length for step in route) for route in routes
    ]
    figure = go.Figure(
        go.Table(
            header={
                "values": ["#", "route", "legs", "sum of leg lengths (m)"],
                "align": "left",
                "fill_color": "#F2F2F2",
                "font": {"color": TEXT_COLOR, "size": 12},
            },
            cells={
                "values": [
                    list(range(1, len(routes) + 1)),
                    [" &#8594; ".join(route) for route in routes],
                    [len(route) for route in routes],
                    [f"{length:,.0f}" for length in lengths],
                ],
                "align": "left",
                "height": 26,
                "font": {"size": 12},
            },
        )
    )
    figure.update_layout(
        title=(
            "Routes permitted by the transition rules"
            "<br><sub>each is a path through the graph, and each stands for a family of flights - "
            "a hand-off may begin anywhere along its source. Lengths sum the legs end to end; a real "
            "flight cuts each corner via a generated transition and flies less.</sub>"
        ),
        height=140 + 30 * max(len(routes), 1),
        margin={"l": 40, "r": 40, "t": 100, "b": 20},
    )
    return figure

def write_report(
    figures: Sequence[go.Figure] | Iterable[go.Figure],
    path: str | Path,
    title: str = "Trajectory set",
) -> Path:
    """Write several Plotly figures into one self-contained HTML page.

    Plotly's JavaScript is inlined into the first figure, so the page opens
    without network access - the same property the folium maps have for
    everything except their basemap tiles.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # The first block inlines plotly.js; the rest must neither re-inline it (the
    # library is ~3 MB) nor fall back to a CDN, which would break offline.
    blocks = [
        figure.to_html(full_html=False, include_plotlyjs=(index == 0))
        for index, figure in enumerate(figures)
    ]
    destination.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{margin:0;padding:16px;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
        "background:#FFFFFF;color:#222222}</style>"
        "</head><body>\n" + "\n".join(blocks) + "\n</body></html>\n",
        encoding="utf-8",
    )
    return destination
