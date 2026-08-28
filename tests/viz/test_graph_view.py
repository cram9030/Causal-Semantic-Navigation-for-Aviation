"""Interactive structural figures: the transition graph, the routes, the profiles.

These check content, not just that rendering succeeds: the figure has to carry
the trajectory set's actual structure, and - the point of the rewrite - it must
be a *structural* view, with node positions that are graph layers rather than
latitudes and longitudes.
"""

from __future__ import annotations

from csnav.trajectory.trajectory import X0_NODE, TrajectorySet, TransitionRule
from csnav.viz.graph_view import (
    layered_layout,
    route_profile_figure,
    route_table_figure,
    transition_graph_figure,
    write_report,
)


def test_layout_puts_x0_first_and_layers_by_transition_count(trajectory_set):
    positions = layered_layout(trajectory_set.to_networkx())

    assert set(positions) == {X0_NODE, "due_east", "parallel_north"}
    assert positions[X0_NODE][0] < positions["due_east"][0] < positions["parallel_north"][0]


def test_layout_is_not_geographic(trajectory_set):
    """The whole point of the structural view: positions are layout units, not degrees."""
    positions = layered_layout(trajectory_set.to_networkx())
    bounds = trajectory_set.bounds

    for x, y in positions.values():
        assert not (bounds.xmin <= x <= bounds.xmax)
        assert not (bounds.ymin <= y <= bounds.ymax)


def test_a_route_reachable_two_ways_sits_at_its_later_layer(due_east, parallel_north, dogleg):
    trajectory_set = TrajectorySet(
        id="diamond",
        trajectories=(due_east, parallel_north, dogleg),
        primary_id="due_east",
        x0=due_east.waypoints[0],
        transitions=(
            TransitionRule(source=X0_NODE, target="due_east"),
            TransitionRule(source=X0_NODE, target="dogleg"),
            TransitionRule(source="due_east", target="parallel_north"),
            TransitionRule(source="parallel_north", target="dogleg"),
        ),
    )
    positions = layered_layout(trajectory_set.to_networkx())
    assert positions["dogleg"][0] > positions["parallel_north"][0]


def test_graph_figure_labels_every_node(trajectory_set, conops):
    figure = transition_graph_figure(trajectory_set, conops)
    labels = {label for trace in figure.data for label in (trace.text or [])}
    assert {"due_east", "parallel_north"} <= labels


def test_graph_figure_edge_hover_reports_the_initiation_window(trajectory_set, conops):
    figure = transition_graph_figure(trajectory_set, conops)
    hovers = [text for trace in figure.data for text in (trace.hovertext or [])]
    transition_hover = next(text for text in hovers if "due_east &#8594; parallel_north" in text)

    assert "may initiate anywhere in" in transition_hover
    assert "sampled paths" in transition_hover
    assert "turns demanded" in transition_hover


def test_graph_figure_marks_the_entry_edge_as_such(trajectory_set, conops):
    hovers = [
        text for trace in transition_graph_figure(trajectory_set, conops).data for text in (trace.hovertext or [])
    ]
    assert any("may be flown from x_0" in text for text in hovers)


def test_graph_figure_node_hover_carries_tube_and_window_counts(trajectory_set, conops):
    hovers = [
        text for trace in transition_graph_figure(trajectory_set, conops).data for text in (trace.hovertext or [])
    ]
    node_hover = next(text for text in hovers if text.startswith("<b>due_east</b>"))

    assert f"tube radius: {conops.tube_radius:.0f} m" in node_hover
    assert "manifest windows" in node_hover


def test_graph_figure_draws_an_arrow_per_edge(trajectory_set, conops):
    figure = transition_graph_figure(trajectory_set, conops)
    arrows = [note for note in figure.layout.annotations if note.showarrow]
    assert len(arrows) == len(trajectory_set.transitions)


def test_graph_figure_works_without_a_conops(trajectory_set):
    figure = transition_graph_figure(trajectory_set)
    assert figure.data


def test_route_table_lists_every_permitted_route(trajectory_set):
    figure = route_table_figure(trajectory_set)
    listed = figure.data[0].cells.values[1]
    assert len(listed) == len(trajectory_set.route_paths())
    assert any("due_east" in row and "parallel_north" in row for row in listed)


def test_profile_figure_has_one_row_per_route(trajectory_set, conops):
    figure = route_profile_figure(trajectory_set, conops)
    axes = {trace.yaxis for trace in figure.data}
    assert len(axes) == len(trajectory_set.trajectories)


def test_profile_figure_shows_the_configured_tube_radius(trajectory_set, conops):
    figure = route_profile_figure(trajectory_set, conops)
    assert any(f"tube {conops.tube_radius:.0f} m" == trace.name for trace in figure.data)


def test_profile_figure_adds_the_camera_reach_trace_only_with_a_camera(trajectory_set, conops):
    with_camera = route_profile_figure(trajectory_set, conops)
    without = route_profile_figure(trajectory_set, conops.__class__(
        tube_radius=conops.tube_radius, window_length=conops.window_length
    ))
    assert any("ground reach" in (trace.name or "") for trace in with_camera.data)
    assert not any("ground reach" in (trace.name or "") for trace in without.data)


def test_profile_figure_marks_a_window_boundary_per_window(trajectory_set, conops):
    figure = route_profile_figure(trajectory_set, conops)
    expected = sum(
        len(t.windows(conops.window_length)) for t in trajectory_set.trajectories
    )
    assert len(figure.layout.shapes) == expected


def test_report_is_a_single_self_contained_html_page(trajectory_set, conops, tmp_path):
    destination = write_report(
        [transition_graph_figure(trajectory_set, conops), route_table_figure(trajectory_set)],
        tmp_path / "nested" / "report.html",
        title="test set",
    )
    html = destination.read_text(encoding="utf-8")

    assert html.startswith("<!DOCTYPE html>")
    assert "<title>test set</title>" in html
    assert "Plotly.newPlot" in html


def test_report_inlines_plotly_once_and_never_loads_it_from_a_cdn(trajectory_set, conops, tmp_path):
    """The maps open offline; so must this. Every figure shares the one inlined bundle."""
    destination = write_report(
        [
            transition_graph_figure(trajectory_set, conops),
            route_table_figure(trajectory_set),
            route_profile_figure(trajectory_set, conops),
        ],
        tmp_path / "report.html",
    )
    html = destination.read_text(encoding="utf-8")

    assert "<script src=" not in html
    assert html.count("Plotly.newPlot") == 3
    # The library is ~3 MB; three copies would be a 9 MB page.
    assert destination.stat().st_size < 8_000_000
