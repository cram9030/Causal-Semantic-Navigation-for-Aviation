"""Static trajectory-graph and profile figures.

These are smoke-plus-content tests: rendering has to succeed headless, and the
figure has to actually contain the trajectory set's structure (a node per
candidate, an edge per transition), not just any drawing.
"""

from __future__ import annotations

import pytest

from csnav.trajectory.trajectory import X0_NODE
from csnav.viz.graph_view import (
    geographic_layout,
    plot_trajectory_graph,
    plot_trajectory_profile,
    trajectory_set_figure,
)


def test_geographic_layout_places_every_node_including_x0(trajectory_set):
    positions = geographic_layout(trajectory_set)
    assert set(positions) == {X0_NODE, "due_east", "due_north"}

    bounds = trajectory_set.bounds
    for lon, lat in positions.values():
        assert bounds.xmin <= lon <= bounds.xmax
        assert bounds.ymin <= lat <= bounds.ymax


def test_graph_draws_a_label_for_every_node_and_edge(trajectory_set):
    ax = plot_trajectory_graph(trajectory_set)
    texts = {text.get_text() for text in ax.texts}

    assert {"x0", "due_east", "due_north"} <= texts
    assert any("x_east_to_north" in text for text in texts)
    assert any("direct" == text for text in texts)


def test_graph_axes_are_labelled_in_degrees_for_the_geographic_layout(trajectory_set):
    ax = plot_trajectory_graph(trajectory_set, layout="geographic")
    assert "longitude" in ax.get_xlabel()
    assert "latitude" in ax.get_ylabel()
    assert trajectory_set.id in ax.get_title()


def test_structural_layouts_hide_the_coordinate_axes(trajectory_set):
    ax = plot_trajectory_graph(trajectory_set, layout="spring")
    assert not ax.axison


def test_unknown_layout_is_refused(trajectory_set):
    with pytest.raises(ValueError, match="unknown layout"):
        plot_trajectory_graph(trajectory_set, layout="hexagonal")


def test_profile_spans_the_full_arc_length_and_marks_window_edges(due_east, tube):
    ax = plot_trajectory_profile(due_east, tube, window_length=500.0)

    assert ax.get_xlim() == pytest.approx((0.0, due_east.length))
    assert "arc length" in ax.get_xlabel()
    # One dashed vertical per window edge, plus the one at zero.
    assert len(ax.lines) >= len(due_east.windows(500.0))


def test_profile_shows_the_configured_radius_in_its_legend(due_east, tube):
    ax = plot_trajectory_profile(due_east, tube, window_length=500.0)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any(f"{tube.radius:.0f} m" in label for label in labels)


def test_profile_adds_the_fov_trace_when_a_field_of_view_is_given(due_east, tube, conops):
    without = plot_trajectory_profile(due_east, tube, 500.0)
    with_fov = plot_trajectory_profile(due_east, tube, 500.0, field_of_view=conops.field_of_view)
    assert len(with_fov.lines) == len(without.lines) + 1


def test_set_figure_has_one_profile_per_candidate_plus_the_graph(trajectory_set, conops):
    figure = trajectory_set_figure(trajectory_set, conops)
    # Transition corridors are edges of the graph, not their own profile panel.
    assert len(figure.axes) == 1 + len(trajectory_set.candidates)


def test_set_figure_saves_a_non_empty_png(trajectory_set, conops, tmp_path):
    destination = tmp_path / "graph.png"
    trajectory_set_figure(trajectory_set, conops).savefig(destination, dpi=80)
    assert destination.stat().st_size > 5_000
