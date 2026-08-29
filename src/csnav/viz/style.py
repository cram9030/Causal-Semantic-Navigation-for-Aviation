"""Shared colours and labels for the Phase 1 trajectory visualizations.

Kept in one place so the trajectory-graph view and the map view agree: a
trajectory is the same colour in both, and ``t_p`` reads as the primary in
both.
"""

from __future__ import annotations

from csnav.trajectory.waypoints import TrajectoryRole

#: Qualitative palette (Okabe-Ito, chosen for colour-vision-deficiency
#: safety) cycled over trajectories so each keeps a stable colour across
#: figures for a given trajectory set.
TRAJECTORY_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#8C564B",
)

#: Colour used for the primary trajectory ``t_p``, overriding the palette so
#: it is identifiable at a glance.
PRIMARY_COLOR = "#B2182B"

#: Colour for transition corridors - deliberately muted, since they are the
#: contingency structure rather than a candidate route.
TRANSITION_COLOR = "#6E6E6E"

#: Colour for the known start state ``x_0``.
X0_COLOR = "#111111"

#: Colour for imagery tile footprints.
TILE_COLOR = "#444444"

#: Alternating fills for consecutive manifest windows. Adjacent windows share a
#: boundary and each corridor is round-capped, so a single colour makes the
#: overlaps read as a chain of blobs; alternating between two shades keeps the
#: sequence legible even with every window shown at once.
WINDOW_SHADES = ("#009E73", "#0072B2")


def window_shade(index: int) -> str:
    """Fill colour for the ``index``-th window of a trajectory."""
    return WINDOW_SHADES[index % len(WINDOW_SHADES)]

#: Colour for manifest landmarks (candidate roads) and their intersections.
LANDMARK_COLOR = "#F0C808"
INTERSECTION_COLOR = "#FFFFFF"

#: Neutral ink and gridline colours, shared by the Plotly figures and the map
#: tooltips so the two views read as one set.
TEXT_COLOR = "#222222"
GRID_COLOR = "#999999"

ROLE_LABELS = {
    TrajectoryRole.PRIMARY: "primary (t_p)",
    TrajectoryRole.ALTERNATE: "alternate",
    TrajectoryRole.TRANSITION: "transition corridor",
}


def color_for(trajectory_id: str, role: TrajectoryRole, order: tuple[str, ...]) -> str:
    """Stable colour for a trajectory.

    ``order`` is the trajectory-id ordering the palette is cycled over (pass
    the trajectory set's own order so colours don't shift when a subset is
    drawn). Primary and transition roles get fixed colours; everything else
    cycles through :data:`TRAJECTORY_PALETTE`.
    """
    if role is TrajectoryRole.PRIMARY:
        return PRIMARY_COLOR
    if role is TrajectoryRole.TRANSITION:
        return TRANSITION_COLOR
    index = order.index(trajectory_id) if trajectory_id in order else 0
    return TRAJECTORY_PALETTE[index % len(TRAJECTORY_PALETTE)]
