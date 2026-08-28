"""Visualization tools for the Phase 1 trajectory set, tubes, and manifests.

Two complementary views, both emitting self-contained interactive HTML:

* :mod:`csnav.viz.graph_view` (Plotly) - the *structural* view: the transition
  graph of ``T``, the routes its rules permit, and per-route arc-length
  profiles. Node positions are graph layers, not geography.
* :mod:`csnav.viz.map_view` (folium/Leaflet) - the *spatial* view: corridors,
  transition families and the regions they can reach, per-window visible
  footprints, imagery tiles in view, and built manifests, over a real basemap.

Both need the optional visualization dependencies::

    pip install -e ".[viz]"

Importing this package does not pull those in - the submodules do, so the rest
of :mod:`csnav` stays installable without Plotly or folium.
"""

from csnav.viz.style import TRAJECTORY_PALETTE, color_for

__all__ = ["TRAJECTORY_PALETTE", "color_for"]
