"""Visualization tools for the Phase 1 trajectory set, tubes, and manifests.

Two complementary views:

* :mod:`csnav.viz.graph_view` (matplotlib) - the structural graph of ``T`` and
  per-trajectory arc-length profiles. Static, no network needed.
* :mod:`csnav.viz.map_view` (folium/Leaflet) - interactive HTML maps of the
  corridors, per-window visible footprints, imagery tiles in view, and built
  manifests, over a real basemap.

Both need the optional visualization dependencies::

    pip install -e ".[viz]"

Importing this package does not pull those in - the submodules do, so the rest
of :mod:`csnav` stays installable without matplotlib or folium.
"""

from csnav.viz.style import TRAJECTORY_PALETTE, color_for

__all__ = ["TRAJECTORY_PALETTE", "color_for"]
