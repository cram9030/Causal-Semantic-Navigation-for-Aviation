"""A Leaflet control for isolating trajectory windows on a folium map.

A trajectory's manifest windows overlap by design - adjacent windows meet at a
shared boundary and each corridor is round-capped - so drawing them all at once
in one colour reads as a chain of blobs rather than a sequence of windows. The
fix is to put each window on its own layer and give the reader a way to pick
between them; folium's own ``LayerControl`` is flat, so listing 36 window
layers there would just move the problem.

:class:`WindowSelector` renders a collapsible tree instead: a row per
trajectory, expanding to that trajectory's windows, each with a checkbox and a
"solo" button that isolates it. Where a window has more than one kind of
geometry (its footprint, its candidate roads, its intersections, its imagery
tiles) those are **categories**, toggled across all windows at once, so a layer
is drawn when its window is selected *and* its category is enabled.

Layers handed to this control should be created with ``control=False`` so they
stay out of folium's own layer control - this one manages them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from branca.element import MacroElement, Template

#: The control's behaviour, kept in a real ``.js`` file so it can be linted and
#: unit-tested under node rather than living inside a template string.
_SCRIPT_PATH = Path(__file__).parent / "static" / "window_selector.js"

_CSS = """
.csnav-ws {
  background: rgba(255, 255, 255, 0.94);
  padding: 6px 8px;
  border-radius: 4px;
  font: 12px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: #222222;
  max-height: 60vh;
  overflow-y: auto;
  min-width: 190px;
  box-shadow: 0 1px 5px rgba(0, 0, 0, 0.4);
}
.csnav-ws-head { display: flex; align-items: center; gap: 4px; margin-bottom: 4px; }
.csnav-ws-title { font-weight: 600; flex: 1; }
.csnav-ws-cats {
  display: flex; flex-wrap: wrap; gap: 6px;
  padding: 3px 0 5px; margin-bottom: 4px; border-bottom: 1px solid #DDDDDD;
}
.csnav-ws-cat { display: flex; align-items: center; gap: 3px; cursor: pointer; white-space: nowrap; }
.csnav-ws-group { border-top: 1px solid #EEEEEE; }
.csnav-ws-grouphead { display: flex; align-items: center; gap: 4px; padding: 3px 0; cursor: pointer; }
.csnav-ws-caret { width: 9px; color: #666666; }
.csnav-ws-swatch { width: 9px; height: 9px; border-radius: 2px; flex: none; }
.csnav-ws-name { flex: 1; font-weight: 500; }
.csnav-ws-count { color: #666666; font-variant-numeric: tabular-nums; }
.csnav-ws-windows { display: none; padding: 0 0 4px 14px; }
.csnav-ws-window { display: flex; align-items: center; gap: 4px; padding: 1px 0; }
.csnav-ws-winlabel { display: flex; align-items: center; gap: 4px; flex: 1; cursor: pointer; }
.csnav-ws-wintext { font-variant-numeric: tabular-nums; }
.csnav-ws-btn {
  font: inherit; font-size: 10px; line-height: 1;
  padding: 2px 5px; cursor: pointer;
  border: 1px solid #CCCCCC; border-radius: 3px; background: #F7F7F7; color: #333333;
}
.csnav-ws-btn:hover { background: #E8E8E8; }
.csnav-ws input[type="checkbox"] { margin: 0; }
"""


@dataclass(frozen=True)
class WindowLayers:
    """One window's layers, keyed by category.

    ``window_id`` is the manifest window key (``"<trajectory_id>:0000"``);
    ``label`` is what the control shows; ``detail`` becomes the row's tooltip.
    ``layers`` maps a category key to the folium layer holding that category's
    geometry for this window - typically a ``FeatureGroup`` built with
    ``control=False``.
    """

    window_id: str
    label: str
    layers: Mapping[str, Any]
    detail: str = ""


@dataclass(frozen=True)
class WindowGroup:
    """A trajectory's windows, as one collapsible row of the control."""

    id: str
    label: str
    color: str
    windows: tuple[WindowLayers, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WindowCategory:
    """A kind of geometry that exists once per window, toggled across all of them.

    ``enabled`` is the initial state - set it ``False`` for a category that is
    useful but noisy by default, like every window's imagery tiles.
    """

    key: str
    label: str
    enabled: bool = True


class WindowSelector(MacroElement):
    """The collapsible per-window layer control.

    ``groups`` are the trajectories to list; ``categories`` the kinds of
    geometry each window carries (omit for a single unnamed category, in which
    case no category row is drawn). ``title`` heads the panel and ``position``
    is any Leaflet control position.

    Add it to the map *after* the layers it manages: it references their
    JavaScript variables directly, so they have to be declared first.
    """

    _template = Template(
        """
        {% macro header(this, kwargs) %}
            <style>{{ this.css }}</style>
        {% endmacro %}

        {% macro script(this, kwargs) %}
            {{ this.script }}
            csnavWindowSelector({{ this._parent.get_name() }}, {{ this.spec }}, {{ this.options }});
        {% endmacro %}
        """
    )

    def __init__(
        self,
        groups: Sequence[WindowGroup],
        categories: Sequence[WindowCategory] = (),
        title: str = "Windows",
        position: str = "topright",
    ) -> None:
        super().__init__()
        self._name = "WindowSelector"
        self.groups = list(groups)
        self.categories = list(categories)
        self.title = title
        self.position = position
        self.css = _CSS
        self.script = _SCRIPT_PATH.read_text(encoding="utf-8")

    @property
    def options(self) -> str:
        return "{position: %s}" % _js_string(self.position)

    @property
    def spec(self) -> str:
        """The control's spec as a JavaScript object literal.

        Not JSON: layer references have to come out as the bare JavaScript
        variable names folium declared for them, so they resolve to the actual
        Leaflet layers rather than to strings. Every piece of text goes through
        :func:`_js_string`, so a label cannot break out of the literal - or out
        of the ``<script>`` element the literal is written into.
        """
        categories = ", ".join(
            "{key: %s, label: %s, enabled: %s}"
            % (_js_string(category.key), _js_string(category.label), _js_bool(category.enabled))
            for category in self.categories
        )
        groups = ", ".join(self._group_literal(group) for group in self.groups)
        return (
            "{title: %s, categories: [%s], groups: [%s]}"
            % (_js_string(self.title), categories, groups)
        )

    def _group_literal(self, group: WindowGroup) -> str:
        windows = ", ".join(self._window_literal(window) for window in group.windows)
        return "{id: %s, label: %s, color: %s, windows: [%s]}" % (
            _js_string(group.id),
            _js_string(group.label),
            _js_string(group.color),
            windows,
        )

    def _window_literal(self, window: WindowLayers) -> str:
        layers = ", ".join(
            "%s: %s" % (_js_string(key), layer.get_name()) for key, layer in window.layers.items()
        )
        return "{id: %s, label: %s, detail: %s, layers: {%s}}" % (
            _js_string(window.window_id),
            _js_string(window.label),
            _js_string(window.detail),
            layers,
        )


def _js_bool(value: bool) -> str:
    return "true" if value else "false"


def _js_string(value: str) -> str:
    r"""A JavaScript string literal that is safe inside an inline ``<script>`` element.

    ``json.dumps`` alone is not enough: it escapes quotes and backslashes but
    leaves ``<`` untouched, so a label containing ``</script>`` would close the
    element early and everything after it would be parsed as markup. Escaping
    the three markup-significant characters as ``\uXXXX`` keeps the literal
    inert while decoding back to the original text at runtime.
    """
    escaped = json.dumps(value)
    for character, replacement in (("<", r"\u003c"), (">", r"\u003e"), ("&", r"\u0026")):
        escaped = escaped.replace(character, replacement)
    return escaped
