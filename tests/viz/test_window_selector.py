"""The per-window layer control: its emitted spec, and its JavaScript.

The control's behaviour lives in a real ``.js`` file so it can be tested
directly; :func:`test_selector_javascript_logic` runs that node suite as part
of pytest rather than leaving it to be remembered. Everything else here checks
the Python half: that the spec literal wires the right Leaflet layers to the
right windows, and that labels cannot break out of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from csnav.viz.window_selector import (
    WindowCategory,
    WindowGroup,
    WindowLayers,
    WindowSelector,
    _js_string,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_SUITE = REPO_ROOT / "tests" / "viz" / "test_window_selector.js"
JS_SOURCE = REPO_ROOT / "src" / "csnav" / "viz" / "static" / "window_selector.js"


class FakeLayer:
    """Stands in for a folium layer: all the selector needs is its JS variable name."""

    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


def _selector(**overrides) -> WindowSelector:
    kwargs = {
        "groups": [
            WindowGroup(
                id="t_p",
                label="t_p",
                color="#B2182B",
                windows=(
                    WindowLayers(
                        window_id="t_p:0000",
                        label="0000 - 0-981 m",
                        layers={"footprint": FakeLayer("fg_foot_0"), "roads": FakeLayer("fg_road_0")},
                        detail="12 roads",
                    ),
                    WindowLayers(
                        window_id="t_p:0001",
                        label="0001 - 981-1961 m",
                        layers={"footprint": FakeLayer("fg_foot_1")},
                    ),
                ),
            )
        ],
        "categories": [
            WindowCategory("footprint", "footprints"),
            WindowCategory("roads", "roads", enabled=False),
        ],
        "title": "Manifest windows",
    }
    kwargs.update(overrides)
    return WindowSelector(**kwargs)


def test_spec_references_layer_variables_not_strings():
    """The spec is a JS literal, so a layer resolves to the real Leaflet object."""
    spec = _selector().spec
    assert '"footprint": fg_foot_0' in spec
    assert '"roads": fg_road_0' in spec
    assert '"fg_foot_0"' not in spec


def test_spec_carries_window_ids_labels_and_details():
    spec = _selector().spec
    assert '"t_p:0000"' in spec
    assert '"0000 - 0-981 m"' in spec
    assert '"12 roads"' in spec


def test_spec_records_each_categorys_initial_state():
    spec = _selector().spec
    assert '{key: "footprint", label: "footprints", enabled: true}' in spec
    assert '{key: "roads", label: "roads", enabled: false}' in spec


def test_a_label_cannot_break_out_of_the_literal_or_the_script_element():
    """Escaping quotes is not enough - a literal </script> would close the element."""
    selector = _selector(
        groups=[
            WindowGroup(
                id='t"p',
                label="</script><script>alert(1)</script>",
                color="#000000",
                windows=(WindowLayers(window_id="w", label='a "quoted" label', layers={}),),
            )
        ]
    )
    spec = selector.spec

    assert "</script>" not in spec
    assert "<script>" not in spec
    assert r"\u003c/script\u003e" in spec
    assert r"a \"quoted\" label" in spec


@pytest.mark.parametrize("text", ["t_p & <alt>", "</script>", 'a "quoted" label', "0-981 m"])
def test_escaped_text_decodes_back_to_the_original(text):
    """The escapes are a transport detail; the reader must still see the label."""
    assert json.loads(_js_string(text)) == text


def test_an_escaped_label_reaches_the_spec_intact():
    spec = _selector(
        groups=[
            WindowGroup(
                id="g",
                label="t_p & <alt>",
                color="#000000",
                windows=(WindowLayers(window_id="w", label="0-981 m", layers={}),),
            )
        ]
    ).spec
    assert "label: " + _js_string("t_p & <alt>") in spec


def test_options_carry_the_control_position():
    assert _selector(position="bottomleft").options == '{position: "bottomleft"}'


def test_the_shipped_script_is_inlined_so_the_page_works_offline():
    selector = _selector()
    assert "function csnavWindowSelectorState" in selector.script
    assert selector.script == JS_SOURCE.read_text(encoding="utf-8")


def test_rendering_emits_the_style_the_script_and_the_call(trajectory_set, conops):
    """A full render, so the template's macros are exercised the way folium calls them."""
    import folium

    fmap = folium.Map(location=[37.34, -121.88])
    layer = folium.FeatureGroup(name="w", control=False)
    layer.add_to(fmap)
    WindowSelector(
        groups=[
            WindowGroup(
                id="t_p",
                label="t_p",
                color="#B2182B",
                windows=(WindowLayers(window_id="t_p:0000", label="0000", layers={"footprint": layer}),),
            )
        ],
        categories=[WindowCategory("footprint", "footprints")],
    ).add_to(fmap)

    html = fmap.get_root().render()
    assert ".csnav-ws {" in html
    assert "function csnavWindowSelectorState" in html
    assert "csnavWindowSelector(" in html
    # The layer variable is declared before the control that references it.
    assert html.index(layer.get_name() + " = L.featureGroup") < html.index("csnavWindowSelector(")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_selector_javascript_logic():
    """Run the control's own node test suite - solo, all/none, and the category mask."""
    result = subprocess.run(
        ["node", str(JS_SUITE)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_selector_javascript_parses():
    result = subprocess.run(
        ["node", "--check", str(JS_SOURCE)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
