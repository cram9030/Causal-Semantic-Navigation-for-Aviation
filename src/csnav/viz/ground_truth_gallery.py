"""Static HTML paging gallery for exhaustive visual QA of rasterized panoptic labels.

The per-tile counterpart to `csnav.viz.ground_truth_view`'s geographic map:
this is built for a person to page through *every* tile in a label set
quickly - imagery and label are rendered as two separate, pixel-aligned PNGs
per tile and stacked with a CSS-adjustable opacity in the browser, rather
than a single pre-baked blend, so "imagery only" / "labels only" / "overlay
at any strength" are all the same slider with no extra images to generate.
A grid of thumbnails (one fixed-alpha blend each, baked in for quick
scanning) drives which tile is open in the large viewer; arrow keys/buttons
step through them, and a per-tile "flag" checkbox (persisted in the page's
own ``localStorage``, exportable as a plain text list) is how a reviewer
marks tiles worth a second look without leaving the page.

The output is one self-contained directory (``index.html`` + ``images/`` +
``thumbs/`` + a ``tiles.json`` manifest sidecar) meant to be opened directly
in a browser - no server needed, matching this project's other viz outputs
(`csnav.viz.map_view.save_map`). ``tiles.json`` is what lets a later call
into the same directory (a different ``--limit``/``--sample``/``--manifest``
selection, not just the same one) *add to* the page instead of replacing
it - see :func:`write_gallery`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
from PIL import Image

from csnav.data.ground_truth.labels import PanopticClass, PanopticLabel

#: Matches `csnav.viz.style.LANDMARK_COLOR` ("#F0C808") / `INTERSECTION_COLOR`
#: ("#FFFFFF") as RGB, so the gallery reads consistently with the folium map.
_ROAD_COLOR = (240, 200, 8)
_INTERSECTION_COLOR = (255, 255, 255)

#: Longest side, in pixels, of a grid thumbnail.
THUMBNAIL_SIZE = 220

#: Alpha (0-255) baked into a thumbnail's label overlay - a fixed, readable
#: blend for quick scanning; the full viewer's slider covers every strength.
THUMBNAIL_LABEL_ALPHA = 140


def _read_imagery_rgb(imagery_path: Path) -> np.ndarray:
    """The first up-to-3 bands of an imagery GeoTIFF as an (H, W, 3) uint8 array."""
    with rasterio.open(imagery_path) as src:
        band_count = min(src.count, 3)
        data = src.read(list(range(1, band_count + 1)))
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        data = np.concatenate([data, data[:1]], axis=0)
    if data.dtype != np.uint8:
        peak = float(data.max()) or 1.0
        data = (data.astype(np.float64) / peak * 255).astype(np.uint8)
    return np.moveaxis(data, 0, -1)


def _label_rgba(label: PanopticLabel, alpha: int = 255) -> np.ndarray:
    """``label``'s semantic band as an (H, W, 4) RGBA array - transparent where background."""
    height, width = label.semantic.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    road = label.semantic == int(PanopticClass.ROAD)
    intersection = label.semantic == int(PanopticClass.INTERSECTION)
    rgba[road, :3] = _ROAD_COLOR
    rgba[road, 3] = alpha
    rgba[intersection, :3] = _INTERSECTION_COLOR
    rgba[intersection, 3] = alpha
    return rgba


@dataclass(frozen=True)
class GalleryTile:
    """One tile's entry in the gallery's embedded manifest - paths are relative to ``index.html``."""

    stem: str
    road_count: int
    intersection_count: int
    default_width_count: int
    total_segments: int
    thumb: str
    imagery: str
    label_png: str
    #: One entry per rasterized road instance - {objectid, name, width_m,
    #: default_width} - so a reviewer can see exactly which CSJ OBJECTID(s)
    #: contributed to a tile (and which fell back to the default width)
    #: without leaving the gallery page. Not included for intersections,
    #: which have no OBJECTID of their own (they're derived junctions of the
    #: roads meeting there, already listed here individually).
    segments: tuple[dict[str, Any], ...] = ()


def render_tile_images(
    label: PanopticLabel,
    imagery_path: str | Path,
    output_dir: str | Path,
    thumbnail_size: int = THUMBNAIL_SIZE,
    overwrite: bool = False,
) -> GalleryTile:
    """Write one tile's imagery/label/thumbnail PNGs under ``output_dir`` and describe them.

    ``overwrite=False`` (the default, matching ``scripts/build_ground_truth.py``'s own
    ``--overwrite`` convention) skips the imagery read and PNG/thumbnail
    encoding entirely when all three output files already exist, returning
    the same `GalleryTile` a fresh render would - a resumed run over a very
    large label set (a full-AOI gallery can mean hundreds of thousands of
    files) picks up where an interrupted one left off instead of redoing
    already-finished tiles.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    thumbs_dir = output_dir / "thumbs"
    images_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    imagery_rel = f"images/{label.stem}_imagery.png"
    label_rel = f"images/{label.stem}_label.png"
    thumb_rel = f"thumbs/{label.stem}.png"
    road_segments = [s for s in label.segments if s.class_id == int(PanopticClass.ROAD)]
    gallery_tile = GalleryTile(
        stem=label.stem,
        road_count=len(road_segments),
        intersection_count=sum(1 for s in label.segments if s.class_id == int(PanopticClass.INTERSECTION)),
        default_width_count=sum(1 for s in label.segments if s.default_width_used),
        total_segments=len(label.segments),
        thumb=thumb_rel,
        imagery=imagery_rel,
        label_png=label_rel,
        segments=tuple(
            {
                "objectid": s.segment_id or "",
                "name": s.name or "",
                "width_m": round(s.width_m, 1) if s.width_m else None,
                "default_width": s.default_width_used,
            }
            for s in sorted(road_segments, key=lambda s: s.segment_id or "")
        ),
    )

    already_rendered = (
        (output_dir / imagery_rel).exists()
        and (output_dir / label_rel).exists()
        and (output_dir / thumb_rel).exists()
    )
    if already_rendered and not overwrite:
        return gallery_tile

    imagery_rgb = _read_imagery_rgb(Path(imagery_path))
    imagery_img = Image.fromarray(imagery_rgb, mode="RGB")
    label_img = Image.fromarray(_label_rgba(label), mode="RGBA")
    imagery_img.save(output_dir / imagery_rel)
    label_img.save(output_dir / label_rel)

    thumb_label = Image.fromarray(_label_rgba(label, alpha=THUMBNAIL_LABEL_ALPHA), mode="RGBA")
    blended = Image.alpha_composite(imagery_img.convert("RGBA"), thumb_label).convert("RGB")
    blended.thumbnail((thumbnail_size, thumbnail_size))
    blended.save(output_dir / thumb_rel)

    return gallery_tile


def _safe_json(value: object) -> str:
    """``json.dumps`` with markup-significant characters neutralized for safe inline-``<script>`` embedding.

    Same escaping `csnav.viz.window_selector._js_string` uses: ``json.dumps``
    alone leaves ``<``/``>``/``&`` untouched, so a value containing
    ``</script>`` would close the element early.
    """
    backslash = chr(92)
    encoded = json.dumps(value)
    for character, escape in (("<", "003c"), (">", "003e"), ("&", "0026")):
        encoded = encoded.replace(character, backslash + "u" + escape)
    return encoded


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<header class="gt-header">
  <h1>{title}</h1>
  <div class="gt-summary" id="gt-summary"></div>
  <button id="gt-export">Export flagged tiles</button>
</header>
<main class="gt-main">
  <section class="gt-viewer">
    <div class="gt-stage">
      <img id="gt-imagery" alt="imagery">
      <img id="gt-label" alt="label overlay">
    </div>
    <div class="gt-controls">
      <button id="gt-prev">&larr; prev</button>
      <span id="gt-position"></span>
      <button id="gt-next">next &rarr;</button>
      <label class="gt-flag"><input type="checkbox" id="gt-flag-box"> flag this tile</label>
    </div>
    <div class="gt-controls">
      <label>overlay opacity
        <input type="range" id="gt-opacity" min="0" max="100" value="60">
      </label>
      <button data-opacity="0">imagery only</button>
      <button data-opacity="60">overlay</button>
      <button data-opacity="100">labels only</button>
    </div>
    <div class="gt-info" id="gt-info"></div>
  </section>
  <section class="gt-grid" id="gt-grid"></section>
</main>
<script>
var TILES = {tiles_json};
{script}
</script>
</body>
</html>
"""

_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0; font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: #222; background: #fafafa;
}
.gt-header {
  display: flex; align-items: center; gap: 16px; padding: 10px 16px;
  background: #fff; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 2;
}
.gt-header h1 { font-size: 16px; margin: 0; flex: none; }
.gt-summary { color: #555; flex: 1; }
#gt-export {
  font: inherit; padding: 6px 12px; border: 1px solid #999; border-radius: 4px; background: #fff; cursor: pointer;
}
.gt-main { display: flex; gap: 16px; padding: 16px; align-items: flex-start; }
.gt-viewer { flex: 2; min-width: 360px; }
.gt-stage {
  position: relative; width: 100%; aspect-ratio: 1 / 1; background: #111;
  border-radius: 4px; overflow: hidden;
}
.gt-stage img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
#gt-label { opacity: 0.6; }
.gt-controls { display: flex; align-items: center; gap: 10px; margin-top: 8px; flex-wrap: wrap; }
.gt-controls button {
  font: inherit; padding: 5px 10px; border: 1px solid #999; border-radius: 4px; background: #fff; cursor: pointer;
}
.gt-flag { margin-left: auto; }
.gt-info { margin-top: 8px; color: #444; }
.gt-segments-wrap { max-height: 220px; overflow-y: auto; margin-top: 6px; border: 1px solid #eee; border-radius: 4px; }
.gt-segments { border-collapse: collapse; width: 100%; font-size: 12px; }
.gt-segments th, .gt-segments td { text-align: left; padding: 2px 6px; border-bottom: 1px solid #eee; }
.gt-segments thead th { position: sticky; top: 0; background: #fafafa; }
.gt-segments tr.gt-default-width { background: #fff4e5; }
.gt-grid {
  flex: 3; display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px; max-height: 82vh; overflow-y: auto; padding: 4px;
}
.gt-thumb {
  position: relative; cursor: pointer; border: 3px solid transparent; border-radius: 4px; overflow: hidden;
}
.gt-thumb img { display: block; width: 100%; height: auto; }
.gt-thumb.current { border-color: #0072B2; }
.gt-thumb.flagged { border-color: #D55E00; }
.gt-thumb .gt-thumb-label {
  position: absolute; left: 2px; bottom: 2px; background: rgba(0,0,0,0.6); color: #fff;
  font-size: 10px; padding: 1px 4px; border-radius: 2px;
}
"""

_SCRIPT = """
(function () {
  var STORAGE_KEY = 'csnav-ground-truth-gallery-flags';
  var flags = {};
  try {
    flags = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}');
  } catch (err) {
    flags = {};
  }

  var current = 0;
  var grid = document.getElementById('gt-grid');
  var imageryEl = document.getElementById('gt-imagery');
  var labelEl = document.getElementById('gt-label');
  var opacityEl = document.getElementById('gt-opacity');
  var positionEl = document.getElementById('gt-position');
  var infoEl = document.getElementById('gt-info');
  var flagBox = document.getElementById('gt-flag-box');
  var summaryEl = document.getElementById('gt-summary');

  function saveFlags() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(flags));
    } catch (err) { /* private-browsing or storage disabled: flags just won't persist */ }
  }

  function flagCount() {
    var count = 0;
    for (var key in flags) { if (flags[key]) { count += 1; } }
    return count;
  }

  function updateSummary() {
    summaryEl.textContent = TILES.length + ' tile(s), ' + flagCount() + ' flagged';
  }

  function thumbEl(index) {
    return grid.children[index];
  }

  function escapeHtml(value) {
    var entities = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
    return String(value).replace(/[&<>"']/g, function (ch) { return entities[ch]; });
  }

  function segmentRows(segments) {
    return (segments || []).map(function (seg) {
      return '<tr' + (seg.default_width ? ' class="gt-default-width"' : '') + '>' +
        '<td>' + escapeHtml(seg.objectid) + '</td>' +
        '<td>' + escapeHtml(seg.name || '(unnamed)') + '</td>' +
        '<td>' + (seg.width_m != null ? seg.width_m + ' m' : '—') + '</td>' +
        '<td>' + (seg.default_width ? 'yes' : 'no') + '</td>' +
        '</tr>';
    }).join('');
  }

  function render() {
    var tile = TILES[current];
    imageryEl.src = tile.imagery;
    labelEl.src = tile.label_png;
    positionEl.textContent = (current + 1) + ' / ' + TILES.length;
    var summary = tile.stem + ' - ' + tile.road_count + ' road instance(s), ' +
      tile.intersection_count + ' intersection(s)' +
      (tile.default_width_count ? ', ' + tile.default_width_count + ' using the default width' : '');
    var rows = segmentRows(tile.segments);
    infoEl.innerHTML = '<div>' + escapeHtml(summary) + '</div>' + (rows
      ? '<div class="gt-segments-wrap"><table class="gt-segments"><thead><tr><th>OBJECTID</th>' +
        '<th>name</th><th>width</th><th>default width?</th></tr></thead><tbody>' + rows +
        '</tbody></table></div>'
      : '');
    flagBox.checked = !!flags[tile.stem];
    for (var i = 0; i < grid.children.length; i++) {
      grid.children[i].classList.toggle('current', i === current);
    }
    updateSummary();
  }

  function goTo(index) {
    current = (index + TILES.length) % TILES.length;
    render();
  }

  TILES.forEach(function (tile, index) {
    var cell = document.createElement('div');
    cell.className = 'gt-thumb' + (flags[tile.stem] ? ' flagged' : '');
    cell.innerHTML = '<img loading="lazy" src="' + tile.thumb + '" alt="' + tile.stem + '">' +
      '<span class="gt-thumb-label">' + tile.stem + '</span>';
    cell.addEventListener('click', function () { goTo(index); });
    grid.appendChild(cell);
  });

  document.getElementById('gt-prev').addEventListener('click', function () { goTo(current - 1); });
  document.getElementById('gt-next').addEventListener('click', function () { goTo(current + 1); });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'ArrowLeft') { goTo(current - 1); }
    if (event.key === 'ArrowRight') { goTo(current + 1); }
  });

  opacityEl.addEventListener('input', function () {
    labelEl.style.opacity = (parseInt(opacityEl.value, 10) / 100).toString();
  });
  Array.prototype.forEach.call(document.querySelectorAll('.gt-controls button[data-opacity]'), function (button) {
    button.addEventListener('click', function () {
      opacityEl.value = button.getAttribute('data-opacity');
      labelEl.style.opacity = (parseInt(opacityEl.value, 10) / 100).toString();
    });
  });

  flagBox.addEventListener('change', function () {
    var tile = TILES[current];
    flags[tile.stem] = flagBox.checked;
    saveFlags();
    thumbEl(current).classList.toggle('flagged', flagBox.checked);
    updateSummary();
  });

  document.getElementById('gt-export').addEventListener('click', function () {
    var lines = [];
    for (var key in flags) { if (flags[key]) { lines.push(key); } }
    var blob = new Blob([lines.join('\\n') + '\\n'], {type: 'text/plain'});
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = 'flagged_tiles.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  });

  if (TILES.length > 0) { goTo(0); }
})();
"""


#: Sidecar next to ``index.html`` recording every tile the gallery has ever
#: listed - what lets `write_gallery` merge instead of replace (see below).
TILES_MANIFEST_FILENAME = "tiles.json"


def _load_existing_tiles(output_dir: Path) -> dict[str, GalleryTile]:
    """Previously-written tiles (by stem), from this gallery's own manifest sidecar - {} if none yet."""
    manifest_path = output_dir / TILES_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiles: dict[str, GalleryTile] = {}
    for entry in raw:
        entry = dict(entry)
        entry["segments"] = tuple(entry.get("segments") or ())
        tile = GalleryTile(**entry)
        tiles[tile.stem] = tile
    return tiles


def write_gallery(
    tiles: Sequence[GalleryTile], output_dir: str | Path, title: str = "Ground truth QA gallery"
) -> Path:
    """Write ``index.html`` for a gallery whose per-tile PNGs are already under ``output_dir``.

    ``tiles`` is normally the list :func:`render_tile_images` returned for
    each label, in the order the gallery should present them - but this
    *merges* them (by ``stem``, a fresh entry overriding a same-stem existing
    one) into whatever this gallery directory's own `TILES_MANIFEST_FILENAME`
    sidecar already lists, rather than replacing the page outright.

    That distinction matters because a later run against the same
    ``output_dir`` is not always the same selection as before: a smaller
    ``--limit``/``--sample`` re-run (`scripts/visualize_ground_truth.py`),
    or one scoped to a different ``--manifest``, still leaves every earlier
    run's already-rendered PNGs sitting on disk under this same directory.
    Without merging, that later run's ``index.html`` would list *only* its
    own (smaller) selection - silently dropping every previously-included
    tile from the page even though its images are still right there on disk
    (a real incident: a full run followed by a ``--limit`` re-run left most
    of a label set's thumbnails unreachable from the gallery page, looking
    exactly like a rendering bug rather than the truncation it actually
    was). Always writing the union instead is what makes
    "re-invoke with the same/a smaller/a differently-scoped selection to
    resume or extend" actually safe, rather than only safe for the exact
    same selection every time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = _load_existing_tiles(output_dir)
    for tile in tiles:
        merged[tile.stem] = tile
    ordered = sorted(merged.values(), key=lambda t: t.stem)

    manifest_path = output_dir / TILES_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps([asdict(tile) for tile in ordered], indent=2), encoding="utf-8")

    html = _PAGE_TEMPLATE.format(
        title=title,
        css=_CSS,
        script=_SCRIPT,
        tiles_json=_safe_json([asdict(tile) for tile in ordered]),
    )
    destination = output_dir / "index.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def build_gallery(
    labels_and_imagery: Iterable[tuple[PanopticLabel, str | Path]],
    output_dir: str | Path,
    title: str = "Ground truth QA gallery",
    thumbnail_size: int = THUMBNAIL_SIZE,
    overwrite: bool = False,
) -> Path:
    """Render every tile's images and write the gallery page, in one call.

    ``labels_and_imagery`` pairs each `PanopticLabel` with its source imagery
    GeoTIFF path; sort by ``label.stem`` before calling for a stable tile
    order. Consumed as a single-pass iterator - each pair's images are
    rendered and written to disk (:func:`render_tile_images`) before the next
    pair is produced, so passing a generator that loads one `PanopticLabel`
    from disk at a time (see ``scripts/visualize_ground_truth.py``) keeps at
    most one tile's full-resolution rasters in memory regardless of how many
    tiles are in the set.

    ``overwrite=False`` (the default) skips re-rendering a tile whose three
    output PNGs already exist - a full-AOI gallery can mean hundreds of
    thousands of files, so being resumable after an interrupted run matters
    here in a way it wouldn't for a small one. This call's own ``tiles`` is
    only the *current* selection, though - :func:`write_gallery` merges it
    with whatever this ``output_dir`` already listed, so a later call scoped
    to fewer/different tiles (a smaller ``--limit``/``--sample``, a
    different ``--manifest``) still leaves every earlier call's tiles
    visible on the page rather than silently dropping them.
    """
    output_dir = Path(output_dir)
    tiles = [
        render_tile_images(label, imagery_path, output_dir, thumbnail_size=thumbnail_size, overwrite=overwrite)
        for label, imagery_path in labels_and_imagery
    ]
    return write_gallery(tiles, output_dir, title=title)
