"""Client for San Jose's LIDAR-derived elevation product.

Unlike CSJ Streets and San Jose's own aerial imagery (both served through
`geo.sanjoseca.gov`'s ArcGIS Server - see `csnav.data.arcgis`), the
county-wide LIDAR DEM behind San Jose's "Imagery & Elevation" data isn't an
ArcGIS service at all: Valley Water (Santa Clara Valley Water District)
publishes it as two static ZIP downloads, one per resolution::

    https://gis.valleywater.org/Download/LiDAR1FT.zip   (1 ft/px)
    https://gis.valleywater.org/Download/LiDAR5FT.zip   (5 ft/px)

There is no per-request bounding box or query endpoint to discover here -
the two URLs are fixed and each covers the whole county, so unlike
`csnav.data.arcgis`, this module does no catalog discovery. Instead,
:class:`LidarElevationClient` downloads the chosen product once (cached
locally - a later call is a no-op unless ``overwrite=True``, since these are
large, whole-county archives, not something to re-fetch per query), extracts
it, and serves every AOI/point elevation query after that as a local
windowed read against whatever raster(s) the archive contains - no live
network call per query. Results are always returned in EPSG:4326 regardless
of the source raster's native CRS (commonly a state-plane or UTM CRS for
this kind of county GIS data); `read_elevation_window` reprojects on read if
needed, per CLAUDE.md's "never assume a source is already in EPSG:4326" rule.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform, transform_bounds
from rasterio.windows import Window
from tqdm import tqdm

from .arcgis.models import Extent
from .arcgis.reproject import ReprojectedTile

logger = logging.getLogger(__name__)

LIDAR_PRODUCT_URLS = {
    "1ft": "https://gis.valleywater.org/Download/LiDAR1FT.zip",
    "5ft": "https://gis.valleywater.org/Download/LiDAR5FT.zip",
}

RASTER_EXTENSIONS = (".tif", ".tiff", ".img", ".asc", ".adf")

OUTPUT_WKID = 4326
OUTPUT_CRS = f"EPSG:{OUTPUT_WKID}"


class LidarElevationError(RuntimeError):
    """Raised when a LIDAR archive can't be fetched, extracted, or queried."""


def download_archive(
    url: str,
    dest_path: Path,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    chunk_size: int = 1 << 20,
    overwrite: bool = False,
) -> Path:
    """Stream ``url`` to ``dest_path``, skipping the download if it already exists.

    Writes to a temp file and renames into place only once complete, so an
    interrupted download never leaves a partial file that a later call would
    mistake for a finished one - the same pattern `fetch_historic_imagery.py`
    uses for individual tiles, applied here to one (potentially very large)
    whole-county archive.
    """
    if dest_path.exists() and not overwrite:
        logger.info("already downloaded: %s", dest_path)
        return dest_path

    session = session or requests.Session()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")

    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name, leave=False)
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))
        progress.close()

    tmp_path.replace(dest_path)
    return dest_path


def extract_archive(zip_path: Path, dest_dir: Path, overwrite: bool = False) -> list[Path]:
    """Extract every member of ``zip_path`` into ``dest_dir``; return the raster files found."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".extracted"
    if marker.exists() and not overwrite:
        logger.info("already extracted: %s", dest_dir)
        return find_raster_files(dest_dir)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    marker.write_text("")
    return find_raster_files(dest_dir)


def find_raster_files(root: Path) -> list[Path]:
    """Every file under ``root`` (recursively) with a recognized raster extension."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in RASTER_EXTENSIONS)


def _find_intersecting(raster_paths: list[Path], bbox: Extent) -> list[Path]:
    hits = []
    for path in raster_paths:
        with rasterio.open(path) as src:
            if src.crs is None:
                continue
            left, bottom, right, top = transform_bounds(src.crs, OUTPUT_CRS, *src.bounds)
        if left <= bbox.xmax and right >= bbox.xmin and bottom <= bbox.ymax and top >= bbox.ymin:
            hits.append(path)
    return hits


def read_elevation_window(raster_paths: list[Path], bbox: Extent) -> ReprojectedTile:
    """Read + mosaic whichever of ``raster_paths`` intersect ``bbox``, in EPSG:4326.

    ``bbox`` must already be EPSG:4326. Reprojects from the source raster(s)'
    native CRS if it isn't already 4326 - this client never assumes the
    downstream data is already in the storage CRS (see CLAUDE.md §2).
    Raises :class:`LidarElevationError` if no supplied raster covers ``bbox``.
    """
    if bbox.wkid != OUTPUT_WKID:
        raise ValueError(f"bbox must be EPSG:{OUTPUT_WKID}, got wkid={bbox.wkid}")

    intersecting = _find_intersecting(raster_paths, bbox)
    if not intersecting:
        raise LidarElevationError(f"no LIDAR tile covers {bbox}")

    datasets = [rasterio.open(p) for p in intersecting]
    try:
        src_crs = datasets[0].crs
        left, bottom, right, top = transform_bounds(OUTPUT_CRS, src_crs, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
        mosaic, mosaic_transform = rio_merge(datasets, bounds=(left, bottom, right, top))
    finally:
        for ds in datasets:
            ds.close()

    if src_crs.to_epsg() == OUTPUT_WKID:
        return ReprojectedTile(
            data=mosaic, transform=mosaic_transform, crs=OUTPUT_CRS,
            width=mosaic.shape[-1], height=mosaic.shape[-2],
        )

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, OUTPUT_CRS, mosaic.shape[-1], mosaic.shape[-2], left, bottom, right, top,
    )
    dst_data = np.zeros((mosaic.shape[0], dst_height, dst_width), dtype=mosaic.dtype)
    for band in range(mosaic.shape[0]):
        reproject(
            source=mosaic[band], destination=dst_data[band],
            src_transform=mosaic_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=OUTPUT_CRS,
            resampling=Resampling.bilinear,
        )
    return ReprojectedTile(data=dst_data, transform=dst_transform, crs=OUTPUT_CRS, width=dst_width, height=dst_height)


class LidarElevationClient:
    """Downloads + caches a Valley Water LIDAR DEM product, then serves local windowed reads.

    ``product`` is ``"1ft"`` or ``"5ft"`` (see :data:`LIDAR_PRODUCT_URLS`).
    ``cache_dir`` holds the downloaded archive and its extracted rasters
    across calls/runs, so repeated use (e.g. several AOI pulls in one
    trajectory-planning cycle) doesn't re-download a multi-gigabyte
    whole-county archive each time - :meth:`ensure_local` is a no-op once
    both already exist, unless ``overwrite=True``.
    """

    def __init__(
        self,
        cache_dir: Path,
        product: str = "5ft",
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        if product not in LIDAR_PRODUCT_URLS:
            raise ValueError(f"product must be one of {sorted(LIDAR_PRODUCT_URLS)}, got {product!r}")
        self.cache_dir = Path(cache_dir)
        self.product = product
        self.session = session or requests.Session()
        self.timeout = timeout
        self._raster_paths: list[Path] | None = None

    @property
    def archive_path(self) -> Path:
        return self.cache_dir / f"LiDAR{self.product.upper()}.zip"

    @property
    def extract_dir(self) -> Path:
        return self.cache_dir / self.product

    def ensure_local(self, overwrite: bool = False) -> list[Path]:
        """Download (if needed) and extract the archive; return the raster files found.

        Raises :class:`LidarElevationError` if extraction produced no file
        with a recognized raster extension (see :data:`RASTER_EXTENSIONS`).
        """
        download_archive(
            LIDAR_PRODUCT_URLS[self.product], self.archive_path,
            session=self.session, timeout=self.timeout, overwrite=overwrite,
        )
        self._raster_paths = extract_archive(self.archive_path, self.extract_dir, overwrite=overwrite)
        if not self._raster_paths:
            raise LidarElevationError(
                f"{self.archive_path} extracted to {self.extract_dir} but no recognized raster "
                f"file ({', '.join(RASTER_EXTENSIONS)}) was found there"
            )
        return self._raster_paths

    def read_window(self, bbox: Extent) -> ReprojectedTile:
        """AOI elevation raster, in EPSG:4326. Downloads/extracts the archive on first use."""
        paths = self._raster_paths or self.ensure_local()
        return read_elevation_window(paths, bbox)

    def identify(self, lon: float, lat: float) -> float | None:
        """Single-point elevation via a direct 1x1 pixel read; ``None`` where no tile covers it.

        Reads exactly the pixel containing ``(lon, lat)`` out of whichever
        source raster covers it, rather than building a small bbox and
        reusing :meth:`read_window` - a bbox epsilon small enough to stay a
        "point" can still end up narrower than the source raster's own pixel
        size (this DEM's pixels can be tens of meters wide), which would
        make :func:`~rasterio.merge.merge` round the output window to zero
        pixels. Indexing directly into the source raster has no such issue.
        """
        paths = self._raster_paths or self.ensure_local()
        point = Extent(xmin=lon, ymin=lat, xmax=lon, ymax=lat, wkid=OUTPUT_WKID)
        for path in _find_intersecting(paths, point):
            with rasterio.open(path) as src:
                xs, ys = transform(OUTPUT_CRS, src.crs, [lon], [lat])
                row, col = src.index(xs[0], ys[0])
                if not (0 <= row < src.height and 0 <= col < src.width):
                    continue
                value = src.read(1, window=Window(col, row, 1, 1))[0, 0]
                if src.nodata is not None and value == src.nodata:
                    continue
                value = float(value)
                return None if np.isnan(value) else value
        return None
