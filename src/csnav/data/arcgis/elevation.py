"""Client for San Jose's Imagery & Elevation LIDAR raster service.

Exposed by ArcGIS Server as an ImageServer, this is the source for
ground-elevation values used to correct GPS-derived height to true AGL and to
support FOV occlusion modeling (see `docs/INTEGRATION_PLAN.md` §2, §3.2) -
not just visualization. Resolve the service URL with
:meth:`csnav.data.arcgis.catalog.ArcGISCatalog.discover_services` rather than
hardcoding it, for the same reason imagery services are discovered rather
than assumed (see `docs/phase0_arcgis_tile_client.md`).

Two access patterns are supported:

* :meth:`LidarElevationClient.identify` - a single ``(lon, lat)`` point
  elevation, for one-off AGL corrections.
* :meth:`LidarElevationClient.export_elevation` (+ :func:`load_elevation_tile`)
  - a georeferenced raster over an AOI, for building the local ground-elevation
  surface an occlusion check needs.

Both request output directly in EPSG:4326 (``sr``/``bboxSR``/``imageSR``) -
unlike the Web Mercator-only imagery tile cache, this service can reproject
server-side, so no client-side ``rasterio.warp`` step is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from .models import Extent
from .reproject import ReprojectedTile

OUTPUT_WKID = 4326


class LidarElevationError(RuntimeError):
    """Raised when the elevation service returns an error payload or bad data."""


@dataclass(frozen=True)
class ImageServerMetadata:
    service_url: str
    extent: Extent | None
    pixel_size_x: float | None
    pixel_size_y: float | None
    band_count: int | None
    pixel_type: str | None


def _extent_from_json(raw: dict[str, Any] | None) -> Extent | None:
    if not raw:
        return None
    sr = raw.get("spatialReference", {})
    wkid = sr.get("latestWkid") or sr.get("wkid") or OUTPUT_WKID
    return Extent(xmin=raw["xmin"], ymin=raw["ymin"], xmax=raw["xmax"], ymax=raw["ymax"], wkid=wkid)


class LidarElevationClient:
    """Client for a single ArcGIS ImageServer elevation service, in EPSG:4326."""

    def __init__(
        self,
        service_url: str,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self._metadata: ImageServerMetadata | None = None

    def get_metadata(self, refresh: bool = False) -> ImageServerMetadata:
        if self._metadata is not None and not refresh:
            return self._metadata

        resp = self.session.get(self.service_url, params={"f": "json"}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise LidarElevationError(f"ArcGIS error for {self.service_url}: {data['error']}")

        self._metadata = ImageServerMetadata(
            service_url=self.service_url,
            extent=_extent_from_json(data.get("extent") or data.get("fullExtent")),
            pixel_size_x=data.get("pixelSizeX"),
            pixel_size_y=data.get("pixelSizeY"),
            band_count=data.get("bandCount"),
            pixel_type=data.get("pixelType"),
        )
        return self._metadata

    def identify(self, lon: float, lat: float) -> float | None:
        """Ground elevation at a single point, or ``None`` where the raster has no data."""
        params = {
            "f": "json",
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "sr": OUTPUT_WKID,
            "returnGeometry": "false",
        }
        resp = self.session.get(f"{self.service_url}/identify", params=params, timeout=self.timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as exc:
            raise LidarElevationError(f"non-JSON response from {self.service_url}/identify") from exc
        if isinstance(data, dict) and data.get("error"):
            raise LidarElevationError(f"ArcGIS error identifying {self.service_url}: {data['error']}")

        value = data.get("value")
        if value is None or str(value).strip().lower() in {"", "nodata"}:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise LidarElevationError(f"non-numeric elevation value {value!r} from {self.service_url}") from exc

    def export_elevation(
        self,
        extent: Extent,
        width: int,
        height: int,
        pixel_type: str = "F32",
    ) -> bytes:
        """Fetch a raw (single-band, georeferenced-by-caller) elevation raster over ``extent``.

        Returns the raw TIFF bytes from ``exportImage`` - pass them to
        :func:`load_elevation_tile` along with the same ``extent``/``width``/
        ``height`` to get a georeferenced array.
        """
        if extent.wkid != OUTPUT_WKID:
            raise ValueError(f"extent must be EPSG:{OUTPUT_WKID}, got wkid={extent.wkid}")

        params = {
            "f": "image",
            "bbox": f"{extent.xmin},{extent.ymin},{extent.xmax},{extent.ymax}",
            "bboxSR": OUTPUT_WKID,
            "imageSR": OUTPUT_WKID,
            "size": f"{width},{height}",
            "format": "tiff",
            "pixelType": pixel_type,
            "noDataInterpretation": "esriNoDataMatchAny",
            "interpolation": "RSP_BilinearInterpolation",
        }
        resp = self.session.get(f"{self.service_url}/exportImage", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content


def load_elevation_tile(image_bytes: bytes, extent: Extent, width: int, height: int) -> ReprojectedTile:
    """Decode ``exportImage`` bytes into a georeferenced raster tile.

    The transform is built from the *requested* ``extent``/``width``/``height``
    rather than trusted from whatever georeferencing (if any) the returned
    TIFF embeds - the same "we always supply our own transform" approach
    ``reproject_tile_to_4326`` uses for imagery tiles, since ``extent`` was
    already EPSG:4326 (the request's ``imageSR``) no CRS warp is needed here.
    """
    if extent.wkid != OUTPUT_WKID:
        raise ValueError(f"extent must be EPSG:{OUTPUT_WKID}, got wkid={extent.wkid}")

    transform = from_bounds(extent.xmin, extent.ymin, extent.xmax, extent.ymax, width, height)
    with MemoryFile(image_bytes) as memfile, memfile.open() as src:
        data = src.read()

    return ReprojectedTile(data=data, transform=transform, crs="EPSG:4326", width=width, height=height)
