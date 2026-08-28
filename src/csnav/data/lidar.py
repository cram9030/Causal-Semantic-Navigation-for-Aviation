"""Client for ground elevation via USGS 3DEP's national elevation ImageServer.

San Jose's own "Imagery & Elevation" LIDAR product (Valley Water,
`gis.valleywater.org`) turned out, on inspection of a real download, to be
**contour lines** (a single `MultiLineString` layer inside an Esri File
Geodatabase), not a raster DEM - `ogrinfo`/`gdalinfo` against the extracted
archive confirmed there is no raster/mosaic dataset in there at all. Getting
an elevation *surface* value at an arbitrary `(lon, lat)` out of contour
lines needs interpolation (TIN, IDW, ...), which is real, unbuilt work with
its own accuracy tradeoffs for the 200-4000 ft AGL operating envelope this
project targets - not something to build silently. See
`docs/phase0_csj_streets_lidar.md` for that investigation; an earlier
version of this module tried to download + read that archive directly.

Instead, this module uses USGS's 3D Elevation Program (3DEP) national
elevation mosaic - a live ArcGIS ImageServer USGS maintains specifically for
this kind of programmatic per-AOI/per-point access::

    https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer

Unlike San Jose's own ArcGIS catalog (`csnav.data.arcgis`), there's no
discovery needed here - it's a fixed, publicly documented federal endpoint,
not a service whose name/location can move under a generic catalog folder.
And unlike the Valley Water archive, there's no local
download/extract/cache step either: every `read_window()`/`identify()` call
is a live per-request raster query, requesting/specifying EPSG:4326
throughout (`bboxSR`/`imageSR` for `read_window`; an embedded
`spatialReference` on the point geometry for `identify` - see its
docstring for why a bare `sr` param doesn't work) so the service
reprojects server-side - no client-side `rasterio.warp` step needed here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from .arcgis.models import Extent
from .arcgis.reproject import ReprojectedTile

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"

OUTPUT_WKID = 4326
OUTPUT_CRS = f"EPSG:{OUTPUT_WKID}"


class LidarElevationError(RuntimeError):
    """Raised when the elevation service returns an error payload or bad data."""


@dataclass(frozen=True)
class ImageServerMetadata:
    service_url: str
    extent: Extent | None
    pixel_size_x: float | None
    pixel_size_y: float | None
    pixel_type: str | None


def _extent_from_json(raw: dict[str, Any] | None) -> Extent | None:
    if not raw:
        return None
    sr = raw.get("spatialReference", {})
    wkid = sr.get("latestWkid") or sr.get("wkid") or OUTPUT_WKID
    return Extent(xmin=raw["xmin"], ymin=raw["ymin"], xmax=raw["xmax"], ymax=raw["ymax"], wkid=wkid)


class LidarElevationClient:
    """Client for USGS 3DEP's national elevation ImageServer, always in EPSG:4326."""

    def __init__(
        self,
        service_url: str = DEFAULT_SERVICE_URL,
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
        try:
            data = resp.json()
        except ValueError as exc:
            raise LidarElevationError(f"non-JSON response from {self.service_url}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise LidarElevationError(f"ArcGIS error for {self.service_url}: {data['error']}")

        self._metadata = ImageServerMetadata(
            service_url=self.service_url,
            extent=_extent_from_json(data.get("extent") or data.get("fullExtent")),
            pixel_size_x=data.get("pixelSizeX"),
            pixel_size_y=data.get("pixelSizeY"),
            pixel_type=data.get("pixelType"),
        )
        return self._metadata

    def identify(self, lon: float, lat: float) -> float | None:
        """Ground elevation at a single point, or ``None`` where the service has no data.

        The point's spatial reference must be embedded *in* the ``geometry``
        JSON object, not passed as a separate ``sr`` query param - a bare
        ``sr`` alongside a plain ``"lon,lat"`` string is silently ignored by
        this operation (confirmed against the live service: the point comes
        back echoed under the service's native Web Mercator SR instead of
        the EPSG:4326 it was given, which reads as a real coordinate only by
        accident - most inputs land far outside any coverage and misreport
        as ``NoData`` instead of erroring loudly).
        """
        geometry = json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": OUTPUT_WKID}})
        params = {
            "f": "json",
            "geometry": geometry,
            "geometryType": "esriGeometryPoint",
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

    def read_window(
        self,
        bbox: Extent,
        width: int = 512,
        height: int = 512,
        pixel_type: str = "F32",
    ) -> ReprojectedTile:
        """Georeferenced elevation raster over ``bbox`` (EPSG:4326), at ``width``x``height`` pixels."""
        if bbox.wkid != OUTPUT_WKID:
            raise ValueError(f"bbox must be EPSG:{OUTPUT_WKID}, got wkid={bbox.wkid}")

        params = {
            "f": "image",
            "bbox": f"{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax}",
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

        # The transform is built from the *requested* bbox/size rather than
        # trusted from whatever georeferencing the returned TIFF embeds - the
        # same "always supply our own transform" approach
        # `reproject_tile_to_4326` uses for imagery tiles.
        transform = from_bounds(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, width, height)
        with MemoryFile(resp.content) as memfile, memfile.open() as src:
            data = src.read()

        return ReprojectedTile(data=data, transform=transform, crs=OUTPUT_CRS, width=width, height=height)
