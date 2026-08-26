"""Reproject fetched ArcGIS tiles from EPSG:3857 (Web Mercator) to EPSG:4326.

Tiles come back from the client as raw PNG/JPEG bytes with no embedded
georeferencing, so the caller supplies the tile's known bounds (computed via
:func:`csnav.data.arcgis.tiles.tile_bounds`) and this module builds the
source transform itself before warping.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

from .models import Extent
from .projections import WEB_MERCATOR_WKIDS

DST_CRS = "EPSG:4326"
SRC_CRS = "EPSG:3857"


@dataclass
class ReprojectedTile:
    data: np.ndarray  # shape (bands, height, width)
    transform: rasterio.Affine
    crs: str
    width: int
    height: int

    def to_geotiff(self, path: str | Path) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=self.data.shape[0],
            dtype=self.data.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as dst:
            dst.write(self.data)


def reproject_tile_to_4326(
    image_bytes: bytes,
    bounds_3857: Extent,
    resampling: Resampling = Resampling.bilinear,
) -> ReprojectedTile:
    """Warp one fetched tile (image bytes + its Web Mercator bounds) to EPSG:4326."""
    if bounds_3857.wkid not in WEB_MERCATOR_WKIDS:
        raise ValueError(f"expected a Web Mercator extent, got wkid={bounds_3857.wkid}")

    # The source PNG/JPEG bytes carry no georeferencing of their own - we
    # always supply our own transform below (from the tile's known bounds),
    # so rasterio's warning that the file itself lacks one is expected noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(image_bytes) as memfile, memfile.open() as src:
            src_transform = from_bounds(
                bounds_3857.xmin, bounds_3857.ymin, bounds_3857.xmax, bounds_3857.ymax,
                src.width, src.height,
            )

            # calculate_default_transform wants the source-CRS bounds; it derives
            # the destination transform/size itself.
            dst_transform, dst_width, dst_height = calculate_default_transform(
                SRC_CRS, DST_CRS, src.width, src.height,
                bounds_3857.xmin, bounds_3857.ymin, bounds_3857.xmax, bounds_3857.ymax,
            )

            dst_data = np.zeros((src.count, dst_height, dst_width), dtype=src.dtypes[0])
            for band in range(1, src.count + 1):
                reproject(
                    source=src.read(band),
                    destination=dst_data[band - 1],
                    src_transform=src_transform,
                    src_crs=SRC_CRS,
                    dst_transform=dst_transform,
                    dst_crs=DST_CRS,
                    resampling=resampling,
                )

    return ReprojectedTile(
        data=dst_data,
        transform=dst_transform,
        crs=DST_CRS,
        width=dst_width,
        height=dst_height,
    )
