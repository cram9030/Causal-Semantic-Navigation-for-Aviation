"""Data models shared by the ArcGIS REST catalog and tile client."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServiceRef:
    """Reference to a single service found while walking an ArcGIS REST catalog."""

    folder: str
    name: str
    service_type: str

    @property
    def full_name(self) -> str:
        return f"{self.folder}/{self.name}" if self.folder else self.name


@dataclass(frozen=True)
class LevelOfDetail:
    level: int
    resolution: float  # spatial-reference units per pixel
    scale: float


@dataclass(frozen=True)
class TileInfo:
    rows: int
    cols: int
    image_format: str
    origin_x: float
    origin_y: float
    wkid: int
    lods: tuple[LevelOfDetail, ...] = field(default_factory=tuple)

    def lod_for_level(self, level: int) -> LevelOfDetail:
        for lod in self.lods:
            if lod.level == level:
                return lod
        raise KeyError(f"level {level} not present in tile scheme (have: "
                        f"{sorted(lod.level for lod in self.lods)})")

    @property
    def max_level(self) -> int:
        return max(lod.level for lod in self.lods)


@dataclass(frozen=True)
class Extent:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    wkid: int = 4326


@dataclass(frozen=True)
class ServiceMetadata:
    service_url: str
    capabilities: tuple[str, ...]
    tile_info: TileInfo | None
    full_extent: Extent | None
    wmts_url: str | None = None

    @property
    def supports_tiles(self) -> bool:
        return self.tile_info is not None

    @property
    def supports_wmts(self) -> bool:
        return self.wmts_url is not None

    @property
    def supports_export(self) -> bool:
        return "Map" in self.capabilities or "Image" in self.capabilities
