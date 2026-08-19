"""Discovery of ArcGIS REST services, including historic imagery layers.

San Jose's ArcGIS Server (geo.sanjoseca.gov) publishes each vintage of aerial
imagery as its own service under the ``Imagery`` folder (the current cached
basemap plus one service per historic capture, e.g. flown years). Training
data for this project needs every vintage, not just the latest, so this
module walks the REST catalog recursively and returns *every* matching
service rather than assuming a single well-known name.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import requests

from .models import ServiceRef

DEFAULT_BASE_URL = "https://geo.sanjoseca.gov/server/rest/services"

_YEAR_RE = re.compile(r"(19|20)\d{2}")


class ArcGISCatalogError(RuntimeError):
    """Raised when the ArcGIS REST catalog returns an error payload or bad data."""


def extract_year(name: str) -> int | None:
    """Best-effort extraction of a 4-digit capture year from a service name."""
    match = _YEAR_RE.search(name)
    return int(match.group(0)) if match else None


class ArcGISCatalog:
    """Client for the ArcGIS Server REST *services directory* (not a single service).

    Example::

        catalog = ArcGISCatalog()
        services = catalog.discover_imagery_services()
        for ref in services:
            print(ref.full_name, extract_year(ref.name), catalog.service_rest_url(ref))
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get_json(self, folder: str) -> dict:
        url = f"{self.base_url}/{folder}".rstrip("/") if folder else self.base_url
        resp = self.session.get(url, params={"f": "json"}, timeout=self.timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as exc:
            raise ArcGISCatalogError(f"non-JSON response from {url}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise ArcGISCatalogError(f"ArcGIS error listing {url}: {data['error']}")
        return data

    def list_folder(self, folder: str = "") -> tuple[list[str], list[ServiceRef]]:
        """Return ``(sub_folder_paths, services)`` directly under ``folder``."""
        data = self._get_json(folder)

        sub_folders = [f"{folder}/{name}" if folder else name for name in data.get("folders") or []]

        services: list[ServiceRef] = []
        for svc in data.get("services") or []:
            full_name = svc["name"]
            svc_type = svc["type"]
            if "/" in full_name:
                parent, short_name = full_name.rsplit("/", 1)
            else:
                parent, short_name = "", full_name
            services.append(ServiceRef(folder=parent, name=short_name, service_type=svc_type))

        return sub_folders, services

    def walk(self, root: str = "") -> Iterator[ServiceRef]:
        """Recursively yield every service reachable under ``root``."""
        pending = [root]
        visited: set[str] = set()
        while pending:
            folder = pending.pop()
            if folder in visited:
                continue
            visited.add(folder)
            sub_folders, services = self.list_folder(folder)
            yield from services
            pending.extend(sub_folders)

    def discover_imagery_services(
        self,
        root: str = "Imagery",
        name_contains: str = "DPW_Imagery",
        service_types: tuple[str, ...] = ("MapServer", "ImageServer"),
    ) -> list[ServiceRef]:
        """Find every imagery service under ``root`` whose name matches.

        This intentionally does not stop at the most recent vintage: it
        returns *all* matches (``DPW_ImageryCached`` plus any dated historic
        services such as ``DPW_Imagery_2012``) so downstream training-data
        collection can pull from the full historic archive.
        """
        needle = name_contains.lower()
        matches = [
            ref
            for ref in self.walk(root)
            if ref.service_type in service_types and needle in ref.name.lower()
        ]
        # Most recent first is a convenient default order, but nothing here
        # discards older vintages - callers get the full list.
        matches.sort(key=lambda ref: (extract_year(ref.name) or -1, ref.name), reverse=True)
        return matches

    def service_rest_url(self, ref: ServiceRef) -> str:
        path = f"{ref.folder}/{ref.name}/{ref.service_type}" if ref.folder else f"{ref.name}/{ref.service_type}"
        return f"{self.base_url}/{path}"
