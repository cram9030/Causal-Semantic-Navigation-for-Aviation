"""Discovery of ArcGIS REST services, including historic imagery layers.

San Jose's ArcGIS Server (geo.sanjoseca.gov) publishes each vintage of aerial
imagery as its own service under the ``Imagery`` folder (the current cached
basemap plus one service per historic capture, e.g. flown years). Training
data for this project needs every vintage, not just the latest, so this
module walks the REST catalog recursively and returns *every* matching
service rather than assuming a single well-known name.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import requests

from .models import ServiceRef

DEFAULT_BASE_URL = "https://geo.sanjoseca.gov/server/rest/services"

_YEAR_RE = re.compile(r"(19|20)\d{2}")

logger = logging.getLogger(__name__)


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
        return self._get_json_at(url)

    def _get_json_at(self, url: str) -> dict:
        """Same as :meth:`_get_json`, for a caller-supplied absolute REST URL."""
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
        """Recursively yield every service reachable under ``root``.

        A folder that fails to list with a plain HTTP 403/404 - some ArcGIS
        servers advertise folders in the directory JSON that turn out to be
        access-restricted or stale when actually requested - is logged and
        skipped rather than aborting the whole walk, since one bad folder
        shouldn't hide every service elsewhere in the tree. An ArcGIS error
        *payload* (a 200 response with an ``error`` body) still propagates,
        since that typically signals a real problem with the request itself
        rather than "this folder doesn't exist here".
        """
        pending = [root]
        visited: set[str] = set()
        while pending:
            folder = pending.pop()
            if folder in visited:
                continue
            visited.add(folder)
            try:
                sub_folders, services = self.list_folder(folder)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in (403, 404):
                    logger.warning("skipping catalog folder %r: HTTP %s", folder, status)
                    continue
                raise
            yield from services
            pending.extend(sub_folders)

    def discover_services(
        self,
        root: str = "",
        name_contains: str = "",
        service_types: tuple[str, ...] = ("MapServer", "FeatureServer", "ImageServer"),
    ) -> list[ServiceRef]:
        """Find every service under ``root`` whose name contains ``name_contains``.

        This is the generic building block behind :meth:`discover_imagery_services`
        and, elsewhere, ``CSJStreetsClient``/``LidarElevationClient`` discovery -
        it never assumes a single well-known service name, since San Jose
        reorganizes/renames services independently of this codebase.
        """
        needle = name_contains.lower()
        return [
            ref
            for ref in self.walk(root)
            if ref.service_type in service_types and needle in ref.name.lower()
        ]

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
        matches = self.discover_services(root=root, name_contains=name_contains, service_types=service_types)
        # Most recent first is a convenient default order, but nothing here
        # discards older vintages - callers get the full list.
        matches.sort(key=lambda ref: (extract_year(ref.name) or -1, ref.name), reverse=True)
        return matches

    def service_rest_url(self, ref: ServiceRef) -> str:
        path = f"{ref.folder}/{ref.name}/{ref.service_type}" if ref.folder else f"{ref.name}/{ref.service_type}"
        return f"{self.base_url}/{path}"

    def find_layers(
        self,
        layer_name_contains: str,
        root: str = "",
        service_name_contains: str = "",
        service_types: tuple[str, ...] = ("MapServer", "FeatureServer"),
    ) -> list[tuple[str, str]]:
        """Every sublayer whose name matches, as ``(layer_url, layer_name)`` pairs.

        :meth:`find_layer` returns only the *first* match, silently - fine when
        there is really only one, but San Jose's catalog can expose more than
        one layer whose name contains the same substring (e.g. a full-attribute
        "Streets" layer alongside a separate, sparser "Street Centerlines"
        reference/geocoding layer), and picking the wrong one by luck-of-
        iteration-order is exactly how a real ground-truth build ended up
        querying a layer with no width field at all after CSJ's catalog
        reorganized (see `docs/phase0_csj_streets_lidar.md`). This walks the
        same services :meth:`find_layer` does but collects every match instead
        of returning on the first one, so a caller can compare and pick
        deliberately via ``--layer-url`` rather than trust discovery blindly.
        """
        needle = layer_name_contains.lower()
        candidates = self.discover_services(root=root, name_contains=service_name_contains, service_types=service_types)
        matches: list[tuple[str, str]] = []
        for ref in candidates:
            service_url = self.service_rest_url(ref)
            try:
                data = self._get_json_at(service_url)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in (403, 404):
                    logger.warning("skipping service %r: HTTP %s", service_url, status)
                    continue
                raise
            for layer in data.get("layers") or []:
                name = str(layer.get("name", ""))
                if needle in name.lower():
                    matches.append((f"{service_url}/{layer['id']}", name))
        return matches

    def find_layer(
        self,
        layer_name_contains: str,
        root: str = "",
        service_name_contains: str = "",
        service_types: tuple[str, ...] = ("MapServer", "FeatureServer"),
    ) -> str:
        """Resolve a single sublayer's REST URL by name, without hardcoding its service.

        Some datasets (e.g. CSJ ``Streets``) are published as one layer inside a
        shared, generically-named service (``OPN_OpenDataService/MapServer/60``)
        rather than as their own top-level service - so name-matching at the
        service level alone (:meth:`discover_services`) isn't enough. This walks
        every service matching ``service_name_contains`` under ``root``, inspects
        each one's own layer list, and returns the REST URL of the *first* layer
        whose name contains ``layer_name_contains`` (case-insensitive) - use
        :meth:`find_layers` instead when more than one match is possible and it
        matters which one you get.

        Raises :class:`ArcGISCatalogError` if no matching layer is found in any
        matching service.
        """
        matches = self.find_layers(
            layer_name_contains, root=root, service_name_contains=service_name_contains, service_types=service_types
        )
        if matches:
            return matches[0][0]
        raise ArcGISCatalogError(
            f"no layer matching {layer_name_contains!r} found matching {service_name_contains!r} "
            f"under root {root!r}"
        )
