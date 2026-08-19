# Causal Semantic Navigation for Aviation

## Phase 0: data pipeline

This phase builds the ArcGIS tile client used to collect aerial imagery
training data from San Jose's ArcGIS Server (`geo.sanjoseca.gov`), covering
the `DPW_ImageryCached` service and its historic vintages. See
[`docs/phase0_arcgis_tile_client.md`](docs/phase0_arcgis_tile_client.md) for
design details.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Tests

```bash
pytest
```

### Fetching historic imagery for an area of interest

```bash
python scripts/fetch_historic_imagery.py \
    --bbox -121.95 37.30 -121.85 37.36 \
    --output-dir data/raw/dpw_imagery
```

This discovers every `DPW_Imagery*` service under the `Imagery` folder
(current + all historic vintages), fetches the tiles covering the bounding
box for each one, reprojects them from EPSG:3857 to EPSG:4326, and writes
one GeoTIFF per tile under `data/raw/dpw_imagery/<service-name>/`.
