# Input-data Manifest

Lists every file in `data/` that the pipeline depends on, with the
expected size and the consumer in `src/batch_delivery/`. Files are
gitignored — see [`README.md`](README.md) for how to obtain the inputs.

| Path | Approx. size | Consumed by |
|---|---|---|
| `demand/hagrid_demand_*.shp` (+ sidecars) | ~250 MB | `batch_delivery.io.demand.load_hagrid_demand` |
| `demand/weekday_subsets/*.shp` | ~80 MB | `batch_delivery.io.demand.load_weekday_demand` |
| `geodata/region_hannover.geojson` | ~5 MB | `batch_delivery.io.demand.load_region_polygon` |
| `geodata/plz_areas.shp` (+ sidecars) | ~12 MB | `batch_delivery.io.demand.load_plz_areas` |
| `geodata/cluster_raumtyp.csv` | ~15 KB | `batch_delivery.io.demand.load_raumtyp_assignment` |
| `geodata/plz_raumtyp.csv` | ~20 KB | same |
| `hubs/kep_hubs.csv` | ~3 KB | `batch_delivery.io.hubs.load_kep_hubs` |
| `vehicles/hagrid_vehicle_types.csv` | ~2 KB | `batch_delivery.config.load_vehicle_specs` |

## Provenance

* HAGRID model output: produced by our sister project
  [HAGRID](https://github.com/TUBS-IVS/HAGRID) (same author group, TU
  Braunschweig — Institute of Transportation and Urban Engineering).
  The specific snapshot bundled here is the May 2025 run used by the
  EWGT 2026 paper.
* PLZ geometry: derived from OpenStreetMap via Geofabrik
  Niedersachsen-Bremen extract, May 2025.
* `cluster_raumtyp.csv`: BBSR Raumtyp 2017 classification at the
  postal-code level, joined manually for the Region Hannover study area.
* `kep_hubs.csv`: hand-curated from LSP public information + Google
  Maps geocoding, audit-trail in
  `scripts/data/build_plz_clusters.py`.

## Versioning

The pipeline depends on the specific snapshot used for the EWGT 2026
paper (May 2025 HAGRID run, 2025-05 OSM extract). Re-running the pipeline
with a different demand snapshot will produce numerically different
results — that is expected and is why the input bundle is frozen for
reproducibility.

For the canonical reproduction, request the exact archive used in
2026-05.
