# `data/` — Input data (not tracked in git)

This folder is **gitignored** because the HAGRID parcel-demand shapefiles
and the Region Hannover postal-code geodata are large (~400 MB) and have
their own licensing terms that prevent redistribution from this repository.

## Layout (expected once populated)

```
data/
├── README.md            ← this file (tracked)
├── MANIFEST.md          ← input-data inventory (tracked)
├── demand/              HAGRID parcel-demand shapefiles
│   └── *.shp + .shx + .dbf + .prj (per-weekday subsets)
├── geodata/             Region Hannover spatial data
│   ├── region_hannover.geojson
│   ├── plz_areas.shp + sidecars
│   ├── cluster_raumtyp.csv
│   └── plz_raumtyp.csv
├── hubs/                LSP hub/depot CSVs
│   └── kep_hubs.csv
└── vehicles/            HAGRID vehicle type definitions
    └── hagrid_vehicle_types.csv
```

## Obtaining the data

The HAGRID dataset is maintained by the *Niedersächsisches Forschungszentrum
für Mobilität* (NFM) at TU Braunschweig. For the EWGT 2026 paper
reproducibility package, please contact the corresponding author
(lasse.bienzeisler@tu-braunschweig.de) to request the data archive used in
this study.

A future commit will replace this manual step with a
`scripts/data/download_inputs.py` script that fetches the inputs from a
Zenodo DOI, but at the time of paper submission the data are not yet
mirrored to a public archive.

## Licensing

The Region Hannover PLZ geometries are derived from public-domain
OpenStreetMap data (© OpenStreetMap contributors, ODbL).

The HAGRID parcel-demand model output is shared for academic reproduction
of the EWGT 2026 results only. Redistribution requires separate permission
from NFM. The MIT license that covers the rest of this repository does
NOT extend to the data folder.
