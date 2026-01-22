# VROOM-based Last-Mile Delivery Routing with Valhalla

This repository provides a reproducible routing backend for last-mile delivery experiments using Valhalla for road network routing and VROOM for vehicle routing optimization.

## Architecture
- Valhalla: routing engine (Dockerized)
- VROOM: optimization layer
- Python scripts: experiment orchestration

## Data
OSM data and routing tiles are not part of this repository and must be provided locally.

## Reproducibility
All infrastructure dependencies are containerized.
