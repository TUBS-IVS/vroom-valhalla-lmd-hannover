import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests


def read_json_with_bom(path: Path) -> Dict[str, Any]:
    # Use utf-8-sig to safely handle Windows UTF-8 BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract_ordered_locations_from_vroom(vroom_solution: Dict[str, Any]) -> List[Dict[str, Any]]:
    routes = vroom_solution.get("routes", [])
    if not routes:
        raise ValueError("No routes found in VROOM solution.")

    steps = routes[0].get("steps", [])
    if not steps:
        raise ValueError("No steps found in the first VROOM route.")

    ordered = []
    for idx, step in enumerate(steps):
        loc = step.get("location")
        if not loc or len(loc) != 2:
            continue

        lon = float(loc[0])
        lat = float(loc[1])

        step_type = str(step.get("type", "unknown"))
        step_id = step.get("id")
        job_id = step.get("job")

        label_parts = [f"{idx}:{step_type}"]
        if job_id is not None:
            label_parts.append(f"job={job_id}")
        if step_id is not None:
            label_parts.append(f"id={step_id}")

        ordered.append(
            {
                "idx": idx,
                "lat": lat,
                "lon": lon,
                "type": step_type,
                "label": " ".join(label_parts),
            }
        )

    if len(ordered) < 2:
        raise ValueError("Need at least two locations to render a route.")

    return ordered


def decode_polyline6(polyline: str) -> List[Tuple[float, float]]:
    # Decodes Valhalla polyline6 into list of (lon, lat).
    index = 0
    lat = 0
    lon = 0
    coords: List[Tuple[float, float]] = []

    while index < len(polyline):
        shift = 0
        result = 0
        while True:
            b = ord(polyline[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(polyline[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon

        coords.append((lon / 1e6, lat / 1e6))

    return coords


def try_fetch_valhalla_geometry(
    ordered_points: List[Dict[str, Any]],
    valhalla_url: str,
    costing: str,
    timeout_s: int = 120,
) -> Optional[List[Tuple[float, float]]]:
    # Requests route geometry from Valhalla. Returns list of (lon, lat) or None if unavailable.
    locations = [{"lat": p["lat"], "lon": p["lon"], "type": "break"} for p in ordered_points]

    payload = {
        "locations": locations,
        "costing": costing,
        # Try GeoJSON first; if Valhalla returns polyline string, we decode below.
        "shape_format": "geojson",
    }

    try:
        resp = requests.post(valhalla_url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    trip = data.get("trip", {})
    legs = trip.get("legs", [])
    if not legs:
        return None

    coords: List[Tuple[float, float]] = []

    # Case A: GeoJSON dict per leg
    for leg in legs:
        shape = leg.get("shape")
        if isinstance(shape, dict) and "coordinates" in shape:
            for lon, lat in shape["coordinates"]:
                coords.append((float(lon), float(lat)))

    if coords:
        return coords

    # Case B: Polyline string per leg
    for leg in legs:
        shape = leg.get("shape")
        if isinstance(shape, str) and shape:
            coords.extend(decode_polyline6(shape))

    return coords if coords else None


def compute_bounds(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return min(lats), min(lons), max(lats), max(lons)


def build_html(
    ordered_points: List[Dict[str, Any]],
    line_coords: List[Tuple[float, float]],
    title: str,
) -> str:
    # Leaflet wants [lat, lon]
    markers_js = []
    for p in ordered_points:
        markers_js.append(
            (
                "{lat: %.8f, lon: %.8f, label: %s}"
                % (p["lat"], p["lon"], json.dumps(p["label"]))
            )
        )

    poly_js = []
    for lon, lat in line_coords:
        poly_js.append("[%.8f, %.8f]" % (lat, lon))

    all_points_for_bounds = [(p["lon"], p["lat"]) for p in ordered_points]
    all_points_for_bounds.extend(line_coords)
    min_lat, min_lon, max_lat, max_lon = compute_bounds(all_points_for_bounds)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>

  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    html, body {{ height: 100%; margin: 0; }}
    #map {{ height: 100%; width: 100%; }}

    .panel {{
      position: absolute;
      top: 16px;
      left: 16px;
      z-index: 1000;
      background: rgba(255, 255, 255, 0.92);
      padding: 12px 14px;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.18);
      font-family: Arial, sans-serif;
      max-width: 360px;
    }}

    .title {{
      font-size: 16px;
      font-weight: 700;
      margin: 0 0 8px 0;
    }}

    .meta {{
      font-size: 12px;
      color: #333;
      line-height: 1.35;
    }}

    .chip {{
      display: inline-block;
      font-size: 12px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #f3f4f6;
      margin-right: 6px;
      margin-top: 6px;
    }}

    .stop-label {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(0,0,0,0.25);
      border-radius: 10px;
      padding: 2px 6px;
      font-size: 12px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}
  </style>
</head>

<body>
  <div id="map"></div>

  <div class="panel">
    <div class="title">{title}</div>
    <div class="meta">
      <div class="chip">Stops: {len(ordered_points)}</div>
      <div class="chip">Line points: {len(line_coords)}</div>
      <div class="chip">Source: VROOM solution</div>
    </div>
  </div>

  <script>
    const map = L.map('map', {{ zoomControl: true }});

    const tiles = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors'
    }});
    tiles.addTo(map);

    const bounds = L.latLngBounds(
      L.latLng({min_lat:.8f}, {min_lon:.8f}),
      L.latLng({max_lat:.8f}, {max_lon:.8f})
    );
    map.fitBounds(bounds.pad(0.15));

    const markers = [{", ".join(markers_js)}];
    const polylineCoords = [{", ".join(poly_js)}];

    const routeLine = L.polyline(polylineCoords, {{
      weight: 5,
      opacity: 0.9
    }}).addTo(map);

    function makeCircleMarker(lat, lon, idx) {{
      return L.circleMarker([lat, lon], {{
        radius: idx === 0 || idx === markers.length - 1 ? 7 : 6,
        weight: 2,
        opacity: 1.0,
        fillOpacity: 0.9
      }});
    }}

    markers.forEach((m, i) => {{
      const marker = makeCircleMarker(m.lat, m.lon, i).addTo(map);
      marker.bindTooltip(m.label, {{
        permanent: true,
        direction: 'right',
        offset: [10, 0],
        className: 'stop-label'
      }});
      marker.bindPopup('<b>Stop</b><br>' + m.label + '<br><br>' + 'lat=' + m.lat + '<br>lon=' + m.lon);
    }});
  </script>
</body>
</html>
"""
    return html


def main() -> None:
    # Input and output paths
    vroom_solution_path = Path("results/vroom_solution.json")
    output_html_path = Path("results/route_map.html")
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    vroom_solution = read_json_with_bom(vroom_solution_path)
    ordered_points = extract_ordered_locations_from_vroom(vroom_solution)

    # Try Valhalla geometry first. If it fails, fall back to straight line between stops.
    valhalla_route_url = "http://localhost:8002/route"
    costing = "auto"

    line_coords = try_fetch_valhalla_geometry(
        ordered_points=ordered_points,
        valhalla_url=valhalla_route_url,
        costing=costing,
        timeout_s=120,
    )

    if line_coords is None:
        line_coords = [(p["lon"], p["lat"]) for p in ordered_points]

    html = build_html(
        ordered_points=ordered_points,
        line_coords=line_coords,
        title="VROOM LMD Route Map",
    )

    output_html_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {output_html_path.resolve()}")


if __name__ == "__main__":
    main()
