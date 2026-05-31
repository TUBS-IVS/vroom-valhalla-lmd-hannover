"""Utility functions: logging setup, coordinate transforms, helpers."""

import logging
import re

import pandas as pd

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def get_logger(name: str = "lmd", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


log = get_logger()


def parse_unfound_location(error_msg: str) -> list[list[float]]:
    """Extract unfound [lon, lat] coordinates from a VROOM error message."""
    matches = re.findall(r"\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]", error_msg)
    return [[float(lon), float(lat)] for lon, lat in matches]


def compute_weighted_speed_factor(
    df_routes: pd.DataFrame,
    traffic_factors: dict[int, float] | None = None,
    vehicle_time_window: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Compute vehicle-activity-weighted traffic and speed factors.

    Counts how many vehicles are active in each hour from baseline
    route data, then weights the hourly traffic factors by vehicle count.
    Hours with more active vehicles contribute more to the average.

    Parameters
    ----------
    df_routes : DataFrame
        Parsed route results; must have ``start_time`` and ``end_time``
        columns (seconds since midnight).
    traffic_factors : dict, optional
        Hourly traffic factors {hour: factor}. Defaults to config values.
    vehicle_time_window : tuple, optional
        (start_s, end_s). Defaults to config values.

    Returns
    -------
    (weighted_traffic_factor, weighted_speed_factor) : tuple[float, float]
    """
    from batch_delivery.config.constants import TRAFFIC_FACTORS, VEHICLE_TIME_WINDOW

    if traffic_factors is None:
        traffic_factors = TRAFFIC_FACTORS
    if vehicle_time_window is None:
        vehicle_time_window = VEHICLE_TIME_WINDOW

    veh_start_h = vehicle_time_window[0] // 3600
    veh_end_h = min(vehicle_time_window[1] // 3600, 23)
    hour_labels = list(range(veh_start_h, veh_end_h))

    start_times = df_routes["start_time"].values
    end_times = df_routes["end_time"].values

    vehicle_counts = {}
    for h in hour_labels:
        h_start = h * 3600
        h_end = (h + 1) * 3600
        active = int(((start_times < h_end) & (end_times >= h_start)).sum())
        vehicle_counts[h] = active

    weighted_sum = sum(vehicle_counts[h] * traffic_factors[h] for h in hour_labels)
    total_vehicles = sum(vehicle_counts[h] for h in hour_labels)

    if total_vehicles == 0:
        from batch_delivery.config.constants import AVG_TRAFFIC_FACTOR, SPEED_FACTOR
        log.warning("No active vehicles — falling back to simple average")
        return AVG_TRAFFIC_FACTOR, SPEED_FACTOR

    wtf = round(weighted_sum / total_vehicles, 4)
    wsf = round(1.0 / wtf, 3)

    log.debug(
        f"Weighted traffic factor: {wtf:.4f}, speed factor: {wsf:.3f} "
        f"(from {len(df_routes)} routes, "
        f"{total_vehicles} vehicle-hours)"
    )
    return wtf, wsf


def fmt_time(seconds: int) -> str:
    """Format seconds since midnight as HH:MM."""
    h, m = divmod(seconds, 3600)
    m //= 60
    return f"{int(h):02d}:{int(m):02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint persistence (pickle)
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_dir():
    from batch_delivery.config.constants import RESULTS_DIR
    d = RESULTS_DIR / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(name: str, **data) -> None:
    """Pickle-serialize *data* to ``results/checkpoints/{name}.pkl``.

    If a checkpoint with the same name already exists, it is backed up to
    ``results/checkpoints/archive/{name}_{YYYYMMDD_HHMMSS}.pkl`` before
    being overwritten, so no data is ever lost.
    """
    import pickle
    from datetime import datetime

    ck_dir = _checkpoint_dir()
    p = ck_dir / f"{name}.pkl"

    # Archive existing checkpoint before overwriting
    if p.exists():
        archive_dir = ck_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"{name}_{ts}.pkl"
        archive_path.write_bytes(p.read_bytes())
        log.info("Archived old checkpoint: %s", archive_path.name)

    p.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    size_mb = p.stat().st_size / 1_048_576
    log.debug("Checkpoint saved: %s (%.1f MB)", p.name, size_mb)


def load_checkpoint(name: str):
    """Load checkpoint if it exists and FORCE_RECOMPUTE is False.

    Returns
    -------
    dict | None
        Unpickled data dict, or *None* if not found / forced recompute.
    """
    import pickle

    from batch_delivery.config.constants import FORCE_RECOMPUTE
    if FORCE_RECOMPUTE:
        return None
    p = _checkpoint_dir() / f"{name}.pkl"
    if not p.exists():
        return None
    data = pickle.loads(p.read_bytes())
    log.debug("Checkpoint loaded: %s", p.name)
    return data
