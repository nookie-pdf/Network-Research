"""Data transforms for the historical three-station flood-research dashboard.

This module deliberately has no Streamlit dependencies so it can be tested offline.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

TZ = "Asia/Bangkok"
STATIONS = {
    "RF.BKA.01": {"name": "บางเขน / Bang Khen", "lat": 13.87349, "lon": 100.59643},
    "RF.SMI.01": {"name": "สายไหม / Sai Mai", "lat": 13.89531, "lon": 100.66090},
    "RF.LSI.01": {"name": "หลักสี่ / Lak Si", "lat": 13.88748, "lon": 100.57893},
}
FLOOD_SENSORS = {
    "FL.BKA.02": "พหลโยธิน 55/1 (บางเขน)",
    "FL.SMI.01": "พหลโยธิน 60/1 (สายไหม)",
    "FL.LSI.06": "งามวงศ์วาน ซอยชินเขต (หลักสี่)",
}


def local_index(values) -> pd.DatetimeIndex:
    """Return Asia/Bangkok-aware timestamps without shifting naive local records."""
    times = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    return times.tz_localize(TZ) if times.tz is None else times.tz_convert(TZ)


def csv_times(path: Path, *, col: str = "datetime", fill_hourly: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if col not in df:
        raise ValueError(f"Missing datetime column {col}: {path}")
    idx = local_index(df.pop(col))
    if idx.has_duplicates:
        raise ValueError(f"Duplicate timestamps: {path}")
    df.index = idx
    df = df.sort_index()
    if fill_hourly and not df.empty:
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    return df


def prediction_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    col = "datetime" if "datetime" in raw.columns else "Unnamed: 0"
    if col not in raw:
        raise ValueError(f"No time column: {path}")
    idx = local_index(raw.pop(col))
    raw.index = idx
    if not {"probability", "warning"}.issubset(raw.columns):
        raise ValueError(f"Prediction columns missing: {path}")
    raw["probability"] = pd.to_numeric(raw["probability"], errors="raise")
    if not raw["probability"].between(0, 1).all():
        raise ValueError("Prediction score must be between zero and one")
    raw["warning"] = raw["warning"].astype(str).str.lower().eq("true")
    return raw.sort_index()


def load_bundle(root: Path) -> dict:
    path = Path(root) / "data"
    d = {
        "rain": csv_times(path / "station_rainfall.csv", fill_hourly=True),
        "flood": csv_times(path / "station_flood.csv", fill_hourly=True),
        "levels": csv_times(path / "road_flood_levels_5min.csv"),
        "test": prediction_csv(path / "test_predictions.csv"),
        "validation": prediction_csv(path / "validation_predictions.csv"),
        "replay": prediction_csv(path / "replay_predictions.csv"),
        "events": pd.read_csv(path / "observed_flood_events_2025.csv"),
        "report": json.loads((path / "pilot_report.json").read_text(encoding="utf-8")),
        "audit": json.loads((path / "source_audit.json").read_text(encoding="utf-8")),
    }
    for station in STATIONS:
        if station not in d["rain"]:
            raise ValueError(f"Rain station not found: {station}")
    return d


def rain_before(rain: pd.DataFrame, end_time: pd.Timestamp, hours: int = 24) -> pd.Series:
    """Fixed historical window; never silently turn unobserved hours into zero."""
    t = pd.Timestamp(end_time)
    window = rain.reindex(pd.date_range(t - pd.Timedelta(hours=hours-1), t, freq="h"))
    sums = window.sum(min_count=hours)
    return sums


def flood_confusion(pred: pd.DataFrame, threshold: float) -> dict[str, int]:
    if "target" not in pred.columns:
        raise ValueError("Ground-truth target is required")
    known = pred.dropna(subset=["target", "probability"])
    y = known["target"].astype(int).to_numpy()
    p = (known["probability"].to_numpy() >= threshold).astype(int)
    return {
        "tp": int(((y == 1) & (p == 1)).sum()),
        "fp": int(((y == 0) & (p == 1)).sum()),
        "tn": int(((y == 0) & (p == 0)).sum()),
        "fn": int(((y == 1) & (p == 0)).sum()),
    }


def classification_metrics(pred: pd.DataFrame, threshold: float) -> dict:
    c = flood_confusion(pred, threshold)
    precision = c["tp"] / max(1, c["tp"] + c["fp"])
    recall = c["tp"] / max(1, c["tp"] + c["fn"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    from sklearn.metrics import average_precision_score, brier_score_loss
    y = pred["target"].astype(int)
    prob = pred["probability"].astype(float)
    return {**c, "precision": precision, "recall": recall, "f1": f1,
            "average_precision": float(average_precision_score(y, prob)),
            "brier": float(brier_score_loss(y, prob)), "n": len(y),
            "positive": int(y.sum())}


def make_features(rain: pd.DataFrame) -> pd.DataFrame:
    """The 15 rainfall lags used by the saved Random Forest model."""
    if (rain[list(STATIONS)] < 0).any().any() or np.isinf(rain[list(STATIONS)].to_numpy()).any():
        raise ValueError("Rainfall must be nonnegative, finite or missing")
    features = pd.DataFrame(index=rain.index)
    for station in STATIONS:
        for h in (1, 3, 6, 12, 24):
            features[f"{station}_sum_{h}h"] = rain[station].rolling(h, min_periods=h).sum()
    return features


def scenario_features(rain: pd.DataFrame, timestamp: pd.Timestamp,
                      station: str, added_mm_each_hour: float, preceding_hours: int) -> pd.DataFrame:
    """Counterfactual perturbation of OBSERVED PAST rain only (not a real forecast)."""
    if station not in STATIONS or preceding_hours not in (1, 3, 6):
        raise ValueError("Invalid counterfactual settings")
    if not (0 <= added_mm_each_hour <= 100):
        raise ValueError("Expected 0-100 extra millimetres per hour")
    if timestamp not in rain.index:
        raise ValueError("Selected replay hour not found")
    start = timestamp - pd.Timedelta(hours=preceding_hours-1)
    window = rain.loc[timestamp - pd.Timedelta(hours=23):timestamp, list(STATIONS)].copy()
    if len(window) != 24 or window.isna().any().any():
        raise ValueError("24 complete observed rainfall hours are needed for a scenario")
    window.loc[start:timestamp, station] += added_mm_each_hour
    return make_features(window).loc[[timestamp]]
