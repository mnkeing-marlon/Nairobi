"""
Data pipeline — raw CSV ingestion, per-location processing, feature engineering.

Reads all CSVs from data/raw_data/, filters to 2024+, detects locations with
sufficient data, pivots sensor rows into columns, aggregates to hourly,
adds temporal + lag features, and saves one processed CSV per location.
"""
import calendar
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOCATIONS_MANIFEST = PROCESSED_DIR / "_locations.json"

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
NUMERIC_COLS = ["P0", "P1", "P2", "humidity", "temperature"]
LAG_VARIABLES = ["P0", "P1", "P2"]
LAGS = [1, 2, 3, 4, 5, 24]
MIN_HOURS = 1000  # minimum hourly records to keep a location
DATE_CUTOFF = pd.Timestamp("2024-01-01", tz="UTC")


# ─────────────────────────────────────────────
# STEP 1 — Load all raw CSVs into one DataFrame
# ─────────────────────────────────────────────

_MONTH_NAMES = {m.lower() for m in calendar.month_name if m}


def _file_might_contain_cutoff_year(filename: str) -> bool:
    """Return False if the filename clearly predates DATE_CUTOFF."""
    parts = filename.lower().replace(".csv", "").split("_")
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return int(part) >= DATE_CUTOFF.year
    return True  # can't parse year — include to be safe


def load_all_raw(raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Read every CSV in *raw_dir* (semicolon-separated sensor.community
    format), concatenate, parse timestamps, and filter to 2024+.

    Files whose filename clearly predates the cutoff year are skipped
    to avoid loading tens of millions of unused rows into memory.
    """
    raw_dir = raw_dir or RAW_DATA_DIR
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    skipped = 0
    for fp in csv_files:
        if fp.name.startswith("_"):
            continue  # skip metadata files
        if not _file_might_contain_cutoff_year(fp.name):
            skipped += 1
            continue
        try:
            chunk = pd.read_csv(
                fp, sep=";", on_bad_lines="warn",
                dtype={"value": str, "value_type": str, "location": str},
            )
            frames.append(chunk)
        except Exception as exc:
            log.warning("Skipping %s: %s", fp.name, exc)

    if not frames:
        raise RuntimeError("All CSV files failed to load")

    log.info("Skipped %d pre-%d files by filename.", skipped, DATE_CUTOFF.year)
    df = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d rows from %d raw files.", len(df), len(frames))

    # Drop known junk columns (present in some exports)
    for col in ("Colonne1", "Unnamed: 0"):
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Parse timestamps & filter
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df[df["timestamp"] >= DATE_CUTOFF].copy()
    log.info("After %d+ filter: %d rows.", DATE_CUTOFF.year, len(df))

    # Coerce location to numeric now (was read as str to avoid dtype bloat)
    df["location"] = pd.to_numeric(df["location"], errors="coerce")

    # Drop columns we don't need downstream
    for col in ("sensor_id", "sensor_type"):
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df


# ─────────────────────────────────────────────
# STEP 2 — Detect locations with enough data
# ─────────────────────────────────────────────

def detect_top_locations(df: pd.DataFrame, min_hours: int = MIN_HOURS) -> list[int]:
    """
    Return location IDs that have at least *min_hours* unique hourly
    timestamps.
    """
    tmp = df.copy()
    tmp["hour_bucket"] = tmp["timestamp"].dt.floor("h")
    counts = (
        tmp.groupby("location")["hour_bucket"]
        .nunique()
        .sort_values(ascending=False)
    )
    eligible = counts[counts >= min_hours].index.tolist()
    eligible = [int(loc) for loc in eligible]
    log.info(
        "Locations with >= %d hours: %s (of %d total)",
        min_hours, eligible, len(counts),
    )
    return eligible


# ─────────────────────────────────────────────
# STEP 3 — Pivot + hourly aggregation
# ─────────────────────────────────────────────

def pivot_and_aggregate(df: pd.DataFrame, location: int) -> pd.DataFrame:
    """
    For a single location: pivot value_type rows into columns,
    convert to numeric, and aggregate to hourly means.
    """
    df_loc = df[df["location"] == location].copy()

    df_pivot = df_loc.pivot_table(
        index="timestamp",
        columns="value_type",
        values="value",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns.name = None

    # Numeric conversion (values come as strings after pivot)
    present_num = [c for c in NUMERIC_COLS if c in df_pivot.columns]
    df_pivot[present_num] = df_pivot[present_num].apply(pd.to_numeric, errors="coerce")

    # Hourly floor + mean aggregation
    df_pivot["timestamp"] = df_pivot["timestamp"].dt.floor("h")
    df_hourly = (
        df_pivot.groupby("timestamp")[present_num]
        .mean()
        .reset_index()
    )
    df_hourly = df_hourly.sort_values("timestamp").reset_index(drop=True)
    return df_hourly


# ─────────────────────────────────────────────
# STEP 4 — Feature engineering
# ─────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features (heure, jour_semaine, periode) and lag
    features for P0/P1/P2.
    """
    out = df.copy()

    # Temporal
    ts = out["timestamp"]
    out["heure"] = ts.dt.hour
    out["jour_semaine"] = ts.dt.day_name()
    out["periode"] = out["jour_semaine"].apply(
        lambda d: "weekend" if d in ("Saturday", "Sunday") else "semaine"
    )

    # Temperature sanity (>60 °C is sensor error)
    if "temperature" in out.columns:
        out.loc[out["temperature"] > 60, "temperature"] = np.nan
        out["temperature"] = out["temperature"].interpolate(method="linear")

    # Lag features
    for var in LAG_VARIABLES:
        if var not in out.columns:
            continue
        for lag in LAGS:
            out[f"{var}_lag_{lag}"] = out[var].shift(lag)

    return out


# ─────────────────────────────────────────────
# STEP 5 — Run full pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    locations: list[int] | None = None,
    min_hours: int = MIN_HOURS,
    raw_dir: Path | None = None,
) -> dict:
    """
    Execute the full pipeline:
      1. Load raw data
      2. Detect eligible locations (or use *locations*)
      3. For each location: pivot, aggregate, add features, save CSV
      4. Write _locations.json manifest

    Returns dict mapping location_id -> output CSV path.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_all_raw(raw_dir)
    if df_raw.empty:
        log.warning("No data after loading raw files.")
        return {}

    if locations is None:
        locations = detect_top_locations(df_raw, min_hours=min_hours)

    if not locations:
        log.warning("No locations meet the minimum-hours threshold.")
        return {}

    output_map = {}
    manifest = {}

    for loc_id in locations:
        log.info("Processing location %d ...", loc_id)
        df_hourly = pivot_and_aggregate(df_raw, loc_id)

        if len(df_hourly) < 24:
            log.warning("Location %d has only %d rows after aggregation — skipping.", loc_id, len(df_hourly))
            continue

        df_feat = add_features(df_hourly)
        out_path = PROCESSED_DIR / f"location_{loc_id}.csv"
        df_feat.to_csv(out_path, index=False)
        output_map[loc_id] = str(out_path)
        manifest[str(loc_id)] = {
            "file": f"location_{loc_id}.csv",
            "rows": len(df_feat),
            "start": str(df_feat["timestamp"].min()),
            "end": str(df_feat["timestamp"].max()),
        }
        log.info("  -> Saved %s (%d rows)", out_path.name, len(df_feat))

    # Write manifest
    LOCATIONS_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Manifest written: %s", LOCATIONS_MANIFEST)

    return output_map


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    result = run_pipeline()
    print(f"\nDone. Processed {len(result)} location(s).")
    for loc_id, path in result.items():
        print(f"  {loc_id} -> {path}")
