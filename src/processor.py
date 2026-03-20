"""
Module de traitement des donnees de qualite de l'air — Nairobi
Chargement, preparation, calcul AQI, seuils OMS, KPIs, heatmap
"""
import json
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

SENSOR_LOCATIONS = {
    76:   {"name": "Westlands",  "lat": -1.261, "lon": 36.782},
    3573: {"name": "South B",    "lat": -1.289, "lon": 36.825},
    3966: {"name": "Kibera",     "lat": -1.311, "lon": 36.817},
    3967: {"name": "Kibera II",  "lat": -1.309, "lon": 36.812},
    3981: {"name": "CBD",        "lat": -1.269, "lon": 36.819},
    4013: {"name": "Upper Hill", "lat": -1.284, "lon": 36.823},
}

SUBSTANCE_MAP = {
    "PM2.5":       "P2",
    "PM10":        "P1",
}

SUBSTANCE_UNITS = {
    "PM2.5":       "µg/m³",
    "PM10":        "µg/m³",
}

DAY_EN_TO_FR = {
    "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mer",
    "Thursday": "Jeu", "Friday": "Ven", "Saturday": "Sam", "Sunday": "Dim",
}

DAY_ORDER_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

WHO_THRESHOLDS = {
    "PM2.5": 15,   # µg/m³ moyenne 24 h
    "PM10":  45,   # µg/m³ moyenne 24 h
}

# Breakpoints EPA — PM2.5
AQI_BP_PM25 = [
    (0.0,   12.0,  0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

# Breakpoints EPA — PM10
AQI_BP_PM10 = [
    (0,   54,  0,   50),
    (55,  154, 51,  100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNEES
# ─────────────────────────────────────────────

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "processed")
LOCATIONS_MANIFEST = os.path.join(PROCESSED_DIR, "_locations.json")

# Legacy fallback for single-location CSV
_LEGACY_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "modele_3966_Aug_Dec2025.csv")


def get_available_locations() -> dict:
    """
    Read _locations.json manifest written by the pipeline.
    Returns {location_id_str: {file, rows, start, end}, ...}.
    Falls back to {3966: ...} if only the legacy CSV exists.
    """
    if os.path.exists(LOCATIONS_MANIFEST):
        with open(LOCATIONS_MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback: legacy single-file
    if os.path.exists(_LEGACY_CSV):
        return {"3966": {"file": "modele_3966_Aug_Dec2025.csv", "rows": 0,
                         "start": "", "end": ""}}
    return {}


def load_and_prepare_data(location: int = 3966):
    """
    Load processed data for a given location.
    Looks first in data/processed/location_{id}.csv (pipeline output),
    then falls back to the legacy data/modele_3966_Aug_Dec2025.csv.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Try pipeline-processed file first
    processed_path = os.path.join(project_root, "data", "processed",
                                  f"location_{location}.csv")
    if os.path.exists(processed_path):
        csv_path = processed_path
    elif location == 3966 and os.path.exists(_LEGACY_CSV):
        csv_path = _LEGACY_CSV
    else:
        return None

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Corriger les anomalies de temperature (ex. 299 C)
    if 'temperature' in df.columns:
        df.loc[df['temperature'] > 60, 'temperature'] = np.nan
        df['temperature'] = df['temperature'].interpolate(method='linear')

    return df


# ─────────────────────────────────────────────
# CALCUL AQI
# ─────────────────────────────────────────────

def calculate_aqi(concentration, substance="PM2.5"):
    """Calcule l'AQI EPA à partir d'une concentration."""
    if pd.isna(concentration) or concentration < 0:
        return 0

    breakpoints = AQI_BP_PM25 if substance == "PM2.5" else AQI_BP_PM10

    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (concentration - c_lo) + i_lo)
    return 500


def get_aqi_category(aqi):
    """Retourne (label, couleur_hex, couleur_marqueur) pour un AQI donné."""
    if aqi <= 50:
        return "BON",               "#00e400", "green"
    if aqi <= 100:
        return "MODÉRÉ",            "#f0c800", "gold"
    if aqi <= 150:
        return "MALSAIN SENSIBLES", "#ff7e00", "orange"
    if aqi <= 200:
        return "MALSAIN",           "#ff0000", "red"
    if aqi <= 300:
        return "TRÈS MALSAIN",      "#8f3f97", "purple"
    return "DANGEREUX",             "#7e0023", "maroon"


# ─────────────────────────────────────────────
# SEUILS OMS
# ─────────────────────────────────────────────

def check_who_exceedance(mean_value, substance="PM2.5"):
    """Vérifie si la valeur dépasse le seuil OMS 24 h. Retourne (bool, seuil)."""
    threshold = WHO_THRESHOLDS.get(substance, 15)
    return mean_value > threshold, threshold


# ─────────────────────────────────────────────
# PER-LOCATION AQI SNAPSHOT
# ─────────────────────────────────────────────

def get_all_locations_aqi(substance: str = "PM2.5") -> list[dict]:
    """
    Return a list of dicts with real AQI for every location that has
    processed data *and* known coordinates.

    Each dict: {loc_id, name, lat, lon, aqi, label, color}.
    AQI is computed from the mean of the last 24 hours of data.
    """
    col = SUBSTANCE_MAP.get(substance, "P2")
    avail = get_available_locations()
    results = []
    for loc_str in avail:
        loc_id = int(loc_str)
        info = SENSOR_LOCATIONS.get(loc_id)
        if info is None:
            continue  # no known coordinates for this location
        df = load_and_prepare_data(loc_id)
        if df is None or df.empty or col not in df.columns:
            continue
        last_24 = get_last_n_hours(df, 24)
        mean_val = last_24[col].mean() if not last_24.empty else df[col].mean()
        aqi = calculate_aqi(mean_val, substance)
        label, color, _ = get_aqi_category(aqi)
        results.append({
            "loc_id": loc_id,
            "name": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "aqi": aqi,
            "label": label,
            "color": color,
        })
    return results


# ─────────────────────────────────────────────
# FILTRAGE TEMPOREL
# ─────────────────────────────────────────────

def get_last_n_hours(df, n=24):
    """Retourne les n dernières heures de données."""
    last_ts = df['timestamp'].max()
    start_ts = last_ts - pd.Timedelta(hours=n)
    return df[df['timestamp'] >= start_ts].copy()


# ─────────────────────────────────────────────
# HEATMAP
# ─────────────────────────────────────────────

def get_heatmap_data(df, col="P2"):
    """Génère un pivot heure × jour de la semaine pour la heatmap."""
    tmp = df[['heure', 'jour_semaine', col]].copy()
    tmp['jour_fr'] = tmp['jour_semaine'].map(DAY_EN_TO_FR)

    pivot = tmp.pivot_table(values=col, index='jour_fr', columns='heure', aggfunc='mean')

    present = [d for d in DAY_ORDER_FR if d in pivot.index]
    if present:
        pivot = pivot.loc[present]
    return pivot


# ─────────────────────────────────────────────
# KPIs  (conservés pour rétro-compatibilité)
# ─────────────────────────────────────────────

def calculate_kpis(data, particle, timeframe):
    """
    Calcule les KPIs avec agrégation hiérarchique.
    timeframe: 'D', 'W' ou 'M'
    """
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data = data.sort_values('timestamp')

    if timeframe == 'D':
        return _calculate_daily_kpis(data, particle)
    elif timeframe == 'W':
        return _calculate_weekly_kpis(data, particle)
    elif timeframe == 'M':
        return _calculate_monthly_kpis(data, particle)
    raise ValueError("Timeframe doit être 'D', 'W' ou 'M'")


def _calculate_daily_kpis(data, particle):
    last_date = data['timestamp'].dt.date.iloc[-1]
    last_day = data[data['timestamp'].dt.date == last_date]
    if len(last_day) == 0:
        return None

    c_min, c_mean, c_max = last_day[particle].min(), last_day[particle].mean(), last_day[particle].max()

    prev_date = last_date - pd.Timedelta(days=1)
    prev_day = data[data['timestamp'].dt.date == prev_date]
    p_min  = prev_day[particle].min()  if len(prev_day) > 0 else None
    p_mean = prev_day[particle].mean() if len(prev_day) > 0 else None
    p_max  = prev_day[particle].max()  if len(prev_day) > 0 else None

    return _pack_kpis(c_min, c_mean, c_max, p_min, p_mean, p_max,
                      last_date.strftime('%d/%m/%Y'))


def _calculate_weekly_kpis(data, particle):
    daily = data.resample('D', on='timestamp')[particle].mean().reset_index()
    if len(daily) < 2:
        return None

    last_w  = daily.tail(min(7, len(daily)))
    prev_w  = daily.iloc[-14:-7] if len(daily) >= 14 else (
              daily.iloc[-8:-1] if len(daily) >= 8 else pd.DataFrame())

    c_min, c_mean, c_max = last_w[particle].min(), last_w[particle].mean(), last_w[particle].max()
    p_min  = prev_w[particle].min()  if len(prev_w) > 0 else None
    p_mean = prev_w[particle].mean() if len(prev_w) > 0 else None
    p_max  = prev_w[particle].max()  if len(prev_w) > 0 else None

    s = last_w['timestamp'].min().strftime('%d/%m')
    e = last_w['timestamp'].max().strftime('%d/%m')
    return _pack_kpis(c_min, c_mean, c_max, p_min, p_mean, p_max, f"{s} - {e}")


def _calculate_monthly_kpis(data, particle):
    weekly = data.resample('W-MON', on='timestamp')[particle].mean().reset_index()
    if len(weekly) < 2:
        return None

    last_m = weekly.tail(min(4, len(weekly)))
    prev_m = weekly.iloc[-8:-4] if len(weekly) >= 8 else (
             weekly.iloc[-5:-1] if len(weekly) >= 5 else pd.DataFrame())

    c_min, c_mean, c_max = last_m[particle].min(), last_m[particle].mean(), last_m[particle].max()
    p_min  = prev_m[particle].min()  if len(prev_m) > 0 else None
    p_mean = prev_m[particle].mean() if len(prev_m) > 0 else None
    p_max  = prev_m[particle].max()  if len(prev_m) > 0 else None

    s = last_m['timestamp'].min().strftime('%d/%m')
    e = last_m['timestamp'].max().strftime('%d/%m')
    return _pack_kpis(c_min, c_mean, c_max, p_min, p_mean, p_max, f"{s} - {e}")


def _pack_kpis(c_min, c_mean, c_max, p_min, p_mean, p_max, label):
    min_d, min_dp   = _variation(c_min,  p_min)
    mean_d, mean_dp = _variation(c_mean, p_mean)
    max_d, max_dp   = _variation(c_max,  p_max)
    return {
        'min': c_min, 'mean': c_mean, 'max': c_max,
        'min_delta': min_d,   'min_delta_pct': min_dp,
        'mean_delta': mean_d, 'mean_delta_pct': mean_dp,
        'max_delta': max_d,   'max_delta_pct': max_dp,
        'period_label': label,
    }


def _variation(current, previous):
    if previous is None or pd.isna(previous) or previous == 0:
        return None, None
    delta = current - previous
    return delta, (delta / previous) * 100
