"""
Module de modelisation -- Prophet pour prediction PM2.5 et PM10
Entrainement sur location 3966, prediction recursive N heures,
transfer vers toutes les locations.
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

log = logging.getLogger(__name__)

# ---------------------------------------------
# CONSTANTES
# ---------------------------------------------

TARGET_CONFIG = {
    'P2': {
        'lag_features': ['P2_lag_1', 'P2_lag_2', 'P2_lag_3',
                         'P2_lag_4', 'P2_lag_5', 'P2_lag_24'],
        'model_file':   'prophet_pm25.joblib',
        'metrics_file': 'metrics_pm25.joblib',
        'label':        'PM2.5',
    },
    'P1': {
        'lag_features': ['P1_lag_1', 'P1_lag_2', 'P1_lag_3',
                         'P1_lag_4', 'P1_lag_5', 'P1_lag_24'],
        'model_file':   'prophet_pm10.joblib',
        'metrics_file': 'metrics_pm10.joblib',
        'label':        'PM10',
    },
}

# Backward-compat aliases
LAG_FEATURES = TARGET_CONFIG['P2']['lag_features']
TARGET = 'P2'

MODEL_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'prophet_pm25.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics_pm25.joblib')

TRAIN_LOCATION = 3966  # Kibera -- reference for training
TEST_HOURS = 168       # last week held out for evaluation


def _paths_for(target: str):
    cfg = TARGET_CONFIG[target]
    return (
        os.path.join(MODEL_DIR, cfg['model_file']),
        os.path.join(MODEL_DIR, cfg['metrics_file']),
    )

def _prepare_prophet_df(df: pd.DataFrame, target: str = 'P2') -> pd.DataFrame:
    """
    Build the Prophet-ready DataFrame:
      ds = timestamp (tz-naive), y = target shifted -1 (next-hour),
      plus lag regressors.
    """
    cfg = TARGET_CONFIG[target]
    lag_feats = cfg['lag_features']
    out = df.copy()
    out = out.dropna(subset=lag_feats + [target])

    # Prophet requires tz-naive ds
    out['ds'] = pd.to_datetime(out['timestamp']).dt.tz_localize(None)
    out['y'] = out[target].shift(-1)  # predict next hour
    out = out.dropna(subset=['y'])
    return out


def train_model(df: pd.DataFrame, force: bool = False, target: str = 'P2'):
    """
    Train a Prophet model for *target* ('P2' for PM2.5, 'P1' for PM10).
    Saves model + metrics to models/.
    Returns (model, metrics_dict).
    """
    model_path, metrics_path = _paths_for(target)
    cfg = TARGET_CONFIG[target]
    lag_feats = cfg['lag_features']

    if not force and os.path.exists(model_path) and os.path.exists(metrics_path):
        return load_or_train_model(df, target=target)

    log.info("Training Prophet model for %s ...", cfg['label'])
    pdf = _prepare_prophet_df(df, target=target)

    # Chronological train / test split
    split_idx = len(pdf) - TEST_HOURS
    train = pdf.iloc[:split_idx]
    test  = pdf.iloc[split_idx:]

    model = Prophet(changepoint_prior_scale=0.5)
    for feat in lag_feats:
        model.add_regressor(feat)

    model.fit(train[['ds', 'y'] + lag_feats])

    # Evaluate on test set
    future_test = test[['ds'] + lag_feats].copy()
    forecast = model.predict(future_test)

    y_true = test['y'].values
    y_pred = forecast['yhat'].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    residual_std = float(np.std(y_true - y_pred))

    log.info("%s Prophet MAE = %.2f, residual_std = %.2f",
             cfg['label'], mae, residual_std)

    # Serialize
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)

    metrics = {
        'mae':            mae,
        'residual_std':   residual_std,
        'best_params':    {'changepoint_prior_scale': 0.5},
        'importance':     {f: 0.0 for f in lag_feats},
        'test_actual':    y_true.tolist(),
        'test_predicted': y_pred.tolist(),
    }
    joblib.dump(metrics, metrics_path)
    log.info("%s model saved to %s", cfg['label'], model_path)

    return model, metrics


def load_or_train_model(df: pd.DataFrame, target: str = 'P2'):
    """Load saved Prophet model for *target*, or train from scratch."""
    model_path, metrics_path = _paths_for(target)
    if os.path.exists(model_path) and os.path.exists(metrics_path):
        model   = joblib.load(model_path)
        metrics = joblib.load(metrics_path)
        return model, metrics
    return train_model(df, force=True, target=target)


# ─────────────────────────────────────────────
# PREDICTION RECURSIVE N HEURES
# ─────────────────────────────────────────────

def predict_next_n_hours(model, df, n_hours, mae, residual_std=None, target: str = 'P2'):
    """
    Recursively predict the next *n_hours* using lag features.

    At each step the latest predicted value is fed back as the
    lag_1 regressor (and so on), matching the Prophet training
    setup where y = target.shift(-1).
    """
    cfg = TARGET_CONFIG[target]
    lag_feats = cfg['lag_features']

    if residual_std is None:
        residual_std = mae * 1.2

    history = df.tail(max(50, n_hours)).copy().reset_index(drop=True)
    vals = list(history[target].values)
    timestamps = list(pd.to_datetime(history['timestamp']).values)

    predictions = []

    for i in range(n_hours):
        next_ts = pd.Timestamp(timestamps[-1]) + pd.Timedelta(hours=1)
        next_ts_naive = next_ts.tz_localize(None) if next_ts.tzinfo else next_ts

        feats = {'ds': next_ts_naive}
        for feat in lag_feats:
            # e.g. P2_lag_1 -> lag=1, P1_lag_24 -> lag=24
            lag = int(feat.rsplit('_', 1)[-1])
            feats[feat] = vals[-lag] if len(vals) >= lag else vals[-1]

        forecast = model.predict(pd.DataFrame([feats]))
        pred = max(0, float(forecast['yhat'].iloc[0]))

        # Confidence interval widens with horizon
        ci = 1.96 * residual_std * (1 + 0.04 * min(i, 100))
        lower = max(0, pred - ci)
        upper = pred + ci

        predictions.append({
            'timestamp': next_ts,
            'predicted': pred,
            'lower':     lower,
            'upper':     upper,
        })

        vals.append(pred)
        timestamps.append(next_ts)

    return pd.DataFrame(predictions)


def predict_next_24h(model, df, mae, residual_std=None, target: str = 'P2'):
    """Predict the next 24 hours (convenience wrapper)."""
    return predict_next_n_hours(model, df, 24, mae, residual_std, target=target)


def predict_next_week(model, df, mae, residual_std=None, target: str = 'P2'):
    """Predict the next 168 hours (convenience wrapper)."""
    return predict_next_n_hours(model, df, 168, mae, residual_std, target=target)


# ─────────────────────────────────────────────
# TENDANCE
# ─────────────────────────────────────────────

def get_trend(predictions, current_mean):
    """Return (icon, label, colour) for the predicted trend."""
    pred_mean = predictions['predicted'].mean()
    if current_mean == 0:
        diff_pct = 0
    else:
        diff_pct = ((pred_mean - current_mean) / current_mean) * 100

    if diff_pct > 5:
        return "↗", "hausse", "#e53e3e"
    if diff_pct < -5:
        return "↘", "baisse", "#38a169"
    return "→", "stable", "#d69e2e"
