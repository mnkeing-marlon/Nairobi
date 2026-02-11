"""
Module de traitement des données de particules
Contient les fonctions de chargement, préparation et calcul des KPIs
"""
import pandas as pd
import os


def load_and_prepare_data():
    """
    Charge et prépare le jeu de données.
    Cette fonction est mise en cache pour des performances optimales.
    
    Returns:
        pd.DataFrame: DataFrame avec les données nettoyées et préparées
    """
    # Obtenir le chemin du fichier CSV (relatif à la racine du projet)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "modele_3966_Aug_Dec2025.csv")
    
    df = pd.read_csv(csv_path)    
    
    # Conversion de la colonne 'timestamp' en datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Tri par timestamp pour assurer la cohérence des séries temporelles
    df = df.sort_values('timestamp')
    
    return df


def calculate_kpis(data, particle, timeframe):
    """
    Calcule les KPIs avec agrégation hiérarchique.
    
    Args:
        data: DataFrame avec colonne 'timestamp' (datetime) et 'particle'
        particle: Nom de la colonne à analyser
        timeframe: 'D', 'W', ou 'M'
    
    Returns:
        dict avec min, mean, max et leurs variations
    """
    # S'assurer que timestamp est en datetime
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    
    # Trier par date
    data = data.sort_values('timestamp')
    
    if timeframe == 'D':
        return _calculate_daily_kpis(data, particle)
    elif timeframe == 'W':
        return _calculate_weekly_kpis(data, particle)
    elif timeframe == 'M':
        return _calculate_monthly_kpis(data, particle)
    else:
        raise ValueError("Timeframe doit être 'D', 'W' ou 'M'")


def _calculate_daily_kpis(data, particle):
    """Logique pour le mode journalier"""
    # Dernière journée disponible
    last_date = data['timestamp'].dt.date.iloc[-1]
    last_day_data = data[data['timestamp'].dt.date == last_date]
    
    if len(last_day_data) == 0:
        return None
    
    # Calcul des métriques pour le dernier jour
    current_min = last_day_data[particle].min()
    current_mean = last_day_data[particle].mean()
    current_max = last_day_data[particle].max()
    
    # Jour précédent pour variation
    prev_date = last_date - pd.Timedelta(days=1)
    prev_day_data = data[data['timestamp'].dt.date == prev_date]

    # Calcul des métriques pour le precedent
    previous_min = prev_day_data[particle].min()
    previous_mean = prev_day_data[particle].mean()
    previous_max = prev_day_data[particle].max()
    
    # Calcul des variations si données disponibles
    mean_delta, mean_delta_pct = _calculate_variation(
        current_mean, previous_mean if len(prev_day_data) > 0 else None
    )
    min_delta, min_delta_pct = _calculate_variation(
        current_min, previous_min if len(prev_day_data) > 0 else None
    )
    max_delta, max_delta_pct = _calculate_variation(
        current_max, previous_max if len(prev_day_data) > 0 else None
    )
    
    return {
        'min': current_min,
        'mean': current_mean,
        'max': current_max,
        'min_delta': min_delta,
        'min_delta_pct': min_delta_pct,
        'mean_delta': mean_delta,
        'mean_delta_pct': mean_delta_pct,
        'max_delta': max_delta,
        'max_delta_pct': max_delta_pct,
        'period_label': last_date.strftime('%d/%m/%Y')
    }


def _calculate_weekly_kpis(data, particle):
    """Logique pour le mode hebdomadaire"""
    # 1. Agrégation journalière
    data_daily = data.resample('D', on='timestamp')[particle].mean().reset_index()
    
    if len(data_daily) < 2:
        return None
    
    # 2. Dernière semaine (7 derniers jours max)
    last_week_data = data_daily.tail(min(7, len(data_daily)))
    
    # 3. Semaine précédente (les 7 jours avant)
    if len(data_daily) >= 14:
        prev_week_data = data_daily.iloc[-14:-7]
    elif len(data_daily) >= 8:
        prev_week_data = data_daily.iloc[-8:-1]
    else:
        prev_week_data = pd.DataFrame()
    
    # Calculs
    current_min = last_week_data[particle].min()
    current_mean = last_week_data[particle].mean()
    current_max = last_week_data[particle].max()
    # Calcul des métriques pour le precedent
    previous_min = prev_week_data[particle].min()
    previous_mean = prev_week_data[particle].mean()
    previous_max = prev_week_data[particle].max()

    # Calcul des variations si données disponibles
    mean_delta, mean_delta_pct = _calculate_variation(
        current_mean, previous_mean if len(prev_week_data) > 0 else None
    )
    min_delta, min_delta_pct = _calculate_variation(
        current_min, previous_min if len(prev_week_data) > 0 else None
    )
    max_delta, max_delta_pct = _calculate_variation(
        current_max, previous_max if len(prev_week_data) > 0 else None
    )
    
    # Label avec dates de début et fin
    start_date = last_week_data['timestamp'].min().strftime('%d/%m')
    end_date = last_week_data['timestamp'].max().strftime('%d/%m')
    
    return {
        'min': current_min,
        'mean': current_mean,
        'max': current_max,
        'min_delta': min_delta,
        'min_delta_pct': min_delta_pct,
        'mean_delta': mean_delta,
        'mean_delta_pct': mean_delta_pct,
        'max_delta': max_delta,
        'max_delta_pct': max_delta_pct,
        'period_label': f"{start_date} - {end_date}"
    }


def _calculate_monthly_kpis(data, particle):
    """Logique pour le mode mensuel"""
    # 1. Agrégation hebdomadaire (lundi à dimanche)
    data_weekly = data.resample('W-MON', on='timestamp')[particle].mean().reset_index()
    
    if len(data_weekly) < 2:
        return None
    
    # 2. Dernier mois (~4 dernières semaines)
    last_month_data = data_weekly.tail(min(4, len(data_weekly)))
    
    # 3. Mois précédent (les 4 semaines avant)
    if len(data_weekly) >= 8:
        prev_month_data = data_weekly.iloc[-8:-4]
    elif len(data_weekly) >= 5:
        prev_month_data = data_weekly.iloc[-5:-1]
    else:
        prev_month_data = pd.DataFrame()
    
    # Calculs
    current_min = last_month_data[particle].min()
    current_mean = last_month_data[particle].mean()
    current_max = last_month_data[particle].max()

    # Calcul des métriques pour le precedent
    previous_min = prev_month_data[particle].min()
    previous_mean = prev_month_data[particle].mean()
    previous_max = prev_month_data[particle].max()

    # Calcul des variations si données disponibles
    mean_delta, mean_delta_pct = _calculate_variation(
        current_mean, previous_mean if len(prev_month_data) > 0 else None
    )
    min_delta, min_delta_pct = _calculate_variation(
        current_min, previous_min if len(prev_month_data) > 0 else None
    )
    max_delta, max_delta_pct = _calculate_variation(
        current_max, previous_max if len(prev_month_data) > 0 else None
    ) 
    
    # Label
    start_date = last_month_data['timestamp'].min().strftime('%d/%m')
    end_date = last_month_data['timestamp'].max().strftime('%d/%m')
    
    return {
        'min': current_min,
        'mean': current_mean,
        'max': current_max,
        'min_delta': min_delta,
        'min_delta_pct': min_delta_pct,
        'mean_delta': mean_delta,
        'mean_delta_pct': mean_delta_pct,
        'max_delta': max_delta,
        'max_delta_pct': max_delta_pct,
        'period_label': f"{start_date} - {end_date}"
    }


def _calculate_variation(current_val, previous_val):
    """Calcule la variation absolue et en pourcentage"""
    if previous_val is None or pd.isna(previous_val) or previous_val == 0:
        return None, None
    
    delta = current_val - previous_val
    delta_pct = (delta / previous_val) * 100
    
    return delta, delta_pct
