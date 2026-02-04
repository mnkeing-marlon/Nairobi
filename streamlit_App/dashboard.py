import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------
# CONFIGURATION DE LA PAGE
# -------------------------------
st.set_page_config(
    page_title="Dashboard Analyse Particules",
    layout="wide"
)

# -------------------------------
# 1. CHARGEMENT DES DONNÉES
# -------------------------------
@st.cache_data
def load_and_prepare_data():
    """
    Charge et prépare le jeu de données.
    Cette fonction est mise en cache pour des performances optimales.
    """
    # REMPLACEZ 'votre_fichier.csv' par le chemin réel de votre fichier
    df=pd.read_csv("modele_3966_Aug_Dec2025.csv")    
    # Conversion de la colonne 'timestamp' en datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Tri par timestamp pour assurer la cohérence des séries temporelles
    df = df.sort_values('timestamp')
    return df

# -------------------------------
# 2. FONCTION DE CRÉATION DU GRAPHIQUE
# -------------------------------
def create_time_series_plot(data, particle, timeframe):
    """
    Crée un graphique de série temporelle avec remplissage.
    
    Parameters:
    - data: DataFrame filtré et préparé
    - particle: Nom de la colonne de la particule à tracer (ex: 'P2')
    - timeframe: Fréquence d'agrégation ('D', 'W-MON', 'MS')
    
    Returns:
    - Un objet plotly.graph_objects.Figure
    """
    # Dictionnaire de mapping pour les fréquences et les titres
    freq_map = {
        'D': ('D', 'Journalier', 'Date'),
        'W': ('W-MON', 'Hebdomadaire (début lundi)', 'Semaine'),
        'M': ('MS', 'Mensuel', 'Mois')
    }
    
    # Récupération des paramètres
    freq, freq_label, x_label = freq_map.get(timeframe, ('D', 'Journalier', 'Date'))
    
    # Agrégation des données
    df_agg = data.resample(freq, on='timestamp')[particle].mean().reset_index()
    
    # Vérification que les données ne sont pas vides après agrégation
    if df_agg.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée disponible pour la sélection actuelle",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Création du graphique avec Plotly
    fig = px.line(
        df_agg, 
        x='timestamp', 
        y=particle,
        title=f"<b>{particle} - Vue {freq_label}</b>",
        labels={'timestamp': x_label, particle: f'Valeur moyenne {particle}'}
    )
    
    # Personnalisation du graphique
    colors = {'D': '#1f77b4', 'W': '#2ca02c', 'M': '#ff7f0e'}  # Palette professionnelle
    line_color = colors.get(timeframe, '#1f77b4')
    
    fig.update_traces(
        line=dict(width=3, color=line_color),
        mode='lines+markers',
        marker=dict(size=6, color=line_color),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(int(line_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}'
    )
    
    # Mise en forme du layout
    fig.update_layout(
        plot_bgcolor='white',
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)',
            tickformat='%d/%m/%Y' if timeframe == 'D' else None
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.5)'
        ),
        title_font=dict(size=20),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# -------------------------------
# 3. FONCTION DE CALCUL DES KPIs
# -------------------------------
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

# -------------------------------
# INTERFACE UTILISATEUR PRINCIPALE
# -------------------------------
def main():
    # Titre principal
    st.title("📊 Dashboard d'Analyse des Particules")
    st.markdown("---")
    
    # Chargement des données
    with st.spinner('Chargement des données...'):
        df = load_and_prepare_data()
    
    # Vérification que les données sont chargées
    if df is None or df.empty:
        st.error("Erreur lors du chargement des données. Vérifiez le chemin du fichier.")
        return
    
    # -------------------------------
    # 4. BARRE LATÉRALE - FILTRES
    # -------------------------------
    st.sidebar.header("🔧 Filtres d'Analyse")
    
    # Filtre de période
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Période d'observation (optionnel)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtre de particule (obligatoire)
    particle_options = ['P0', 'P1', 'P2', 'temperature', 'humidity']
    selected_particle = st.sidebar.selectbox(
        "Particule / Variable à analyser",
        options=particle_options,
        index=2  # P2 par défaut
    )
    
    # Filtre de TimeFrame (obligatoire)
    timeframe_options = ['D', 'W', 'M']
    timeframe_labels = ['Journalier (D)', 'Hebdomadaire (W)', 'Mensuel (M)']
    selected_timeframe = st.sidebar.radio(
        "Granularité temporelle",
        options=timeframe_options,
        format_func=lambda x: dict(zip(timeframe_options, timeframe_labels))[x],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Note:** Les graphiques affichent toujours la **moyenne** pour la période sélectionnée.")
    
    # -------------------------------
    # 5. FILTRAGE DES DONNÉES
    # -------------------------------
    # Application du filtre de dates
 # Ensuite votre code devient plus simple
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_date = pd.Timestamp(start_date, tz='UTC')
        end_date = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
        #end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        
        df_filtered = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()   
    else:
        df_filtered = df.copy()
    
    # Suppression des autres colonnes de particules
    other_particles = [p for p in particle_options if p != selected_particle and p in df_filtered.columns]
    df_for_plot = df_filtered.drop(columns=other_particles)
    
    # -------------------------------
    # 6. CALCUL ET AFFICHAGE DES KPIs
    # -------------------------------
    st.header("📈 Indicateurs Clés")

    try:
        
        kpi_data = calculate_kpis(df_filtered, selected_particle, selected_timeframe)
        
        if kpi_data:
            # Création de 3 colonnes pour les KPIs
            col1, col2, col3 = st.columns(3)
            
            # KPI 1 : Minimum
            with col1:
                delta_display_min = f"{kpi_data['min_delta']:.2f}" if kpi_data['min_delta'] is not None else None
                st.metric(
                    label=f"Minimum ({kpi_data['period_label']})",
                    value=f"{kpi_data['min']:.2f}",
                    delta=delta_display_min
                )
            
            # KPI 2 : Moyenne avec variation
            with col2:
                delta_display = f"{kpi_data['mean_delta']:.2f}" if kpi_data['mean_delta'] is not None else None
                st.metric(
                    label=f"Moyenne ({kpi_data['period_label']})",
                    value=f"{kpi_data['mean']:.2f}",
                    delta=delta_display
                )
            
            # KPI 3 : Maximum
            with col3:
                delta_display_max = f"{kpi_data['max_delta']:.2f}" if kpi_data['max_delta'] is not None else None
                st.metric(
                    label=f"Maximum ({kpi_data['period_label']})",
                    value=f"{kpi_data['max']:.2f}",
                    delta=delta_display_max
                )
        else:
            st.warning("Données insuffisantes pour calculer les indicateurs.")
    except:
         st.warning("Données insuffisantes pour calculer les indicateurs avec vos parametres.")
         st.warning("Augmentez la plage ou reduisez le time frame.")

    
    st.markdown("---")
    
    # -------------------------------
    # 7. AFFICHAGE DU GRAPHIQUE
    # -------------------------------
    st.header("📊 Visualisation Temporelle")
    
    if df_for_plot.empty:
        st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
    else:
        # Création et affichage du graphique
        fig = create_time_series_plot(df_for_plot, selected_particle, selected_timeframe)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    # -------------------------------
    # 8. INFORMATIONS SUPPLEMENTAIRES
    # -------------------------------
    with st.expander("ℹ️ À propos de ce tableau de bord"):
        st.markdown("""
        **Fonctionnalités :**
        - Visualisation des données de particules (P0, P1, P2) et des variables environnementales
        - Agrégation temporelle : Journalière (D), Hebdomadaire (W), Mensuelle (M)
        - Indicateurs clés pour la dernière période complète
        - Interface de filtrage intuitive
        
        **Note technique :**
        - Tous les graphiques utilisent la **moyenne** comme fonction d'agrégation
        - La période "Hebdomadaire" commence le lundi (W-MON)
        - La période "Mensuelle" commence le premier jour du mois (MS)
        """)
        


# -------------------------------
# POINT D'ENTREE DE L'APPLICATION
# -------------------------------
if __name__ == "__main__":

    main()
