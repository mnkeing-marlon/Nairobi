"""
Page d'Exploration des Données
Analyse exploratoire avec visualisations et KPIs
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Ajouter le répertoire parent au path pour importer src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.processor import load_and_prepare_data, calculate_kpis

# Configuration de la page
st.set_page_config(
    page_title="Exploration - Dashboard Particules",
    page_icon="🔍",
    layout="wide"
)


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


def main():
    # Titre principal
    st.title("🔍 Exploration des Données")
    st.markdown("---")
    
    # Chargement des données
    with st.spinner('Chargement des données...'):
        df = st.cache_data(load_and_prepare_data)()
    
    # Vérification que les données sont chargées
    if df is None or df.empty:
        st.error("Erreur lors du chargement des données. Vérifiez le chemin du fichier.")
        return
    
    # -------------------------------
    # BARRE LATÉRALE - FILTRES
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
    # FILTRAGE DES DONNÉES
    # -------------------------------
    # Application du filtre de dates
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_date = pd.Timestamp(start_date, tz='UTC')
        end_date = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
        
        df_filtered = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()   
    else:
        df_filtered = df.copy()
    
    # Suppression des autres colonnes de particules
    other_particles = [p for p in particle_options if p != selected_particle and p in df_filtered.columns]
    df_for_plot = df_filtered.drop(columns=other_particles)
    
    # -------------------------------
    # CALCUL ET AFFICHAGE DES KPIs
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
        st.warning("Données insuffisantes pour calculer les indicateurs avec vos paramètres.")
        st.warning("Augmentez la plage ou réduisez le time frame.")

    st.markdown("---")
    
    # -------------------------------
    # AFFICHAGE DU GRAPHIQUE
    # -------------------------------
    st.header("📊 Visualisation Temporelle")
    
    if df_for_plot.empty:
        st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
    else:
        # Création et affichage du graphique
        fig = create_time_series_plot(df_for_plot, selected_particle, selected_timeframe)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    
    # -------------------------------
    # INFORMATIONS SUPPLÉMENTAIRES
    # -------------------------------
    with st.expander("ℹ️ À propos de cette page"):
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


if __name__ == "__main__":
    main()
