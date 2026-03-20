"""
Page d'Exploration Détaillée
Statistiques, distributions, corrélations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.processor import (
    load_and_prepare_data, calculate_kpis, get_available_locations,
    SUBSTANCE_MAP, SUBSTANCE_UNITS, DAY_EN_TO_FR, DAY_ORDER_FR,
    SENSOR_LOCATIONS,
)

st.set_page_config(page_title="Exploration — Nairobi Air", page_icon="🔍", layout="wide")


@st.cache_data(ttl=1800)
def get_data(location: int = 3966):
    return load_and_prepare_data(location=location)


def main():
    st.title("Exploration Detaillee des Donnees")
    st.markdown("---")

    # Location selector
    avail = get_available_locations()
    loc_ids = list(avail.keys()) if avail else ["3966"]
    loc_labels = {lid: SENSOR_LOCATIONS.get(int(lid), {}).get("name", f"Location {lid}")
                  for lid in loc_ids}
    selected_label = st.selectbox(
        "Localisation",
        [loc_labels[lid] for lid in loc_ids],
        index=loc_ids.index("3966") if "3966" in loc_ids else 0,
    )
    selected_loc = int([lid for lid, lbl in loc_labels.items() if lbl == selected_label][0])

    df = get_data(location=selected_loc)
    if df is None or df.empty:
        st.error("❌ Impossible de charger les données.")
        return

    # ── Sidebar ──
    with st.sidebar:
        st.header("🔧 Paramètres")

        substance = st.selectbox("Variable", list(SUBSTANCE_MAP.keys()), index=0)
        col = SUBSTANCE_MAP[substance]
        unit = SUBSTANCE_UNITS[substance]

        min_d, max_d = df['timestamp'].min().date(), df['timestamp'].max().date()
        date_range = st.date_input("Période", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)

        tf = st.radio("Granularité", ['D', 'W', 'M'],
                      format_func=lambda x: {'D': 'Jour', 'W': 'Semaine', 'M': 'Mois'}[x])

    # ── Filtrage ──
    df_f = df.copy()
    if len(date_range) == 2:
        s, e = date_range
        df_f = df_f[(df_f['timestamp'] >= pd.Timestamp(s, tz='UTC')) &
                    (df_f['timestamp'] <= pd.Timestamp(e, tz='UTC') + pd.Timedelta(days=1))]

    # ── KPIs ──
    st.header("📈 Indicateurs Clés")
    kpis = calculate_kpis(df_f, col, tf)
    if kpis:
        c1, c2, c3 = st.columns(3)
        with c1:
            d = f"{kpis['min_delta']:.2f}" if kpis['min_delta'] is not None else None
            st.metric(f"Minimum ({kpis['period_label']})", f"{kpis['min']:.2f}", delta=d)
        with c2:
            d = f"{kpis['mean_delta']:.2f}" if kpis['mean_delta'] is not None else None
            st.metric(f"Moyenne ({kpis['period_label']})", f"{kpis['mean']:.2f}", delta=d)
        with c3:
            d = f"{kpis['max_delta']:.2f}" if kpis['max_delta'] is not None else None
            st.metric(f"Maximum ({kpis['period_label']})", f"{kpis['max']:.2f}", delta=d)
    else:
        st.warning("Données insuffisantes pour calculer les indicateurs.")

    st.markdown("---")

    # ── Série temporelle ──
    st.header("📊 Série Temporelle")
    freq_map = {'D': 'D', 'W': 'W-MON', 'M': 'MS'}
    agg = df_f.resample(freq_map[tf], on='timestamp')[col].mean().reset_index()
    if not agg.empty:
        fig_ts = px.line(agg, x='timestamp', y=col,
                         title=f"{substance} — Vue {tf}",
                         labels={'timestamp': 'Date', col: f'{substance} ({unit})'})
        fig_ts.update_traces(line_color='#2d5aa0', fill='tozeroy',
                             fillcolor='rgba(45,90,160,0.1)')
        fig_ts.update_layout(plot_bgcolor='white', height=350)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Distribution ──
    st.header("📉 Distribution")
    c_hist, c_box = st.columns(2)
    with c_hist:
        fig_h = px.histogram(df_f, x=col, nbins=50, title=f"Distribution de {substance}",
                             labels={col: f'{substance} ({unit})'}, color_discrete_sequence=['#2d5aa0'])
        fig_h.add_vline(x=df_f[col].mean(), line_dash="dash", line_color="red",
                        annotation_text=f"Moy: {df_f[col].mean():.1f}")
        fig_h.update_layout(plot_bgcolor='white', height=320)
        st.plotly_chart(fig_h, use_container_width=True)

    with c_box:
        df_box = df_f.copy()
        df_box['jour_fr'] = df_box['jour_semaine'].map(DAY_EN_TO_FR)
        order = [d for d in DAY_ORDER_FR if d in df_box['jour_fr'].unique()]
        fig_b = px.box(df_box, x='jour_fr', y=col, title=f"{substance} par jour",
                       labels={'jour_fr': 'Jour', col: f'{substance} ({unit})'},
                       category_orders={'jour_fr': order}, color_discrete_sequence=['#2d5aa0'])
        fig_b.update_layout(plot_bgcolor='white', height=320)
        st.plotly_chart(fig_b, use_container_width=True)

    # ── Boxplot par heure ──
    st.header("🕐 Variation Horaire")
    fig_bh = px.box(df_f, x='heure', y=col,
                    title=f"{substance} par heure de la journée",
                    labels={'heure': 'Heure', col: f'{substance} ({unit})'},
                    color_discrete_sequence=['#2d5aa0'])
    fig_bh.update_layout(plot_bgcolor='white', height=350)
    st.plotly_chart(fig_bh, use_container_width=True)

    # ── Corrélation ──
    st.header("🔗 Corrélations")
    num_cols = ['P0', 'P1', 'P2', 'humidity', 'temperature']
    corr = df_f[num_cols].corr()
    fig_c = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                      title="Matrice de corrélation", aspect="auto")
    fig_c.update_layout(height=400)
    st.plotly_chart(fig_c, use_container_width=True)

    # ── Données brutes ──
    with st.expander("📋 Données brutes"):
        st.dataframe(df_f.describe())
        st.dataframe(df_f.head(50))


if __name__ == "__main__":
    main()
