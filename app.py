"""
Dashboard Qualité de l'Air — Nairobi
Interface principale : analyse descriptive + prédictive
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.processor import (
    load_and_prepare_data,
    calculate_aqi,
    get_aqi_category,
    check_who_exceedance,
    get_heatmap_data,
    get_last_n_hours,
    get_available_locations,
    get_all_locations_aqi,
    SENSOR_LOCATIONS,
    SUBSTANCE_MAP,
    SUBSTANCE_UNITS,
)
from src.model import load_or_train_model, predict_next_24h, predict_next_week, predict_next_n_hours, get_trend, TARGET_CONFIG

# ─── Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="Qualité de l'Air — Nairobi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Barre d'en-tête */
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2d5aa0 100%);
        color: white; padding: 15px 25px; border-radius: 8px;
        margin-bottom: 20px; margin-top:40px; font-size: 24px; font-weight: bold;
    }
    /* Titres de section */
    .section-header {
        background: #f0f4f8; border-left: 4px solid #2d5aa0;
        padding: 10px 15px; margin: 20px 0 12px 0;
        font-weight: bold; font-size: 16px; color: #1a365d;
    }
    /* Cartes KPI */
    .kpi-card {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 15px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); height: 100%;
    }
    .kpi-label { font-size: 11px; color: #718096; font-weight: 600;
                 text-transform: uppercase; margin-bottom: 5px; }
    .kpi-value { font-size: 26px; font-weight: bold; color: #1a365d; }
    .kpi-unit  { font-size: 13px; color: #a0aec0; }
    /* Alerte */
    .alert-banner {
        background: #fff8e1; border: 1px solid #ffcc02;
        border-radius: 6px; padding: 10px 15px;
        font-size: 13px; color: #856404; margin: 10px 0;
    }
    /* Légende carte */
    .legend-item { display:inline-flex; align-items:center;
                   margin-right:14px; font-size:11px; }
    .legend-dot  { width:11px; height:11px; border-radius:50%;
                   display:inline-block; margin-right:4px; }
    /* Streamlit overrides */
    div[data-testid="stMetric"] {
        background-color: white; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 10px 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Chargement & cache ──────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des donnees...", ttl=1800)
def get_data(location: int = 3966):
    return load_and_prepare_data(location=location)

@st.cache_resource(show_spinner="Chargement du modele Prophet...", ttl=1800)
def get_model(location: int = 3966, target: str = 'P2'):
    df = load_and_prepare_data(location=location)
    return load_or_train_model(df, target=target)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # ─── SIDEBAR ── FILTRES ───────────────────────────────────────
    with st.sidebar:
        st.markdown("## FILTRES")

        # 0 — Location selector
        st.markdown("**LOCALISATION (CAPTEUR)**")
        avail = get_available_locations()
        if avail:
            loc_ids = list(avail.keys())
            loc_labels = {lid: SENSOR_LOCATIONS.get(int(lid), {}).get("name", f"Location {lid}")
                          for lid in loc_ids}
            selected_label = st.selectbox(
                "loc_sel", [loc_labels[lid] for lid in loc_ids],
                index=loc_ids.index("3966") if "3966" in loc_ids else 0,
                label_visibility="collapsed",
            )
            selected_loc = int([lid for lid, lbl in loc_labels.items() if lbl == selected_label][0])
        else:
            selected_loc = 3966
        st.markdown("---")

    df = get_data(location=selected_loc)
    if df is None or df.empty:
        st.error("Impossible de charger les donnees pour cette localisation.")
        return

    with st.sidebar:
        # 1 — Substance
        st.markdown("**SUBSTANCE**")
        substance = st.selectbox("sub", list(SUBSTANCE_MAP.keys()),
                                 index=0, label_visibility="collapsed")
        col_name = SUBSTANCE_MAP[substance]
        unit = SUBSTANCE_UNITS[substance]
        st.markdown("---")

        # 2 — Timeframe
        st.markdown("**TIMEFRAME**")
        tf_opts = {"24h": 24, "7 jours": 168, "30 jours": 720, "Personnalise": None}
        timeframe = st.selectbox("tf", list(tf_opts.keys()),
                                 index=0, label_visibility="collapsed")
        custom_range = None
        if timeframe == "Personnalise":
            custom_range = st.date_input(
                "Periode",
                value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
                min_value=df['timestamp'].min().date(),
                max_value=df['timestamp'].max().date(),
            )
        st.markdown("---")

        # 4 — Type de jour
        st.markdown("**TYPE DE JOUR\\***")
        type_jour = st.selectbox("tj", ["Tous", "Semaine", "Week-end"],
                                 index=0, label_visibility="collapsed")
        st.markdown("---")

        # 5 — Statut capteur
        st.markdown("**STATUT DU CAPTEUR\\***")
        statut = st.selectbox("st", ["Tous", "Actif", "Inactif"],
                              index=0, label_visibility="collapsed")
        st.markdown("---")

        st.button("🔍 Appliquer Filtres", type="primary", use_container_width=True)

    # ─── APPLICATION DES FILTRES ──────────────────────────────────
    df_f = df.copy()

    # Timeframe
    hours = tf_opts.get(timeframe)
    if hours:
        df_f = get_last_n_hours(df_f, hours)
    elif timeframe == "Personnalisé" and custom_range and len(custom_range) == 2:
        s, e = custom_range
        df_f = df_f[(df_f['timestamp'] >= pd.Timestamp(s, tz='UTC')) &
                    (df_f['timestamp'] <= pd.Timestamp(e, tz='UTC') + pd.Timedelta(days=1))]

    # Type de jour
    if type_jour == "Semaine":
        df_f = df_f[df_f['periode'] == 'semaine']
    elif type_jour == "Week-end":
        df_f = df_f[df_f['periode'] == 'weekend']

    if df_f.empty:
        st.warning("⚠️ Aucune donnée pour les filtres sélectionnés.")
        return

    # Heatmap : sur données complètes (plus de patterns)
    df_heat = df.copy()
    if type_jour == "Semaine":
        df_heat = df_heat[df_heat['periode'] == 'semaine']
    elif type_jour == "Week-end":
        df_heat = df_heat[df_heat['periode'] == 'weekend']

    # ─── EN-TÊTE ──────────────────────────────────────────────────
    st.markdown('<div class="main-header">QUALITÉ DE L\'AIR — NAIROBI</div>',
                unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # LIGNE 1 — Jauge AQI + carte
    # ═════════════════════════════════════════════════════════════
    col_aqi, col_map = st.columns([2, 3])

    mean_pm25 = df_f['P2'].mean()
    aqi_val = calculate_aqi(mean_pm25, "PM2.5")
    cat_label, cat_color, _ = get_aqi_category(aqi_val)

    with col_aqi:
        # — Jauge AQI —
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            number=dict(font=dict(size=42, color=cat_color)),
            title=dict(text="SCORE AQI ACTUEL (MOYEN)*", font=dict(size=13)),
            gauge=dict(
                axis=dict(range=[0, 300], tickwidth=1),
                bar=dict(color=cat_color, thickness=0.3),
                bgcolor="white",
                steps=[
                    dict(range=[0,   50],  color='#d4edda'),
                    dict(range=[50,  100], color='#fff3cd'),
                    dict(range=[100, 150], color='#ffe0b2'),
                    dict(range=[150, 200], color='#f8d7da'),
                    dict(range=[200, 300], color='#e2c6e9'),
                ],
                threshold=dict(line=dict(color=cat_color, width=4),
                               thickness=0.75, value=aqi_val),
            ),
        ))
        fig_g.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10),
                            paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g, use_container_width=True, key="gauge")

        st.markdown(
            f'<div style="text-align:center;margin-top:-10px;">'
            f'<span style="color:{cat_color};font-weight:bold;font-size:18px;">'
            f'{cat_label} ({aqi_val})</span></div>',
            unsafe_allow_html=True,
        )

        # — Alerte —
        exceeds, thr = check_who_exceedance(mean_pm25, "PM2.5")
        if exceeds:
            st.markdown(
                '<div class="alert-banner">'
                '⚠️ <strong>ALERTE*:</strong> Prédiction PM2.5 élevée pour demain matin.'
                '</div>',
                unsafe_allow_html=True,
            )

    with col_map:
        st.markdown('<div class="section-header">ANALYSE DESCRIPTIVE (LE PRÉSENT)</div>',
                    unsafe_allow_html=True)

        # — Carte Mapbox — real per-location AQI —
        loc_aqi_data = get_all_locations_aqi(substance)

        fig_map = go.Figure()
        for loc in loc_aqi_data:
            is_selected = (loc["loc_id"] == selected_loc)
            fig_map.add_trace(go.Scattermapbox(
                lat=[loc["lat"]], lon=[loc["lon"]], mode='markers',
                marker=dict(
                    size=24 if is_selected else 14,
                    color=loc["color"],
                    opacity=1.0 if is_selected else 0.65,
                ),
                text=(
                    f"<b>{loc['name']}</b> (#{loc['loc_id']})<br>"
                    f"AQI {substance}: {loc['aqi']} ({loc['label']})"
                    + ("<br><i>\u2714 Selectionne</i>" if is_selected else "")
                ),
                hoverinfo='text', showlegend=False,
            ))

        # Center map on the selected location if it has coordinates
        sel_info = SENSOR_LOCATIONS.get(selected_loc, {})
        center_lat = sel_info.get("lat", -1.283)
        center_lon = sel_info.get("lon", 36.815)

        fig_map.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=center_lat, lon=center_lon), zoom=12),
            height=310, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True, key="map")

        st.markdown(
            '<div style="font-size:11px;color:#555;padding:3px 0;">'
            '<span class="legend-item"><span class="legend-dot" style="background:#00e400"></span>Bon</span>'
            '<span class="legend-item"><span class="legend-dot" style="background:#f0c800"></span>Modere</span>'
            '<span class="legend-item"><span class="legend-dot" style="background:#ff7e00"></span>Malsain sensibles</span>'
            '<span class="legend-item"><span class="legend-dot" style="background:#ff0000"></span>Malsain</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ═════════════════════════════════════════════════════════════
    # LIGNE 2 — Analyse descriptive
    # ═════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">ANALYSE DESCRIPTIVE (LE PRÉSENT)</div>',
                unsafe_allow_html=True)

    mean_val = df_f[col_name].mean()
    min_val  = df_f[col_name].min()
    max_val  = df_f[col_name].max()
    exc_who, thr_who = check_who_exceedance(mean_val, substance)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">VALEUR MOYENNE ({substance})</div>'
            f'<div class="kpi-value">{mean_val:.0f} <span class="kpi-unit">{unit}</span></div></div>',
            unsafe_allow_html=True)
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">MIN / MAX ({timeframe})</div>'
            f'<div class="kpi-value">{min_val:.0f} / {max_val:.0f} <span class="kpi-unit">{unit}</span></div></div>',
            unsafe_allow_html=True)
    with k3:
        who_txt = f"Oui (>{thr_who} {unit})" if exc_who else f"Non (>{thr_who} {unit})"
        who_clr = "#dc3545" if exc_who else "#28a745"
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">DÉPASSEMENT SEUILS OMS*</div>'
            f'<div class="kpi-value" style="color:{who_clr}">{who_txt}</div></div>',
            unsafe_allow_html=True)

    # — Graphiques descriptifs —
    c_chart, c_heat = st.columns([2, 1])

    with c_chart:
        st.markdown(f"**ÉVOLUTION SUR LES DERNIÈRES {timeframe}**")
        fig_evo = go.Figure()
        fig_evo.add_trace(go.Scatter(
            x=df_f['timestamp'], y=df_f[col_name],
            mode='lines', line=dict(color='#2d5aa0', width=2),
            fill='tozeroy', fillcolor='rgba(45,90,160,0.1)',
            name=substance,
            hovertemplate='%{x|%d/%m %H:%M}<br>' + substance + ': %{y:.1f} ' + unit + '<extra></extra>',
        ))
        tick_fmt = '%H:%M' if timeframe == '24h' else ('%d/%m' if timeframe == '7 jours' else '%d/%m/%y')
        fig_evo.update_layout(
            height=280, margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,.3)', tickformat=tick_fmt),
            yaxis=dict(title=f'{substance} ({unit})', showgrid=True, gridcolor='rgba(200,200,200,.3)'),
            plot_bgcolor='white', hovermode='x unified',
        )
        st.plotly_chart(fig_evo, use_container_width=True, key="evo")

    with c_heat:
        st.markdown("**HEATMAP HORAIRE\\***")
        hm = get_heatmap_data(df_heat, col_name)
        if hm is not None and not hm.empty:
            fig_hm = go.Figure(go.Heatmap(
                z=hm.values,
                x=[str(h) for h in hm.columns],
                y=hm.index.tolist(),
                colorscale='YlOrRd', showscale=True,
                colorbar=dict(title=dict(text=unit, side='right'), thickness=12),
                hovertemplate='Jour: %{y}<br>Heure: %{x}h<br>Valeur: %{z:.1f}<extra></extra>',
            ))
            fig_hm.update_layout(
                height=280, margin=dict(l=40, r=60, t=20, b=40),
                xaxis=dict(title="Heure", dtick=2),
                yaxis=dict(title=""),
                plot_bgcolor='white',
            )
            st.plotly_chart(fig_hm, use_container_width=True, key="heatmap")

    # ═════════════════════════════════════════════════════════════
    # LIGNE 3 — Analyse prédictive
    # ═════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">ANALYSE PRÉDICTIVE (LE FUTUR)</div>',
                unsafe_allow_html=True)

    if col_name in TARGET_CONFIG:
        target = col_name  # 'P2' or 'P1'
        with st.spinner("Chargement du modele predictif..."):
            model, metrics = get_model(location=selected_loc, target=target)

        mae = metrics['mae']
        r_std = metrics.get('residual_std', mae * 1.2)

        # Déterminer l'horizon de prédiction basé sur le timeframe
        horizon_hours = tf_opts.get(timeframe, 24)
        if horizon_hours is None:  # Personnalisé
            horizon_hours = 24
        
        # Déterminer le label d'horizon
        if horizon_hours == 24:
            horizon_label = "24h"
            preds = predict_next_24h(model, df, mae, r_std, target=target)
        elif horizon_hours == 168:
            horizon_label = "7 jours"
            preds = predict_next_week(model, df, mae, r_std, target=target)
        else:
            horizon_label = f"{horizon_hours}h"
            preds = predict_next_n_hours(model, df, horizon_hours, mae, r_std, target=target)
        
        p_mean = preds['predicted'].mean()
        p_min  = preds['predicted'].min()
        p_max  = preds['predicted'].max()
        mean_current = df_f[col_name].mean()
        arrow, trend_lbl, trend_clr = get_trend(preds, mean_current)

        # KPIs prédictifs
        pk1, pk2, pk3 = st.columns(3)
        with pk1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">VALEUR MOYENNE PRÉDITE ({horizon_label})</div>'
                f'<div class="kpi-value">{p_mean:.0f} <span class="kpi-unit">{unit}</span></div></div>',
                unsafe_allow_html=True)
        with pk2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">MIN / MAX PRÉDITE ({horizon_label})</div>'
                f'<div class="kpi-value">{p_min:.0f} / {p_max:.0f} <span class="kpi-unit">{unit}</span></div></div>',
                unsafe_allow_html=True)
        with pk3:
            quality = "Bonne" if mae < 5 else ("Correcte" if mae < 10 else "Moyenne")
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">FIABILITÉ DU MODÈLE (MAE)*</div>'
                f'<div class="kpi-value">±{mae:.0f} <span class="kpi-unit">{unit} ({quality})</span></div></div>',
                unsafe_allow_html=True)

        # Graphique prédiction + tendance
        cp, ct = st.columns([3, 1])

        with cp:
            st.markdown(f"**PRÉDICTION {substance} ({horizon_label.upper()})**")

            x_band = list(preds['timestamp']) + list(preds['timestamp'][::-1])
            y_band = list(preds['upper']) + list(preds['lower'][::-1])

            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill='toself', fillcolor='rgba(45,90,160,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', name='Intervalle de confiance',
            ))
            fig_p.add_trace(go.Scatter(
                x=preds['timestamp'], y=preds['predicted'],
                mode='lines', line=dict(color='#2d5aa0', width=2.5),
                name='Prédiction',
                hovertemplate='%{x|%d/%m %H:%M}<br>' + substance + ': %{y:.1f} µg/m³<extra></extra>',
            ))
            who_threshold = 15 if target == 'P2' else 45
            fig_p.add_hline(y=who_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Seuil OMS ({who_threshold} µg/m³)",
                            annotation_position="top right")
            
            # Ajuster le format de l'axe x selon l'horizon
            if horizon_hours <= 24:
                x_tickformat = '%H:%M'
            elif horizon_hours <= 168:
                x_tickformat = '%d/%m'
            else:
                x_tickformat = '%d/%m'
            
            fig_p.update_layout(
                height=280, margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,.3)', tickformat=x_tickformat),
                yaxis=dict(title=f'{substance} (µg/m³)', showgrid=True, gridcolor='rgba(200,200,200,.3)'),
                plot_bgcolor='white',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_p, use_container_width=True, key="pred")

        with ct:
            st.markdown("**TENDANCE PRÉVUE\\***")
            st.markdown(
                f'<div style="text-align:center;padding:30px 10px;">'
                f'<div style="font-size:56px;color:{trend_clr};">{arrow}</div>'
                f'<div style="font-size:16px;font-weight:bold;color:{trend_clr};">{trend_lbl.upper()}</div>'
                f'<div style="font-size:11px;color:#718096;margin-top:10px;">'
                f'Moy. prédite: {p_mean:.0f} {unit}<br>vs actuelle: {mean_current:.0f} {unit}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info(
            f"Les predictions ne sont pas disponibles pour cette substance."
        )

    # ═════════════════════════════════════════════════════════════
    # EXPORT
    # ═════════════════════════════════════════════════════════════
    st.markdown("---")
    _, col_exp = st.columns([4, 1])
    with col_exp:
        export = df_f[['timestamp', col_name]].copy()
        export.columns = ['Horodatage', substance]
        st.download_button(
            label="📥 EXPORT DE DONNÉES*",
            data=export.to_csv(index=False).encode('utf-8'),
            file_name=f"nairobi_{substance}_{timeframe}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
