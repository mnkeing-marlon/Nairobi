"""
Page de Détails du Modèle Prédictif
Performance, feature importance, actual vs predicted
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.processor import load_and_prepare_data, get_available_locations, SENSOR_LOCATIONS, SUBSTANCE_MAP
from src.model import load_or_train_model, TARGET_CONFIG

st.set_page_config(page_title="Prédiction — Nairobi Air", page_icon="🎯", layout="wide")


@st.cache_data(ttl=1800)
def get_data(location: int = 3966):
    return load_and_prepare_data(location=location)

@st.cache_resource(ttl=1800)
def get_model(location: int = 3966, target: str = 'P2'):
    df = load_and_prepare_data(location=location)
    return load_or_train_model(df, target=target)


def main():
    st.title("Modele Predictif -- Details & Performance")
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

    # Substance selector
    substance = st.selectbox("Substance", list(SUBSTANCE_MAP.keys()), index=0)
    target = SUBSTANCE_MAP[substance]  # 'P2' or 'P1'
    cfg = TARGET_CONFIG[target]
    lag_feats = cfg['lag_features']

    df = get_data(location=selected_loc)
    if df is None or df.empty:
        st.error("❌ Impossible de charger les données.")
        return

    with st.spinner("Chargement du modèle…"):
        model, metrics = get_model()

    mae = metrics['mae']
    r_std = metrics.get('residual_std', 0)
    params = metrics.get('best_params', {})
    importance = metrics.get('importance', {})
    y_true = metrics.get('test_actual', [])
    y_pred = metrics.get('test_predicted', [])

    # ── Résumé ──
    st.header("📋 Résumé du Modèle")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", "Prophet")
    c2.metric("MAE", f"{mae:.2f} µg/m³")
    c3.metric("Écart-type résidus", f"{r_std:.2f}")
    c4.metric("Features", str(len(lag_feats)))

    # ── Paramètres ──
    with st.expander("Meilleurs hyper-parametres (Prophet)"):
        if params:
            cols = st.columns(len(params))
            for i, (k, v) in enumerate(params.items()):
                cols[i].metric(k, str(v))
        else:
            st.info("Paramètres non disponibles.")

    st.markdown("---")

    # ── Feature Importance ──
    st.header("📊 Importance des Features")
    if importance:
        imp_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values()),
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         title=f"Importance des variables ({substance} Prophet)",
                         color='Importance', color_continuous_scale='Blues')
        fig_imp.update_layout(plot_bgcolor='white', height=350, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance non disponible.")

    st.markdown("---")

    # ── Actual vs Predicted ──
    st.header("🎯 Prédictions vs Réalité (Jeu de Test)")

    if y_true and y_pred:
        n = min(200, len(y_true))
        idx = list(range(n))

        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=idx, y=y_true[:n], mode='lines',
            name='Réalité', line=dict(color='#2d5aa0', width=2)))
        fig_ap.add_trace(go.Scatter(
            x=idx, y=y_pred[:n], mode='lines',
            name='Prédiction', line=dict(color='#e53e3e', width=2, dash='dash')))
        fig_ap.update_layout(
            title=f"200 premières heures du jeu de test (MAE = {mae:.2f})",
            xaxis_title="Heures", yaxis_title=f"{substance} (µg/m³)",
            plot_bgcolor='white', height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_ap, use_container_width=True)

        # ── Scatter actual vs predicted ──
        c_sc, c_res = st.columns(2)
        with c_sc:
            fig_sc = px.scatter(x=y_true, y=y_pred, labels={'x': 'Réalité', 'y': 'Prédiction'},
                                title="Scatter : Réalité vs Prédiction", opacity=0.4)
            max_v = max(max(y_true), max(y_pred))
            fig_sc.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines',
                                        line=dict(color='red', dash='dash'), name='Idéal'))
            fig_sc.update_layout(plot_bgcolor='white', height=350)
            st.plotly_chart(fig_sc, use_container_width=True)

        with c_res:
            residuals = [a - p for a, p in zip(y_true, y_pred)]
            fig_res = px.histogram(residuals, nbins=50, title="Distribution des résidus",
                                   labels={'value': 'Résidu (µg/m³)'}, color_discrete_sequence=['#2d5aa0'])
            fig_res.add_vline(x=0, line_dash="dash", line_color="red")
            fig_res.update_layout(plot_bgcolor='white', height=350, showlegend=False)
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Données de test non disponibles. Re-lancez l'entraînement du modèle.")

    # ── Note méthodologique ──
    st.markdown("---")
    with st.expander("Note methodologique"):
        st.markdown(f"""
        **Modele :** Prophet (Facebook) avec regresseurs lag

        **Substance :** {substance}

        **Features :** {len(lag_feats)} variables de lag temporel
        - `{target}_lag_1` : {substance} il y a 1 heure
        - `{target}_lag_2` : {substance} il y a 2 heures
        - `{target}_lag_3` : {substance} il y a 3 heures
        - `{target}_lag_4` : {substance} il y a 4 heures
        - `{target}_lag_5` : {substance} il y a 5 heures
        - `{target}_lag_24` : {substance} il y a 24 heures (meme heure, veille)

        **Entrainement :** Sur les donnees de Kibera (location 3966),
        transfere a toutes les autres locations.

        **Validation :** Split chronologique, 168 dernieres heures en test.
        """)


if __name__ == "__main__":
    main()
