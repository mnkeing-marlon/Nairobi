"""
Dashboard d'Analyse des Particules - Page Principale
Point d'entrée de l'application Streamlit
"""
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Analyse Particules",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page d'accueil
st.title("📊 Dashboard d'Analyse des Particules")
st.markdown("---")

st.markdown("""
## Bienvenue sur le Dashboard d'Analyse des Particules

Cette application permet d'analyser et de visualiser les données de particules atmosphériques.

### 📑 Pages disponibles :

- **🔍 Exploration** : Analyse exploratoire des données avec visualisations interactives et KPIs
- **🎯 Prédiction** : Modèles prédictifs et analyses avancées (à venir)

### 🚀 Pour commencer :

Utilisez le menu latéral pour naviguer entre les différentes pages de l'application.

---

### ℹ️ À propos de cette application

**Fonctionnalités :**
- Visualisation des données de particules (P0, P1, P2) et des variables environnementales
- Agrégation temporelle : Journalière, Hebdomadaire, Mensuelle
- Indicateurs clés pour la dernière période complète
- Interface de filtrage intuitive
- Architecture modulaire multi-pages

**Technologies utilisées :**
- Streamlit pour l'interface web
- Pandas pour le traitement des données
- Plotly pour les visualisations interactives
""")

st.markdown("---")
st.info("👈 Sélectionnez une page dans le menu latéral pour commencer l'analyse")
