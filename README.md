# 📊 Dashboard d'Analyse des Particules - Nairobi

Exploration et analyse des données environnementales dans certaines localités de la ville de Nairobi.

## 🏗️ Structure du Projet

```
Nairobi/
├── .streamlit/
│   └── config.toml          # Configuration Streamlit (thèmes, port, etc.)
├── data/
│   └── modele_3966_Aug_Dec2025.csv  # Fichiers de données
├── pages/                   # Pages multi-pages Streamlit
│   ├── 01_Exploration.py    # Page d'exploration des données
│   └── 02_Prediction.py     # Page de prédiction (en développement)
├── src/                     # Code logique (calculs, modèles)
│   ├── __init__.py
│   └── processor.py         # Fonctions de traitement des données
├── .gitignore
├── app.py                   # Point d'entrée principal
├── README.md
└── requirements.txt         # Dépendances Python
```

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/mnkeing-marlon/Nairobi.git
cd Nairobi
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

### 3. Activer l'environnement virtuel

**Windows :**
```bash
venv\Scripts\activate
```

**Linux/Mac :**
```bash
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 🎯 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## 📋 Fonctionnalités

### Page d'Exploration (01_Exploration.py)

- ✅ Visualisation des données de particules (P0, P1, P2)
- ✅ Visualisation des variables environnementales (température, humidité)
- ✅ Agrégation temporelle : Journalière, Hebdomadaire, Mensuelle
- ✅ Indicateurs clés (KPIs) : Min, Moyenne, Max avec variations
- ✅ Interface de filtrage intuitive
- ✅ Graphiques interactifs avec Plotly

### Page de Prédiction (02_Prediction.py)

- 🚧 Modèles de prédiction (en développement)
- 🚧 Analyses avancées (en développement)
- 🚧 Détection d'anomalies (en développement)

## 📦 Dépendances

- **streamlit** : Framework d'application web
- **pandas** : Traitement et analyse des données
- **plotly** : Visualisations interactives
- **python-dateutil** : Gestion des dates

## 🛠️ Technologies Utilisées

- **Python 3.8+**
- **Streamlit** pour l'interface web
- **Pandas** pour le traitement des données
- **Plotly** pour les visualisations interactives

## 📊 Données

Les données sont stockées dans le dossier `data/` et contiennent :
- Données de particules atmosphériques (P0, P1, P2)
- Variables environnementales (température, humidité)
- Timestamps pour l'analyse temporelle

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou soumettre une pull request.

## 📝 Licence

Ce projet est sous licence MIT.

## 👤 Auteur

**mnkeing-marlon**

- GitHub: [@mnkeing-marlon](https://github.com/mnkeing-marlon)
