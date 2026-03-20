# Nairobi Air Quality — Monitoring & Forecasting Platform

Real-time air quality dashboard for Nairobi, Kenya.
Scrapes sensor data from [sensors.AFRICA / openAFRICA](https://open.africa/dataset/sensorsafrica-airquality-archive-nairobi), processes it per location, trains a **Prophet** time-series model, and serves interactive Streamlit pages with AQI gauges, trend charts, heatmaps, and recursive PM2.5 forecasts.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Running the Full Pipeline](#running-the-full-pipeline)
6. [Training the Model](#training-the-model)
7. [Running the Dashboard](#running-the-dashboard)
8. [Scheduled Automation (Windows)](#scheduled-automation-windows)
9. [Testing & Validation](#testing--validation)
10. [Hosting & Deployment](#hosting--deployment)
11. [Project Structure](#project-structure)
12. [Environment Variables & Config](#environment-variables--config)
13. [Troubleshooting](#troubleshooting)

---

## Architecture

```
openAFRICA CKAN API
       |
       v
  scraper.py          <-- downloads raw CSVs every 20 min
       |
       v
  data/raw_data/*.csv
       |
       v
  src/pipeline.py     <-- pivot, hourly agg, lag features, per-location CSVs
       |
       v
  data/processed/location_{id}.csv   +  _locations.json
       |
       v
  src/model.py        <-- Prophet trained on location 3966 (Kibera)
       |
       v
  models/prophet_pm25.joblib  +  metrics.joblib
       |
       v
  app.py  <--  Streamlit dashboard (multi-page)
  pages/01_Exploration.py
  pages/02_Prediction.py
```

**Key design decisions:**

| Decision | Detail |
|---|---|
| Model | Facebook Prophet with 6 lag regressors (P2_lag_1..5, P2_lag_24) |
| Transfer approach | Trained on Kibera (3966), applied to all locations |
| Data window | 2024-01-01 onward |
| Auto-detection | Locations with >= 1 000 hourly records are included |
| Cache | Streamlit `@st.cache_data` / `@st.cache_resource` with 30-min TTL |

---

## Prerequisites

- **Python 3.11+** (tested on 3.13)
- **pip** or **pipx**
- **Git**
- Internet connection (for scraping and Prophet's CmdStan backend)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnkeing-marlon/Nairobi.git
cd Nairobi

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

> **Note:** The `prophet` package compiles CmdStan on first import.
> If you run into build issues on Windows, install the
> [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> or use `conda install -c conda-forge prophet`.

---

## Quick Start

Run the entire pipeline (scrape, process, train, dashboard) with three commands:

```bash
# Step 1 — Scrape raw data from openAFRICA
python scraper.py --once

# Step 2 — Process data + train Prophet model
python run_pipeline_full.py --no-scrape --force-train

# Step 3 — Launch the dashboard
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**.

---

## Running the Full Pipeline

`run_pipeline_full.py` is the single entry-point orchestrator:

```bash
# Scrape + process + retrain (only if new data found)
python run_pipeline_full.py

# Skip scraping, just reprocess and retrain
python run_pipeline_full.py --no-scrape --force-train

# Force retrain even if no new data
python run_pipeline_full.py --force-train
```

**What it does:**

| Step | Module | Action |
|------|--------|--------|
| 1 | `scraper.py` | Fetches resources from the CKAN API, downloads new/updated CSVs |
| 2 | `src/pipeline.py` | Loads all raw CSVs, filters to 2024+, detects locations, pivots, aggregates hourly, adds lag features, saves per-location CSVs |
| 3 | `src/model.py` | Trains Prophet on location 3966, saves `models/prophet_pm25.joblib` |

---

## Training the Model

If you only want to retrain without running the full pipeline:

```bash
# Train only if no saved model exists
python train_model.py

# Force retrain
python train_model.py --force

# Run pipeline first, then train
python train_model.py --pipeline
```

**Output:**

- `models/prophet_pm25.joblib` — serialized Prophet model
- `models/metrics.joblib` — MAE, residual_std, test predictions

---

## Running the Dashboard

```bash
streamlit run app.py
```

| Page | URL path | Description |
|------|----------|-------------|
| Main Dashboard | `/` | AQI gauge, map, time series, heatmap, predictions |
| Exploration | `/01_Exploration` | Distributions, correlations, KPIs |
| Prediction Details | `/02_Prediction` | Model performance, feature importance, residuals |

All pages include a **location selector** — choose any detected sensor location from the sidebar.

---

## Scheduled Automation (Windows)

To run the pipeline automatically every 30 minutes via Windows Task Scheduler:

```powershell
# Run as Administrator
.\setup_scheduler.ps1
```

This creates a task named `NairobiAQ_Pipeline` that executes `run_pipeline_full.py` every 30 minutes.

To remove it:

```powershell
Unregister-ScheduledTask -TaskName "NairobiAQ_Pipeline"
```

**Linux / macOS alternative (cron):**

```bash
crontab -e

# Add this line (adjust paths):
*/30 * * * * cd /path/to/Nairobi && /path/to/venv/bin/python run_pipeline_full.py >> logs/cron.log 2>&1
```

---

## Testing & Validation

### Syntax check all modules

```bash
python -m py_compile src/pipeline.py
python -m py_compile src/model.py
python -m py_compile src/processor.py
python -m py_compile train_model.py
python -m py_compile run_pipeline_full.py
python -m py_compile app.py
python -m py_compile pages/01_Exploration.py
python -m py_compile pages/02_Prediction.py
```

### Test the data pipeline

```bash
python -c "from src.pipeline import run_pipeline; r = run_pipeline(); print(f'{len(r)} locations processed')"
```

Expected: `5 locations processed` (depends on available raw data).

### Test model training

```bash
python train_model.py --force
```

Expected output includes `MAE = X.XX` and model saved confirmation.

### Test predictions (quick smoke test)

```python
from src.processor import load_and_prepare_data
from src.model import load_or_train_model, predict_next_24h

df = load_and_prepare_data(3966)
model, metrics = load_or_train_model(df)
preds = predict_next_24h(model, df, metrics['mae'], metrics['residual_std'])
print(preds.head())
# Should output 24 rows with columns: timestamp, predicted, lower, upper
```

### Test scraper (single cycle, no loop)

```bash
python scraper.py --once
```

### Validate dashboard starts

```bash
streamlit run app.py --server.headless true
# Open http://localhost:8501 in a browser
```

---

## Hosting & Deployment

### Option 1 — Streamlit Community Cloud (free, easiest)

1. Push your code to GitHub (it's already there).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** and select repo `mnkeing-marlon/Nairobi`, branch `feat/ai-dashboard-suggestion`, main file `app.py`.
4. Click **Deploy**.

> **Important:** Streamlit Cloud runs the app only — it does not run the scraper or pipeline.
> You need to either:
> - Commit processed data (`data/processed/`) and model files (`models/`) to the repo (remove those lines from `.gitignore`), **or**
> - Use a scheduled GitHub Action to keep them updated (see Option 4).

### Option 2 — Docker

Create a `Dockerfile` at the project root:

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run pipeline + train on container start, then launch dashboard
CMD python run_pipeline_full.py --no-scrape --force-train && \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

```bash
docker build -t nairobi-aq .
docker run -p 8501:8501 nairobi-aq
```

For production, add the scraper as a sidecar or cron job inside the container.

### Option 3 — Cloud VM (AWS EC2 / Azure / GCP)

```bash
# On the VM:
git clone https://github.com/mnkeing-marlon/Nairobi.git
cd Nairobi
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Initial data load
python scraper.py --once
python run_pipeline_full.py --no-scrape --force-train

# Set up cron for recurring pipeline
crontab -e
# */30 * * * * cd /home/user/Nairobi && /home/user/Nairobi/venv/bin/python run_pipeline_full.py >> logs/cron.log 2>&1

# Run dashboard (background)
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
```

Expose port 8501 via security group / firewall. Add a reverse proxy (nginx / Caddy) for HTTPS.

### Option 4 — GitHub Actions (automated pipeline + Streamlit Cloud)

Create `.github/workflows/pipeline.yml`:

```yaml
name: Data Pipeline
on:
  schedule:
    - cron: '0 */6 * * *'   # every 6 hours
  workflow_dispatch:          # manual trigger

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - run: pip install -r requirements.txt
      - run: python scraper.py --once
      - run: python run_pipeline_full.py --no-scrape --force-train

      - name: Commit updated data & model
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add data/processed/ models/
          git diff --cached --quiet || git commit -m "Auto-update data & model"
          git push
```

This keeps processed data and model files up to date in the repo. Streamlit Cloud will automatically redeploy when the commit lands.

---

## Project Structure

```
Nairobi/
├── app.py                    # Main Streamlit dashboard
├── pages/
│   ├── 01_Exploration.py     # Descriptive stats, distributions, correlations
│   └── 02_Prediction.py      # Model performance, feature importance
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # Raw data -> processed CSVs pipeline
│   ├── processor.py          # Data loading, AQI, KPIs, heatmap
│   └── model.py              # Prophet training + recursive prediction
├── scraper.py                # CKAN API scraper (downloads raw CSVs)
├── train_model.py            # CLI for model training
├── run_pipeline_full.py      # Orchestrator: scrape -> process -> train
├── setup_scheduler.ps1       # Windows Task Scheduler setup
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme & server config
├── .gitignore
├── data/
│   ├── raw_data/             # Downloaded sensor CSVs (gitignored)
│   └── processed/            # Per-location processed CSVs (gitignored)
│       ├── location_3966.csv
│       ├── location_3981.csv
│       └── _locations.json   # Manifest of available locations
├── models/
│   ├── prophet_pm25.joblib   # Trained Prophet model (gitignored)
│   └── metrics.joblib        # Evaluation metrics (gitignored)
├── logs/                     # Scraper logs (gitignored)
└── notebooks/                # Exploratory Jupyter notebooks (reference only)
```

---

## Environment Variables & Config

| Setting | Location | Default |
|---------|----------|---------|
| Streamlit theme | `.streamlit/config.toml` | Blue / white |
| Server port | `.streamlit/config.toml` | 8501 |
| Scrape interval | `scraper.py` | 1200 s (20 min) |
| Min hours per location | `src/pipeline.py` | 1000 |
| Date cutoff | `src/pipeline.py` | 2024-01-01 |
| Training location | `src/model.py` | 3966 (Kibera) |
| Cache TTL | `app.py` | 1800 s (30 min) |

No `.env` file is required. All configuration lives in in-code constants.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: prophet` | `pip install prophet` — may need C++ build tools on Windows |
| Prophet CmdStan won't compile | Use conda: `conda install -c conda-forge prophet` |
| Empty dashboard on first run | Run `python scraper.py --once` then `python run_pipeline_full.py --no-scrape --force-train` |
| No locations detected by pipeline | Ensure raw CSVs contain 2024+ data with at least 1 000 hourly records per location |
| `FileNotFoundError: data/processed/...` | Run the pipeline: `python run_pipeline_full.py --no-scrape` |
| Scraper returns all 404s | openAFRICA may be temporarily down; check `logs/scraper.log` |
| Dashboard cache is stale | Sidebar hamburger menu -> "Clear cache", or restart |
| Port 8501 already in use | `streamlit run app.py --server.port=8502` |

---

## License

This project is for educational and research purposes.
