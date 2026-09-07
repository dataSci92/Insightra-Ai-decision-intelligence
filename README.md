# INSIGHTRA — AI Business Intelligence & Decision Engine

**Live app:** https://insightra-ai-decision-intelligence--workerst6.replit.app

INSIGHTRA turns business data into insights, predictions, explanations, recommendations, and what-if decisions.

## Features
- CSV upload with automatic schema inspection
- Data quality and descriptive analytics
- Interactive Plotly dashboard
- Customer churn prediction when suitable columns exist
- Revenue forecasting when time + numeric revenue data are available
- Explainable churn predictions with SHAP when available
- Business recommendations based on observed signals
- What-if revenue simulator
- AI-style analyst interface using deterministic analytical tools (no API key required)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

A demo dataset is included at `sample_business.csv`.

## Project structure

```text
app.py
analytics.py
ml.py
recommendations.py
simulator.py
analyst.py
sample_business.csv
test_analytics.py
requirements.txt
render.yaml
.env.example
.gitignore
```

Everything is intentionally kept in one flat folder, so the project still
runs correctly even after a plain drag-and-drop upload to GitHub (which does
not preserve subfolders).

## Notes
INSIGHTRA is designed as a portfolio-grade MVP. Models are only trained when the uploaded dataset contains compatible columns; otherwise the application explains what is missing instead of fabricating results.
