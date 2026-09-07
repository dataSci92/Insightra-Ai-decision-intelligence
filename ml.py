import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from analytics import detect_columns

def _find_target(df):
    cols = detect_columns(df)
    if cols["churn"]:
        return cols["churn"]
    for c in df.columns:
        low = c.lower()
        if low in {"churned", "is_churned", "churn_flag"}:
            return c
    return None

def train_churn_model(df, explain=False):
    target = _find_target(df)
    if not target:
        return {"ok": False, "message": "No churn/attrition target column found. Add a binary churn column to enable this model."}

    y = df[target]
    if y.dtype == "object":
        vals = {str(v).strip().lower() for v in y.dropna().unique()}
        mapping = {"yes":1,"no":0,"true":1,"false":0,"churn":1,"active":0}
        y = y.astype(str).str.strip().str.lower().map(mapping)
    else:
        y = pd.to_numeric(y, errors="coerce")

    valid = y.isin([0,1])
    X = df.loc[valid].drop(columns=[target]).copy()
    y = y.loc[valid].astype(int)

    if len(y) < 40 or y.nunique() < 2:
        return {"ok": False, "message": "Not enough valid binary churn records to train a reliable demo model."}

    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1], errors="ignore")
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("prep", prep), ("clf", RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    ))])

    strat = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=strat)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba) if yte.nunique() == 2 else float("nan")

    scored = Xte.copy()
    scored["churn_probability"] = proba
    scored = scored.sort_values("churn_probability", ascending=False)

    explanations = None
    if explain:
        try:
            import shap
            transformed = model.named_steps["prep"].transform(Xte)
            clf = model.named_steps["clf"]
            explainer = shap.TreeExplainer(clf)
            values = explainer.shap_values(transformed)
            arr = values[1] if isinstance(values, list) else values
            names = model.named_steps["prep"].get_feature_names_out()
            imp = np.abs(arr).mean(axis=0)
            explanations = pd.DataFrame({"feature": names, "mean_abs_shap": imp}).sort_values(
                "mean_abs_shap", ascending=False
            ).head(15)
        except Exception:
            explanations = None

    return {"ok": True, "auc": auc, "scored": scored, "explanations": explanations}

def forecast_revenue(df, periods=6):
    cols = detect_columns(df)
    if not cols["date"] or not cols["revenue"]:
        return {"ok": False, "message": "A date/time column and a numeric revenue/sales column are required for forecasting."}

    t = df[[cols["date"], cols["revenue"]]].copy()
    t[cols["date"]] = pd.to_datetime(t[cols["date"]], errors="coerce")
    t[cols["revenue"]] = pd.to_numeric(t[cols["revenue"]], errors="coerce")
    t = t.dropna().sort_values(cols["date"])
    if len(t) < 12:
        return {"ok": False, "message": "At least 12 time observations are recommended for the demo forecast."}

    monthly = t.set_index(cols["date"])[cols["revenue"]].resample("MS").sum().reset_index()
    if len(monthly) < 8:
        return {"ok": False, "message": "Not enough distinct time periods for a stable demo forecast."}

    x = np.arange(len(monthly)).reshape(-1,1)
    y = monthly[cols["revenue"]].values
    model = LinearRegression().fit(x, y)
    future_x = np.arange(len(monthly), len(monthly)+periods).reshape(-1,1)
    future_dates = pd.date_range(monthly[cols["date"]].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    pred = np.maximum(model.predict(future_x), 0)
    forecast = pd.DataFrame({cols["date"]: future_dates, "forecast_revenue": pred.round(2)})

    fig = px.line(pd.concat([
        monthly.rename(columns={cols["revenue"]: "revenue"})[[cols["date"], "revenue"]],
        forecast.rename(columns={"forecast_revenue": "revenue"})
    ]), x=cols["date"], y="revenue", title="Historical + forecast revenue")
    return {"ok": True, "forecast": forecast, "figure": fig}
