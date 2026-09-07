import pandas as pd
import numpy as np

def profile_data(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": int(df.select_dtypes(include=np.number).shape[1]),
    }

def detect_columns(df):
    date_col = None
    revenue_col = None
    churn_col = None

    for c in df.columns:
        low = c.lower()
        if date_col is None and any(x in low for x in ["date", "time", "month"]):
            date_col = c
        if revenue_col is None and any(x in low for x in ["revenue", "sales", "amount", "income"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                revenue_col = c
        if churn_col is None and any(x in low for x in ["churn", "attrition"]):
            churn_col = c

    return {"date": date_col, "revenue": revenue_col, "churn": churn_col}

def quality_report(df):
    out = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [round(float(df[c].isna().mean()*100), 2) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    return out

def numeric_summary(df):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return pd.DataFrame({"message": ["No numeric columns found."]})
    return num.describe().T.round(2).reset_index().rename(columns={"index": "column"})
