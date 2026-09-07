import numpy as np
import pandas as pd
from analytics import detect_columns

def generate_recommendations(df):
    recs = []
    cols = detect_columns(df)

    if cols["revenue"]:
        s = pd.to_numeric(df[cols["revenue"]], errors="coerce").dropna()
        if len(s) >= 10:
            recent = s.tail(max(3, len(s)//5)).mean()
            earlier = s.head(max(3, len(s)//5)).mean()
            if earlier and recent < earlier * .9:
                recs.append({
                    "priority": "HIGH", "title": "Investigate revenue slowdown",
                    "reason": f"Recent average {cols['revenue']} is approximately {abs((recent/earlier-1)*100):.1f}% lower than the early-period average.",
                    "action": "Segment the decline by product, channel, region, and customer cohort before changing pricing."
                })
            elif recent > earlier * 1.1:
                recs.append({
                    "priority": "HIGH", "title": "Protect recent growth",
                    "reason": f"Recent average {cols['revenue']} is approximately {(recent/earlier-1)*100:.1f}% above the early-period average.",
                    "action": "Identify the segments driving growth and test whether the pattern is repeatable."
                })

    churn = cols["churn"]
    if churn:
        y = df[churn].astype(str).str.lower().map({"yes":1,"true":1,"churn":1,"no":0,"false":0,"active":0})
        if y.notna().mean() > .7 and y.mean() > .2:
            recs.append({
                "priority": "HIGH", "title": "Prioritize retention",
                "reason": f"Observed churn rate is approximately {y.mean()*100:.1f}%.",
                "action": "Target high-value at-risk customers with measured retention experiments."
            })

    missing = df.isna().mean().sort_values(ascending=False)
    if len(missing) and missing.iloc[0] > .1:
        recs.append({
            "priority": "MEDIUM", "title": "Fix data quality",
            "reason": f"{missing.index[0]} has {missing.iloc[0]*100:.1f}% missing values.",
            "action": "Define a source-system validation rule and an explicit imputation policy."
        })

    if not recs:
        recs.append({
            "priority": "LOW", "title": "Explore segment-level drivers",
            "reason": "No strong automatic signal crossed the current heuristic thresholds.",
            "action": "Compare revenue, volume, and customer behavior across the most important categorical segments."
        })
    return recs
