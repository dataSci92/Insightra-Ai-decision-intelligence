import re
import pandas as pd
from analytics import detect_columns

def answer_question(df, question):
    q = question.lower()
    cols = detect_columns(df)

    if any(w in q for w in ["revenue", "sales", "income"]):
        if cols["revenue"]:
            s = pd.to_numeric(df[cols["revenue"]], errors="coerce").dropna()
            if len(s):
                return (
                    f"### Revenue analysis\n"
                    f"Total observed **{cols['revenue']}** is **${s.sum():,.0f}**. "
                    f"The median observation is **${s.median():,.0f}** and the mean is **${s.mean():,.0f}**."
                )
        return "I could not identify a numeric revenue/sales column in this dataset."

    if "missing" in q or "quality" in q:
        m = df.isna().mean().sort_values(ascending=False)
        worst = m.head(5)
        lines = [f"- **{idx}**: {val*100:.1f}% missing" for idx, val in worst.items() if val > 0]
        return "### Data quality\n" + ("\n".join(lines) if lines else "No missing values were detected.")

    if "churn" in q:
        if cols["churn"]:
            y = df[cols["churn"]].astype(str).str.lower().map({"yes":1,"true":1,"churn":1,"no":0,"false":0,"active":0})
            return f"### Churn\nObserved churn rate is approximately **{y.mean()*100:.1f}%** based on recognizable binary labels."
        return "No recognizable churn column was found."

    return (
        f"### Dataset summary\n"
        f"This dataset contains **{len(df):,} rows** and **{len(df.columns):,} columns**. "
        f"Available fields include: {', '.join(map(str, df.columns[:12]))}."
    )
