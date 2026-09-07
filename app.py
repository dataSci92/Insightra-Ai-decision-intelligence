import streamlit as st
import pandas as pd
import plotly.express as px

from analytics import profile_data, detect_columns, numeric_summary, quality_report
from ml import train_churn_model, forecast_revenue
from recommendations import generate_recommendations
from simulator import simulate_revenue
from analyst import answer_question

st.set_page_config(page_title="INSIGHTRA", page_icon="📊", layout="wide")

st.title("INSIGHTRA")
st.caption("AI Business Intelligence & Decision Engine — From Data to Smarter Decisions")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload business CSV", type=["csv"])
    st.divider()
    st.info("A demo dataset is used automatically when no file is uploaded.")

@st.cache_data
def load_demo():
    return pd.read_csv("sample_business.csv")

df = pd.read_csv(uploaded) if uploaded else load_demo()

st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns")

tabs = st.tabs(["Overview", "Predictions", "Explain", "Recommendations", "What-if", "Ask INSIGHTRA"])

with tabs[0]:
    st.subheader("Business overview")
    p = profile_data(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{p['rows']:,}")
    c2.metric("Columns", f"{p['columns']:,}")
    c3.metric("Missing cells", f"{p['missing_cells']:,}")
    c4.metric("Numeric fields", f"{p['numeric_columns']:,}")

    st.subheader("Data quality")
    st.dataframe(quality_report(df), use_container_width=True, hide_index=True)

    st.subheader("Numeric summary")
    st.dataframe(numeric_summary(df), use_container_width=True)

    cols = detect_columns(df)
    if cols["date"] and cols["revenue"]:
        temp = df.copy()
        temp[cols["date"]] = pd.to_datetime(temp[cols["date"]], errors="coerce")
        temp = temp.dropna(subset=[cols["date"], cols["revenue"]])
        daily = temp.groupby(cols["date"])[cols["revenue"]].sum().reset_index()
        fig = px.line(daily, x=cols["date"], y=cols["revenue"], title="Revenue trend")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Predictive intelligence")
    churn = train_churn_model(df)
    if churn["ok"]:
        st.metric("Model ROC-AUC", f"{churn['auc']:.3f}")
        st.dataframe(churn["scored"].head(50), use_container_width=True, hide_index=True)
    else:
        st.warning(churn["message"])

    forecast = forecast_revenue(df)
    if forecast["ok"]:
        st.subheader("Revenue forecast")
        st.dataframe(forecast["forecast"], use_container_width=True, hide_index=True)
        st.plotly_chart(forecast["figure"], use_container_width=True)
    else:
        st.info(forecast["message"])

with tabs[2]:
    st.subheader("Explainable AI")
    churn = train_churn_model(df, explain=True)
    if churn["ok"] and churn.get("explanations") is not None:
        st.write("Top model drivers for the current dataset:")
        st.dataframe(churn["explanations"], use_container_width=True, hide_index=True)
    elif churn["ok"]:
        st.info("SHAP explanations were not available for this model run.")
    else:
        st.warning(churn["message"])

with tabs[3]:
    st.subheader("Recommended actions")
    recs = generate_recommendations(df)
    for r in recs:
        st.markdown(f"### {r['priority']} — {r['title']}")
        st.write(r["reason"])
        st.caption(f"Action: {r['action']}")

with tabs[4]:
    st.subheader("What-if revenue simulator")
    st.write("Adjust the assumptions to estimate directional business impact.")
    discount = st.slider("Discount (%)", 0, 30, 10)
    retention = st.slider("Expected retention lift (%)", 0, 30, 5)
    marketing = st.slider("Marketing spend change (%)", -50, 100, 10)
    result = simulate_revenue(df, discount, retention, marketing)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline revenue", f"${result['baseline']:,.0f}")
    c2.metric("Estimated revenue", f"${result['estimated']:,.0f}")
    c3.metric("Estimated net impact", f"${result['net_impact']:,.0f}")
    st.caption(result["assumptions"])

with tabs[5]:
    st.subheader("Ask INSIGHTRA")
    question = st.text_input("Ask a question about the uploaded data",
                             placeholder="Why might revenue be declining?")
    if question:
        answer = answer_question(df, question)
        st.markdown(answer)
