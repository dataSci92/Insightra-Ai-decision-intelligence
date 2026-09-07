import pandas as pd
from analytics import profile_data, detect_columns

def test_profile():
    df = pd.DataFrame({"revenue":[10,20], "customer":["a","b"]})
    p = profile_data(df)
    assert p["rows"] == 2
    assert p["numeric_columns"] == 1

def test_detect_columns():
    df = pd.DataFrame({"order_date":["2026-01-01"], "revenue":[100], "churn":[0]})
    assert detect_columns(df)["date"] == "order_date"
    assert detect_columns(df)["revenue"] == "revenue"
    assert detect_columns(df)["churn"] == "churn"
