import numpy as np
from analytics import detect_columns

def simulate_revenue(df, discount, retention_lift, marketing_change):
    cols = detect_columns(df)
    if cols["revenue"]:
        baseline = float(np.nansum(df[cols["revenue"]]))
    else:
        baseline = 100000.0

    # Directional portfolio demo model; assumptions are intentionally transparent.
    volume_lift = 1 + retention_lift / 100
    marketing_lift = 1 + max(-0.5, marketing_change / 100) * 0.25
    discount_effect = 1 - discount / 100 * 0.35
    estimated = baseline * volume_lift * marketing_lift * discount_effect
    net_impact = estimated - baseline

    assumptions = (
        "Illustrative scenario model: retention lift increases realized revenue, "
        "marketing changes have a dampened effect, and discounts reduce realized revenue. "
        "Replace these assumptions with calibrated causal/experimental estimates for production use."
    )
    return {"baseline": baseline, "estimated": estimated, "net_impact": net_impact, "assumptions": assumptions}
