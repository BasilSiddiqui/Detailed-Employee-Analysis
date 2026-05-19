# ============================================
# KAPLAN-MEIER SURVIVAL ANALYSIS
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(
    "data/processed/cleaned_employees.csv"
)

# Ensure correct types
df["Tenure_Days"] = pd.to_numeric(df["Tenure_Days"], errors="coerce")
df["Exited"] = pd.to_numeric(df["Exited"], errors="coerce")

# Remove missing values
df = df.dropna(subset=["Tenure_Days", "Exited"])

# ============================================
# MODEL
# ============================================

kmf = KaplanMeierFitter()

kmf.fit(
    durations=df["Tenure_Days"],
    event_observed=df["Exited"]
)

# ============================================
# PLOT
# ============================================

plt.figure(figsize=(8, 5))

kmf.plot_survival_function()

plt.title("Employee Tenure Survival Curve")
plt.xlabel("Tenure (Days)")
plt.ylabel("Survival Probability")

plt.tight_layout()

plt.savefig(
    "data/outputs/charts/survival_curve.png",
    dpi=150
)

plt.show()
