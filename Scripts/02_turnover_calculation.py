import pandas as pd

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv(
    "data/processed/cleaned_employees.csv"
)

# -----------------------------
# Ensure datetime types
# -----------------------------
date_columns = [
    "ORIG_HIRE_DT",
    "ACTUAL_TERMINATION_DATE",
    "EndDate"
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# -----------------------------
# Monthly employee presence
# -----------------------------
df["MonthRange"] = df.apply(
    lambda row: pd.date_range(
        start=row["ORIG_HIRE_DT"],
        end=row["EndDate"],
        freq="ME"   # Month End frequency
    ),
    axis=1
)

df_monthly = df.explode("MonthRange")

df_monthly["Month"] = (
    df_monthly["MonthRange"]
    .dt.to_period("M")
)

# ============================================
# MONTHLY TURNOVER
# ============================================

monthly_hc = (
    df_monthly.groupby("Month")["PERSON_NUMBER"]
    .nunique()
)

df["ExitMonth"] = (
    df["ACTUAL_TERMINATION_DATE"]
    .dt.to_period("M")
)

monthly_leavers = (
    df.groupby("ExitMonth")["PERSON_NUMBER"]
    .nunique()
)

turnover_monthly = (
    monthly_leavers / monthly_hc
).dropna()

turnover_monthly.to_csv(
    "data/outputs/turnover_monthly.csv"
)

# ============================================
# QUARTERLY TURNOVER
# ============================================

df_monthly["Quarter"] = (
    df_monthly["Month"]
    .dt.to_timestamp()
    .dt.to_period("Q")
)

quarterly_hc = (
    df_monthly.groupby("Quarter")["PERSON_NUMBER"]
    .nunique()
)

df["ExitQuarter"] = (
    df["ACTUAL_TERMINATION_DATE"]
    .dt.to_period("Q")
)

quarterly_leavers = (
    df.groupby("ExitQuarter")["PERSON_NUMBER"]
    .nunique()
)

turnover_quarterly = (
    quarterly_leavers / quarterly_hc
).dropna()

turnover_quarterly.to_csv(
    "data/outputs/turnover_quarterly.csv"
)

# ============================================
# YEARLY TURNOVER
# ============================================

df_monthly["Year"] = (
    df_monthly["Month"]
    .dt.to_timestamp()
    .dt.to_period("Y")
)

yearly_hc = (
    df_monthly.groupby("Year")["PERSON_NUMBER"]
    .nunique()
)

df["ExitYear"] = (
    df["ACTUAL_TERMINATION_DATE"]
    .dt.to_period("Y")
)

yearly_leavers = (
    df.groupby("ExitYear")["PERSON_NUMBER"]
    .nunique()
)

turnover_yearly = (
    yearly_leavers / yearly_hc
).dropna()

turnover_yearly.to_csv(
    "data/outputs/turnover_yearly.csv"
)

print(
    "Monthly, Quarterly, and Yearly turnover files created."
)
