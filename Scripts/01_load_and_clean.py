import pandas as pd

# -----------------------------
# Load raw data
# -----------------------------
df = pd.read_excel(
    "data/raw/DPW Employee Details.xlsx"
)

# -----------------------------
# BAND cleaning (ordinal)
# -----------------------------
band_mapping = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "TM": 7   # Top Management
}

df["BAND"] = (
    df["BAND"]
    .astype(str)
    .str.strip()
    .map(band_mapping)
)

# -----------------------------
# Apply business filters
# -----------------------------
df_filtered = df[
    df["JOB_LEVEL"].notna() &
    (df["JOB_LEVEL"] != "Not Applicable") &
    df["BAND"].notna()
].copy()

# -----------------------------
# Force BAND to whole numbers
# -----------------------------
df_filtered["BAND"] = (
    df_filtered["BAND"]
    .astype("Int64")
)

# -----------------------------
# Date parsing
# -----------------------------
date_columns = [
    "ORIG_HIRE_DT",
    "ACTUAL_TERMINATION_DATE",
    "DOB"
]

for col in date_columns:
    df_filtered[col] = (
        pd.to_datetime(df_filtered[col])
        .dt.normalize()
    )

# -----------------------------
# Feature engineering
# -----------------------------
today = pd.Timestamp.today().normalize()

# Employee age
df_filtered["Age"] = (
    (today - df_filtered["DOB"]).dt.days / 365.25
)

# End date
df_filtered["EndDate"] = (
    df_filtered["ACTUAL_TERMINATION_DATE"]
    .fillna(today)
)

# Tenure
df_filtered["Tenure_Days"] = (
    df_filtered["EndDate"] -
    df_filtered["ORIG_HIRE_DT"]
).dt.days

df_filtered["Tenure_Months"] = (
    df_filtered["Tenure_Days"] / 30
)

# Exit flag
df_filtered["Exited"] = (
    df_filtered["ACTUAL_TERMINATION_DATE"]
    .notna()
    .astype(int)
)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df_filtered.to_csv(
    "data/processed/cleaned_employees.csv",
    index=False
)

print("Cleaned employee dataset created.")
