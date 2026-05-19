# ============================================
# FEATURE SIGNIFICANCE ANALYSIS & VISUALS
# ============================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

# ============================================
# CONFIG
# ============================================

SAVE_CSV = False   # Change to True if needed

output_dir = "data/outputs/feature_significance"
os.makedirs(output_dir, exist_ok=True)

# ============================================
# LOAD CLEANED DATA
# ============================================

df = pd.read_csv(
    "data/processed/cleaned_employees.csv"
)

# ============================================
# AGGREGATIONS
# ============================================

# Exit Rate by Job Level
exit_by_job = (
    df.groupby("JOB_LEVEL")["Exited"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

# Exit Rate by Band
exit_by_band = (
    df.groupby("BAND")["Exited"]
    .mean()
    .reset_index()
)

# Exit Rate by Contract Type
exit_by_contract = (
    df.groupby("CONTRACT_TYPE")["Exited"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

# Exit Rate by Gender
exit_by_gender = (
    df.groupby("GENDER")["Exited"]
    .mean()
    .reset_index()
)

# ============================================
# CORRELATION MATRIX
# ============================================

corr_vars = [
    "Exited",
    "Age",
    "Tenure_Months",
    "BAND"
]

corr = df[corr_vars].corr()

# ============================================
# LOGISTIC REGRESSION ODDS RATIOS
# ============================================

odds_ratios = pd.DataFrame({
    "Feature": [
        "Age",
        "Tenure_Months",
        "BAND"
    ],
    "Odds Ratio": [
        1.061030,
        0.994640,
        0.771765
    ]
})

# ============================================
# OPTIONAL CSV EXPORTS
# ============================================

if SAVE_CSV:

    exit_by_job.to_csv(
        f"{output_dir}/exit_rate_by_job_level.csv",
        index=False
    )

    exit_by_band.to_csv(
        f"{output_dir}/exit_rate_by_band.csv",
        index=False
    )

    exit_by_contract.to_csv(
        f"{output_dir}/exit_rate_by_contract.csv",
        index=False
    )

    exit_by_gender.to_csv(
        f"{output_dir}/exit_rate_by_gender.csv",
        index=False
    )

    corr.to_csv(
        f"{output_dir}/correlation_with_exited.csv"
    )

    odds_ratios.to_csv(
        f"{output_dir}/logistic_odds_ratios.csv",
        index=False
    )

# ============================================
# VISUAL THEME
# ============================================

PRIMARY_BLUE = "#515488"
ACCENT_RED = "#FF2261"
NEUTRAL_GREY = "#6E6E72"
BG_WHITE = "#FFFFFF"

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.facecolor": BG_WHITE,
        "figure.facecolor": BG_WHITE,
        "grid.color": "#E6E6EE",
        "axes.edgecolor": "#E6E6EE",
        "axes.labelcolor": NEUTRAL_GREY,
        "text.color": "#1A1A1A",
        "xtick.color": NEUTRAL_GREY,
        "ytick.color": NEUTRAL_GREY,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11
    }
)

# ============================================
# EXIT RATE BY JOB LEVEL
# ============================================

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=exit_by_job,
    x="JOB_LEVEL",
    y="Exited",
    color=PRIMARY_BLUE
)

ax.yaxis.set_major_formatter(
    mtick.PercentFormatter(1.0)
)

plt.title("Exit Rate by Job Level")
plt.ylabel("Exit Rate")
plt.xlabel("")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/exit_rate_by_job_level.png",
    dpi=150
)

plt.close()

# ============================================
# EXIT RATE BY BAND
# ============================================

plt.figure(figsize=(6, 4))

ax = sns.barplot(
    data=exit_by_band,
    x="BAND",
    y="Exited",
    color=PRIMARY_BLUE
)

ax.yaxis.set_major_formatter(
    mtick.PercentFormatter(1.0, decimals=0)
)

plt.title("Exit Rate by Band")
plt.ylabel("Exit Rate")
plt.xlabel("")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/exit_rate_by_band.png",
    dpi=150
)

plt.close()

# ============================================
# EXIT RATE BY CONTRACT TYPE
# ============================================

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=exit_by_contract,
    x="CONTRACT_TYPE",
    y="Exited",
    palette=[
        ACCENT_RED,
        PRIMARY_BLUE,
        NEUTRAL_GREY
    ]
)

ax.yaxis.set_major_formatter(
    mtick.PercentFormatter(1.0)
)

plt.title("Exit Rate by Contract Type")
plt.ylabel("Exit Rate")
plt.xlabel("")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/exit_rate_by_contract_type.png",
    dpi=150
)

plt.close()

# ============================================
# EXIT RATE BY GENDER
# ============================================

plt.figure(figsize=(5, 4))

ax = sns.barplot(
    data=exit_by_gender,
    x="GENDER",
    y="Exited",
    color=PRIMARY_BLUE
)

ax.yaxis.set_major_formatter(
    mtick.PercentFormatter(1.0, decimals=0)
)

plt.title("Exit Rate by Gender")
plt.ylabel("Exit Rate")
plt.xlabel("")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/exit_rate_by_gender.png",
    dpi=150
)

plt.close()

# ============================================
# CORRELATION HEATMAP
# ============================================

custom_cmap = LinearSegmentedColormap.from_list(
    "exit_corr",
    [
        ACCENT_RED,
        "#FFC90C",
        PRIMARY_BLUE
    ]
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    corr,
    annot=True,
    cmap=custom_cmap,
    center=0,
    linewidths=0.5,
    cbar=False
)

plt.title("Relationship Between Factors and Exit")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/correlation_heatmap.png",
    dpi=150
)

plt.close()

# ============================================
# LOGISTIC REGRESSION ODDS RATIOS
# ============================================

plt.figure(figsize=(6, 4))

sns.barplot(
    data=odds_ratios,
    x="Feature",
    y="Odds Ratio",
    palette=[
        ACCENT_RED,
        PRIMARY_BLUE,
        PRIMARY_BLUE
    ]
)

plt.axhline(
    1,
    linestyle="--",
    color=NEUTRAL_GREY
)

plt.title("Key Drivers of Exit Risk")
plt.ylabel("Odds Ratio")
plt.xlabel("")

plt.tight_layout()

plt.savefig(
    f"{output_dir}/logistic_odds_ratios.png",
    dpi=150
)

plt.close()

# ============================================
# DONE
# ============================================

print("Feature significance graphs generated.")
