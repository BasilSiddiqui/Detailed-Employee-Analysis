# ============================================
# ACTIVE EMPLOYEE EXIT RISK SCORING
# ============================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv(
    "data/processed/cleaned_employees.csv"
)

# ============================================
# 2. REBUILD FINAL MODEL (TRAINING STEP)
# ============================================

y = df["Exited"]

X_train = df[
    [
        "Age",
        "Tenure_Months",
        "BAND",
        "CONTRACT_TYPE"
    ]
]

# Encode categorical variables
X_train = pd.get_dummies(
    X_train,
    drop_first=True
)

X_train = X_train.astype(float)

# Remove missing values
mask_train = X_train.notna().all(axis=1)

X_train = X_train.loc[mask_train]
y_train = y.loc[mask_train]

# Add constant
X_train_const = sm.add_constant(
    X_train,
    has_constant="add"
)

# Fit logistic regression
model_final = sm.Logit(
    y_train,
    X_train_const
).fit(disp=False)

# Store training feature names (CRITICAL)
model_features = X_train_const.columns

# ============================================
# 3. FILTER ACTIVE EMPLOYEES
# ============================================

df_active = df[df["Exited"] == 0].copy()

# ============================================
# 4. PREPARE SCORING DATA
# ============================================

X_score = df_active[
    [
        "Age",
        "Tenure_Months",
        "BAND",
        "CONTRACT_TYPE"
    ]
]

X_score = pd.get_dummies(
    X_score,
    drop_first=True
)

X_score = X_score.astype(float)

# ============================================
# 5. ALIGN COLUMNS TO TRAINING SET
# ============================================

# ensure same structure as training
feature_cols = [
    c for c in model_features
    if c != "const"
]

for col in feature_cols:
    if col not in X_score.columns:
        X_score[col] = 0

X_score = X_score[feature_cols]

X_score_const = sm.add_constant(
    X_score,
    has_constant="add"
)

# ============================================
# 6. GENERATE PREDICTIONS
# ============================================

df_active["Predicted_Exit_Probability"] = model_final.predict(
    X_score_const
)

df_active["Predicted_Exit_Odds"] = (
    df_active["Predicted_Exit_Probability"]
    / (1 - df_active["Predicted_Exit_Probability"])
)

# ============================================
# 7. RISK BANDING
# ============================================

df_active["Exit_Risk_Band"] = pd.cut(
    df_active["Predicted_Exit_Probability"],
    bins=[0, 0.20, 0.40, 1.00],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

# ============================================
# 8. SAVE OUTPUT
# ============================================

output_path = "data/outputs/active_employee_exit_risk.csv"

df_active.to_csv(output_path, index=False)

print("Exit risk scoring complete.")
print(f"Results saved to: {output_path}")
