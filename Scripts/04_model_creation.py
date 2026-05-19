# ============================================
# LOGISTIC REGRESSION MODELS
# ============================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(
    "data/processed/cleaned_employees.csv"
)

# ============================================
# TARGET VARIABLE
# ============================================

y = df["Exited"]

# ============================================
# MODEL 1: FULL MODEL (WITH GENDER)
# ============================================

X_full = df[
    [
        "Age",
        "Tenure_Months",
        "BAND",
        "CONTRACT_TYPE",
        "GENDER"
    ]
]

# Convert categorical variables
X_full = pd.get_dummies(
    X_full,
    drop_first=True
)

# Ensure numeric format
X_full = X_full.astype(float)

# Remove missing rows
mask_full = X_full.notna().all(axis=1)

X_full = X_full.loc[mask_full]
y_full = y.loc[mask_full]

# Add intercept
X_full_const = sm.add_constant(X_full)

# Fit logistic regression
model_full = sm.Logit(
    y_full.values,
    X_full_const.values
).fit(disp=False)

# ============================================
# FULL MODEL SUMMARY
# ============================================

print("\n===================================")
print("FULL MODEL SUMMARY")
print("===================================")

print(model_full.summary())

# ============================================
# FULL MODEL PREDICTIONS
# ============================================

y_full_prob = model_full.predict(
    X_full_const.values
)

y_full_pred = (
    y_full_prob >= 0.5
).astype(int)

# ============================================
# FULL MODEL METRICS
# ============================================

roc_auc_full = roc_auc_score(
    y_full,
    y_full_prob
)

acc_full = accuracy_score(
    y_full,
    y_full_pred
)

# ============================================
# MODEL 2: FINAL MODEL
# (STRUCTURAL VARIABLES ONLY)
# ============================================

X_final = df[
    [
        "Age",
        "Tenure_Months",
        "BAND",
        "CONTRACT_TYPE"
    ]
]

# Convert categorical variables
X_final = pd.get_dummies(
    X_final,
    drop_first=True
)

# Ensure numeric format
X_final = X_final.astype(float)

# Remove missing rows
mask_final = X_final.notna().all(axis=1)

X_final = X_final.loc[mask_final]
y_final = y.loc[mask_final]

# Add intercept
X_final_const = sm.add_constant(X_final)

# Fit logistic regression
model_final = sm.Logit(
    y_final.values,
    X_final_const.values
).fit(disp=False)

# ============================================
# FINAL MODEL SUMMARY
# ============================================

print("\n===================================")
print("FINAL MODEL SUMMARY")
print("===================================")

print(model_final.summary())

# ============================================
# FINAL MODEL PREDICTIONS
# ============================================

y_final_prob = model_final.predict(
    X_final_const.values
)

y_final_pred = (
    y_final_prob >= 0.5
).astype(int)

# ============================================
# FINAL MODEL METRICS
# ============================================

roc_auc_final = roc_auc_score(
    y_final,
    y_final_prob
)

acc_final = accuracy_score(
    y_final,
    y_final_pred
)

# ============================================
# MODEL COMPARISON
# ============================================

print("\n===================================")
print("MODEL COMPARISON METRICS")
print("===================================")

comparison = pd.DataFrame({
    "Metric": [
        "ROC AUC",
        "Accuracy",
        "AIC",
        "BIC"
    ],

    "Full Model (with Gender)": [
        round(roc_auc_full, 3),
        round(acc_full, 3),
        round(model_full.aic, 2),
        round(model_full.bic, 2)
    ],

    "Final Model (Structural)": [
        round(roc_auc_final, 3),
        round(acc_final, 3),
        round(model_final.aic, 2),
        round(model_final.bic, 2)
    ]
})

print(comparison)
