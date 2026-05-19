# ============================================
# EMPLOYEE CLUSTERING (AGE vs TENURE)
# ============================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(
    "data/processed/cleaned_employees.csv"
)

# ============================================
# FEATURE MATRIX
# ============================================

X = df[["Age", "Tenure_Months"]].fillna(0)

# ============================================
# SCALING
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# K-MEANS CLUSTERING
# ============================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ============================================
# PLOT
# ============================================

os.makedirs("data/outputs/charts", exist_ok=True)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Tenure_Months",
    y="Age",
    hue="Cluster"
)

plt.title("Employee Clusters (Age vs Tenure)")
plt.tight_layout()

plt.savefig(
    "data/outputs/charts/clusters.png",
    dpi=150
)

plt.show()
