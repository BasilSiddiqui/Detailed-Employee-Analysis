# ============================================
# TURNOVER FORECAST USING PROPHET
# ============================================

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ============================================
# LOAD DATA
# ============================================

turnover = pd.read_csv(
    "data/outputs/turnover_monthly.csv"
)

# Prophet requires columns: ds (date), y (value)

turnover.columns = ["ds", "y"]

turnover["ds"] = pd.to_datetime(turnover["ds"])

# Ensure numeric
turnover["y"] = turnover["y"].astype(float)

# ============================================
# MODEL
# ============================================

model = Prophet()

model.fit(turnover)

# ============================================
# FUTURE DATAFRAME
# ============================================

future = model.make_future_dataframe(
    periods=12,
    freq="M"
)

# ============================================
# FORECAST
# ============================================

forecast = model.predict(future)

# ============================================
# PLOT
# ============================================

fig = model.plot(forecast)

plt.savefig(
    "data/outputs/charts/turnover_forecast.png",
    dpi=150
)

plt.show()
