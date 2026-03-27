import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Crude Oil Forecast",
    page_icon="⛽",
    layout="wide"
)

# -------------------------
# Custom CSS (UI Styling)
# -------------------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #FF6B35;
}
.sub-text {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.metric-card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title Section
# -------------------------
st.markdown('<div class="main-title">⛽ Crude Oil Price Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Time Series Forecasting using Auto Regression</div>', unsafe_allow_html=True)

# -------------------------
# Load Dataset
# -------------------------
DATA_PATH = Path("Crude oil.csv")

if not DATA_PATH.exists():
    st.error("❌ Please upload 'Crude oil.csv'")
    st.stop()

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

series = df["Close/Last"].astype(float).ffill().bfill()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚙️ Controls")

lags = st.sidebar.slider("Lag Window", 1, 20, 5)
split_ratio = st.sidebar.slider("Train Size (%)", 60, 90, 80)
horizon = st.sidebar.slider("Forecast Days", 5, 60, 30)

# -------------------------
# Split Data
# -------------------------
split_idx = int(len(series) * split_ratio / 100)
train = series[:split_idx]
test = series[split_idx:]

# -------------------------
# Simple AR Model
# -------------------------
def predict_ar(train_data, test_data, lags):
    history = list(train_data)
    predictions = []

    for t in range(len(test_data)):
        yhat = np.mean(history[-lags:]) if len(history) >= lags else np.mean(history)
        predictions.append(yhat)
        history.append(test_data.iloc[t])

    return pd.Series(predictions, index=test_data.index)

test_preds = predict_ar(train, test, lags)

# -------------------------
# Metrics
# -------------------------
mae = np.mean(np.abs(test - test_preds))
rmse = np.sqrt(np.mean((test - test_preds) ** 2))

# -------------------------
# KPI Cards
# -------------------------
col1, col2 = st.columns(2)

col1.markdown(f'<div class="metric-card"><h3>MAE</h3><h2>{mae:.2f}</h2></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><h3>RMSE</h3><h2>{rmse:.2f}</h2></div>', unsafe_allow_html=True)

# -------------------------
# Charts Section
# -------------------------
st.markdown("### 📈 Model Performance")

chart_df = pd.DataFrame({
    "Train": train,
    "Actual": test,
    "Predicted": test_preds
})

st.line_chart(chart_df, height=400)

# -------------------------
# Forecast
# -------------------------
history = list(series)
future_preds = []

for _ in range(horizon):
    yhat = np.mean(history[-lags:]) if len(history) >= lags else np.mean(history)
    future_preds.append(yhat)
    history.append(yhat)

future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
forecast_series = pd.Series(future_preds, index=future_index)

# -------------------------
# Forecast Chart
# -------------------------
st.markdown("### 🔮 Future Forecast")

forecast_df = pd.DataFrame({
    "Recent Data": series[-200:],
    "Forecast": forecast_series
})

st.line_chart(forecast_df, height=400)

# -------------------------
# Table + Download
# -------------------------
st.markdown("### 📅 Forecast Data")

st.dataframe(forecast_series.reset_index().rename(columns={"index":"Date",0:"Forecast"}))

csv = forecast_series.to_csv().encode("utf-8")
st.download_button("⬇️ Download Forecast", csv, "forecast.csv", "text/csv")
