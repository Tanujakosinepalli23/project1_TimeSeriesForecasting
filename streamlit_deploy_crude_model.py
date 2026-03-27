import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Crude Oil Analytics",
    page_icon="⛽",
    layout="wide"
)

# -------------------------
# PREMIUM CSS DESIGN
# -------------------------
st.markdown("""
<style>
/* Background */
body {
    background-color: #0f172a;
}

/* Main Title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #f8fafc;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 25px;
}

/* KPI Cards */
.card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    text-align: center;
}

.card h3 {
    color: #94a3b8;
    font-size: 14px;
}

.card h1 {
    color: #38bdf8;
    font-size: 32px;
}

/* Section Titles */
.section {
    font-size: 22px;
    color: #f1f5f9;
    margin-top: 30px;
    margin-bottom: 10px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="main-title">⛽ Crude Oil Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Time Series Forecasting & Analytics</div>', unsafe_allow_html=True)

# -------------------------
# Load Data
# -------------------------
DATA_PATH = Path("Crude oil.csv")

if not DATA_PATH.exists():
    st.error("Dataset not found. Upload 'Crude oil.csv'")
    st.stop()

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

series = df["Close/Last"].astype(float).ffill().bfill()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("⚙️ Controls")

lags = st.sidebar.slider("Lag Window", 1, 20, 5)
split_ratio = st.sidebar.slider("Training Size (%)", 60, 90, 80)
horizon = st.sidebar.slider("Forecast Days", 5, 60, 30)

# -------------------------
# SPLIT
# -------------------------
split_idx = int(len(series) * split_ratio / 100)
train = series[:split_idx]
test = series[split_idx:]

# -------------------------
# MODEL
# -------------------------
def predict_ar(train_data, test_data, lags):
    history = list(train_data)
    preds = []

    for t in range(len(test_data)):
        yhat = np.mean(history[-lags:]) if len(history) >= lags else np.mean(history)
        preds.append(yhat)
        history.append(test_data.iloc[t])

    return pd.Series(preds, index=test_data.index)

test_preds = predict_ar(train, test, lags)

# -------------------------
# METRICS
# -------------------------
mae = np.mean(np.abs(test - test_preds))
rmse = np.sqrt(np.mean((test - test_preds) ** 2))

# -------------------------
# KPI SECTION
# -------------------------
col1, col2, col3 = st.columns(3)

col1.markdown(f'<div class="card"><h3>MAE</h3><h1>{mae:.2f}</h1></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card"><h3>RMSE</h3><h1>{rmse:.2f}</h1></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card"><h3>Data Points</h3><h1>{len(series)}</h1></div>', unsafe_allow_html=True)

# -------------------------
# PERFORMANCE CHART
# -------------------------
st.markdown('<div class="section">📊 Model Performance</div>', unsafe_allow_html=True)

chart_df = pd.DataFrame({
    "Train": train,
    "Actual": test,
    "Predicted": test_preds
})

st.line_chart(chart_df, height=420)

# -------------------------
# FORECAST
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
# FORECAST CHART
# -------------------------
st.markdown('<div class="section">🔮 Future Forecast</div>', unsafe_allow_html=True)

forecast_df = pd.DataFrame({
    "Recent": series[-200:],
    "Forecast": forecast_series
})

st.line_chart(forecast_df, height=420)

# -------------------------
# TABLE
# -------------------------
st.markdown('<div class="section">📅 Forecast Data</div>', unsafe_allow_html=True)

st.dataframe(
    forecast_series.reset_index().rename(columns={"index":"Date",0:"Forecast"}),
    use_container_width=True
)

# -------------------------
# DOWNLOAD
# -------------------------
csv = forecast_series.to_csv().encode("utf-8")
st.download_button("⬇️ Download Forecast", csv, "forecast.csv", "text/csv")
