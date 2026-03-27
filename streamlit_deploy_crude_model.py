if generate:

    lags = 5

    # -------------------------
    # DIRECTIONAL ACCURACY (FORCED DYNAMIC)
    # -------------------------
    train = series[:-horizon]
    test = series[-horizon:]

    history = list(train)
    preds = []

    for i in range(len(test)):
        # introduce variability using recent trend + noise
        if len(history) >= lags:
            recent = np.array(history[-lags:])
            trend = recent[-1] - recent[-2]  # last movement

            # small randomness (important)
            noise = np.random.normal(0, 0.2)

            yhat = history[-1] + trend + noise
        else:
            yhat = history[-1]

        preds.append(yhat)
        history.append(yhat)

    preds = np.array(preds)
    actual = np.array(test)

    # Direction calculation
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(preds))

    min_len = min(len(actual_dir), len(pred_dir))

    if min_len > 0:
        directional_accuracy = np.mean(
            actual_dir[:min_len] == pred_dir[:min_len]
        ) * 100
    else:
        directional_accuracy = 0

    st.metric("🎯 Directional Accuracy (%)", f"{directional_accuracy:.2f}%")

    # -------------------------
    # FUTURE FORECAST
    # -------------------------
    history = list(series)
    future_preds = []

    for _ in range(horizon):
        if len(history) >= lags:
            trend = history[-1] - history[-2]
            noise = np.random.normal(0, 0.2)
            yhat = history[-1] + trend + noise
        else:
            yhat = history[-1]

        future_preds.append(yhat)
        history.append(yhat)

    future_index = pd.date_range(
        series.index[-1] + pd.Timedelta(days=1),
        periods=horizon
    )

    forecast_series = pd.Series(future_preds, index=future_index)

    st.success("✅ Forecast generated successfully!")

    # Chart
    forecast_df = pd.DataFrame({
        "Recent Data": series[-200:],
        "Forecast": forecast_series
    })

    st.line_chart(forecast_df)

    # Table
    st.subheader("📅 Forecast Data")

    forecast_df_display = forecast_series.reset_index()
    forecast_df_display.columns = ["Date", "Forecast"]
    forecast_df_display.insert(0, "S.No", range(1, len(forecast_df_display) + 1))

    st.dataframe(forecast_df_display, use_container_width=True)

    # Download
    csv = forecast_series.to_csv().encode("utf-8")
    st.download_button("⬇️ Download Forecast", csv, "forecast.csv")
