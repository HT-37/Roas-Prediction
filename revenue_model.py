import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

def revenue_prediction_ui(df, game_name):
    st.subheader(f"Revenue Prediction for {game_name}")

    # Filter for selected game
    game_df = df[df["product_name"] == game_name][["date", "total_revenue", "total_cost"]]
    game_df = game_df.groupby("date")[["revenue", "cost"]].sum().reset_index()

    # Prepare data for Prophet
    prophet_df = game_df.rename(columns={"date": "ds", "revenue": "y"})

    model = Prophet(
      changepoint_prior_scale = 0.1,
      seasonality_prior_scale = 0.1,
      daily_seasonality = True,
      weekly_seasonality = True,
      yearly_seasonality = True,
      seasonality_mode = 'multiplicative',
      changepoint_range = 0.9
      )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=90)  # next 90 days
    forecast = model.predict(future)

    # --- Chart 1: Prophet model forecast
    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)

    # --- Chart 2: Cumulative revenue vs cost
    planned_cost = st.number_input("Planned Daily Cost", min_value=0.0, value=100.0)
    forecast["cum_rev"] = forecast["yhat"].cumsum()
    forecast["cum_cost"] = (game_df["cost"].sum() + planned_cost * range(1, len(forecast)+1))

    st.line_chart(forecast[["cum_rev", "cum_cost"]])

    # --- Show forecast table
    st.subheader("Forecast Data")
    st.dataframe(forecast[["ds", "yhat", "cum_rev", "cum_cost"]].tail(30))

    # --- Find estimated break-even day
    breakeven = forecast[forecast["cum_rev"] >= forecast["cum_cost"]].head(1)
    if not breakeven.empty:
        st.success(f"Estimated Break-even Date: {breakeven['ds'].values[0]}")
    else:
        st.warning("Break-even not reached within forecast period.")
