import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Utility Functions ===
@st.cache_data
def preprocess(df):
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)
    return df[df['Users'] >= 50]

def load_model(path):
    return joblib.load(path)

# === Model Paths ===
MODEL_PATHS = {
    # ROAS direct
    "D3_D7": "models/ROAS_D7_1208.pkl",
    "D3_D15": "models/ROAS_D15_1208.pkl",
    "D3_D30": "models/ROAS_D30_1208.pkl",
    "D3_D60": "models/ROAS_D60_1208.pkl",
    # ROAS from another day
    "D7_D15": "models/ROAS_D7_D15_1208.pkl",
    "D7_D30": "models/ROAS_D7_D30_1208.pkl",
    "D7_D60": "models/ROAS_D7_D60_1208.pkl",
    "D15_D30": "models/ROAS_D15_D30_1208.pkl",
    "D15_D60": "models/ROAS_D15_D60_1208.pkl",
    "D30_D60": "models/ROAS_D30_D60_1208.pkl",
    # Day of return cost
    "DOR_D7": "models/DOR_D7.pkl",
    "DOR_D15": "models/DOR_D15.pkl",
    "DOR_D30": "models/DOR_D30.pkl"
}

# === Main UI Function ===
def roas_prediction_ui(roas_file):
    try:
        # Load and preprocess
        df = pd.read_csv(roas_file)
        df = preprocess(df)
        st.write("📋 Data Preview", df.head())

        # Load all models at once
        models = {key: load_model(path) for key, path in MODEL_PATHS.items()}

        # Identify last ROAS day available
        roas_days = [3, 7, 15, 30, 60]
        available_days = [d for d in roas_days if f"roas - Rate - day {d}" in df.columns]
        if not available_days:
            st.error("No ROAS columns found in the uploaded file.")
            return

        last_day = max(available_days)
        st.info(f"📅 Last ROAS day available: **Day {last_day}**")

        # Determine which future ROAS to predict
        prediction_targets = [d for d in roas_days if d > last_day]
        st.write(f"🔮 Will predict ROAS for days: {prediction_targets}")

        # Feature columns
        excluded_cols = ['Cohort Day', 'Media Source'] + [f'roas - Rate - day {d}' for d in roas_days if d > last_day]
        feature_cols = [col for col in df.columns if col not in excluded_cols]

        # Predict ROAS future days
        for target_day in prediction_targets:
            model_key = f"D{last_day}_D{target_day}"
            if model_key in models:
                df[f"Predicted ROAS day {target_day}"] = models[model_key].predict(df[feature_cols])
            else:
                st.warning(f"No model found for {model_key}")

        # Predict Day of Return
        #if last_day < 7:
            #st.warning("Not enough data: last available ROAS day < 7. Skipping BED prediction.")
        #else:
            #dor_model_key = f"DOR_D{last_day}"
            #if dor_model_key in models:
                #df["Predicted Break-even Day"] = np.ceil(models[dor_model_key].predict(df[feature_cols]))
            #else:
                #st.warning(f"No model found for Day {last_day}")

        # === Summary ===
        if "Cohort Day" in df.columns:
            start_date = pd.to_datetime(df['Cohort Day']).min()
            end_date = pd.to_datetime(df['Cohort Day']).max()

            st.subheader("📜 Prediction Summary")
            st.write(f"Your input range: **{start_date} → {end_date}**")
            st.write(f"Last ROAS available: **Day {last_day}**")

        # Aggregated prediction ROAS
        for target_day in prediction_targets:
            pred_col = f"Predicted ROAS day {target_day}"
            if pred_col in df.columns:
                agg_roas = (df[pred_col] * df['Average eCPI'] * df['Users']).sum() / 100 / (df['Average eCPI'] * df['Users']).sum()
                st.write(f"Predicted ROAS day {target_day}: **{agg_roas*100:.4f}%**")

        # Compute max DOR relative to earliest cohort
        #if "Cohort Day" in df.columns and "Predicted Break-even Day" in df.columns:
            #df["Cohort Day"] = pd.to_datetime(df["Cohort Day"], errors="coerce")
            #df["Predicted Break-even Day"] = pd.to_numeric(df["Predicted Break-even Day"], errors="coerce")
            #max_dor = (df["Cohort Day"] + pd.to_timedelta(df["Predicted Break-even Day"], unit="D")).max() - df["Cohort Day"].min()
            #max_dor = max_dor.days
            #st.write(f"Predicted Break-even Day (max): **{max_dor} days**")

        # Predictions Display
        st.write(" **Predictions Preview**", df)

        # === Download Button ===
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Result CSV",
            csv,
            "roas_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
