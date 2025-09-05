import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os
from sklearn.metrics import mean_absolute_error


def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        b64_str = base64.b64encode(img_file.read()).decode()
    ext = os.path.splitext(img_path)[1][1:]  # get extension without dot
    mime = f"image/{'jpeg' if ext=='jpg' else ext}"  # jpg -> jpeg for mime
    return f"data:{mime};base64,{b64_str}"

# === Inject background HTML & CSS ===
def set_background():
    leaf_01 = img_to_base64("images/leaf_01.png")
    leaf_02 = img_to_base64("images/leaf_02.png")
    leaf_03 = img_to_base64("images/leaf_03.png")
    leaf_04 = img_to_base64("images/leaf_04.png")

    background_code = f"""
    <style>
    .stApp {{
        background: transparent !important;
    }}
    .leaves {{
        position: absolute;
        width: 100%;
        height: 100vh;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1;
        pointer-events: none;
    }}
    .leaves .set {{
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0; left: 0;
        pointer-events: none;
    }}
    .leaves .set div {{ position: absolute; display: block; }}
    .leaves .set div:nth-child(1) {{ left: 20%; animation: animate 20s linear infinite; }}
    .leaves .set div:nth-child(2) {{ left: 50%; animation: animate 14s linear infinite; }}
    .leaves .set div:nth-child(3) {{ left: 70%; animation: animate 12s linear infinite; }}
    .leaves .set div:nth-child(4) {{ left: 5%;  animation: animate 15s linear infinite; }}
    .leaves .set div:nth-child(5) {{ left: 85%; animation: animate 18s linear infinite; }}
    .leaves .set div:nth-child(6) {{ left: 90%; animation: animate 12s linear infinite; }}
    .leaves .set div:nth-child(7) {{ left: 15%; animation: animate 14s linear infinite; }}
    .leaves .set div:nth-child(8) {{ left: 60%; animation: animate 15s linear infinite; }}

    @keyframes animate {{
        0% {{ opacity: 0; top: -10%; transform: translateX(5px) rotate(0deg); }}
        10% {{ opacity: 1; }}
        20% {{ transform: translateX(-5px) rotate(45deg); }}
        40% {{ transform: translateX(-5px) rotate(90deg); }}
        60% {{ transform: translateX(5px) rotate(180deg); }}
        80% {{ transform: translateX(-5px) rotate(45deg); }}
        100% {{ top: 110%; transform: translateX(5px) rotate(225deg); }}

    </style>

    <div id="custom-bg">
        <div class="leaves">
            <div class="set">
                <div><img src="{leaf_01}"></div>
                <div><img src="{leaf_02}"></div>
                <div><img src="{leaf_03}"></div>
                <div><img src="{leaf_04}"></div>
                <div><img src="{leaf_01}"></div>
                <div><img src="{leaf_02}"></div>
                <div><img src="{leaf_03}"></div>
                <div><img src="{leaf_04}"></div>
            </div>
        </div>
    </div>
    """
    st.markdown(background_code, unsafe_allow_html=True)

# === Call background function ===
set_background()

# === Cached Helpers ===
@st.cache_data
def load_model(path):
    return joblib.load(path)

@st.cache_data
def preprocess(df):
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)
    return df[df['Users'] >= 50]

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

# === App Header ===
st.title("📊 ROAS & Break-Even Day Prediction App")

st.markdown("""
Upload your campaign data with ROAS up to the latest day you have,  
and we’ll predict future ROAS + Break-even Day.

**Required Columns:**  
- `Cohort Day`, `Media Source`, `Users`, `Average eCPI`,  
- `roas - Rate - day 0`, `roas - Rate - day 1`, `roas - Rate - day 2`, `roas - Rate - day 3`,
- `sessions - Unique users - day 1`, `sessions - Unique users - day 2`, `sessions - Unique users - day 3`,
- *Optional*: `roas - Rate - day 7`, `roas - Rate - day 15`, `roas - Rate - day 30`  
""")

# === File Upload ===
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load and preprocess
        df = pd.read_csv(uploaded_file)
        df = preprocess(df)
        st.write("📋 Data Preview", df.head())

        # Load all models at once
        models = {key: load_model(path) for key, path in MODEL_PATHS.items()}

        # Identify last ROAS day available
        roas_days = [3, 7, 15, 30, 60]
        available_days = [d for d in roas_days if f"roas - Rate - day {d}" in df.columns]
    
        last_day = max(available_days)
        st.info(f"📅 Last ROAS day available: **Day {last_day}**")

        # Determine which future ROAS to predict
        prediction_targets = [d for d in roas_days if d > last_day]
        st.write(f"🔮 Will predict ROAS for days: {prediction_targets}")

        # Feature columns (exclude actual future ROAS columns)
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
        if last_day < 7:
            st.warning("Not enough data: last available ROAS day < 7. Skipping BED prediction.")
        else:
            dor_model_key = f"DOR_D{last_day}"
            #feature_cols.append('Media Source')
            if dor_model_key in models:
                df["Predicted Break-even Day"] = np.ceil(models[dor_model_key].predict(df[['Users','Average eCPI','roas - Rate - day 0','roas - Rate - day 1','roas - Rate - day 2','roas - Rate - day 3','roas - Rate - day 7','roas - Rate - day 15','roas - Rate - day 30',
                                                                                           'sessions - Unique users - day 1','sessions - Unique users - day 2','sessions - Unique users - day 3']))
            else:
                st.warning(f"No found model for Day {last_day}")

        # === Summary ===
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
                st.write(f"Predicted ROAS day {target_day}: **{agg_roas*100:.4f}**")

        # Aggregated DOR
        if "Predicted Break-even Day" in df.columns:
            max_dor = df["Predicted Break-even Day"].max()
            st.write(f"Predicted Break-even day: **{max_dor:.0f}**")

        # Predictions Diplay
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
