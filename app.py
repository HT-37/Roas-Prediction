import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Load Models ===
@st.cache_data()
def load_model():
    return joblib.load("ROAS_D30.pkl")

@st.cache_data()
def load_model_2():
    return joblib.load("ROAS_D60.pkl")

@st.cache_data()
def load_model_3():
    return joblib.load("ROAS_D15.pkl")

@st.cache_data()
def preprocess(df):
    df.replace(0, np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[df['Users'] >= 50]
    return df

# === App Header ===
st.title("📊 ROAS Prediction App")

st.markdown("""
Welcome! You can either:  
1️⃣ **Test the model performance** using a dataset that includes actual target ROAS, or  
2️⃣ **Predict ROAS** for a new campaign with D0–D3 data.  

**Required Columns:**  
`Cohort Day`, `Media Source`, `Users`, `Average eCPI`, `roas - Rate - day 0`, `roas - Rate - day 1`,  
`roas - Rate - day 2`, `roas - Rate - day 3`, `sessions - Unique users - day 1`,  
`sessions - Unique users - day 2`, `sessions - Unique users - day 3`  
*(+ `roas - Rate - day 15`, `roas - Rate - day 30`, `roas - Rate - day 60` if testing accuracy)*
""")

# === Mode Selection ===
mode = st.radio("Select mode:", ["🧪 Test Model Accuracy", "🔮 Predict ROAS"])

# === File Upload ===
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = preprocess(df)
        st.write("📋 Data Preview", df.head())

        model = load_model()
        model_15 = load_model_3()
        model_60 = load_model_2()
        feature_cols = [col for col in df.columns if col not in ['Cohort Day', 'Media Source', 'roas - Rate - day 15', 'roas - Rate - day 30', 'roas - Rate - day 60']]

        if mode == "🧪 Test Model Accuracy":
            for col in ['roas - Rate - day 15', 'roas - Rate - day 30', 'roas - Rate - day 60']:
                if (col == 'roas - Rate - day 15') and (col not in df.columns):
                    st.error("❌ 'roas - Rate - day 15' column is missing. Please add it for evaluation.")
                elif col in df.columns:
                    X = df[feature_cols]
                    y_true = df[col]
                    if col == 'roas - Rate - day 15':
                        y_pred = model_15.predict(X)
                    elif col == 'roas - Rate - day 30': 
                        y_pred = model.predict(X)
                    elif col == 'roas - Rate - day 60':
                        y_pred = model_60.predict(X)

                    mae = mean_absolute_error(y_true, y_pred)

                    df[f"Predicted {col}"] = y_pred

                    st.write(f"📈 Model Evaluation for {col}")
                    st.write(f"**MAE:** {mae:.2f}")
                    st.dataframe(df)

        elif mode == "🔮 Predict ROAS":
            X = df[feature_cols]
            df["Predicted ROAS D15"] = model_15.predict(X)
            df["Predicted ROAS D30"] = model.predict(X)
            df["Predicted ROAS D60"] = model_60.predict(X)

            st.write("✅ ROAS Predictions")
            st.dataframe(df)

        # === Download Button ===
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Result CSV", csv, "roas_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
