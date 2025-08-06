import streamlit as st
import pandas as pd
import joblib

# === Load Model ===
@st.cache_data()
def load_model():
    return joblib.load("ROAS_D30.pkl")

@st.cache_data()
def load_model_2():
    return joblib.load("ROAS_D60.pkl")

model = load_model()
model_2 = load_model_2()
# === App Title ===
st.title("📊 ROAS Prediction App")

st.markdown("""
*Hello, Welcome to ROAS Prediction model !* \n
Upload a CSV file with your UA campaign performance data (D0–D3) and get ROAS D15, D30, D60 predictions. \n
**Required Columns:**  
`Cohort Day`,`Media Source`,`Users`, `Average eCPI`, `roas - Rate - day 0`, `roas - Rate - day 1`, `roas - Rate - day 2`, `roas - Rate - day 3`,  
`sessions - Unique users - day 1`, `sessions - Unique users - day 2`, `sessions - Unique users - day 3`
""")

# === File Upload ===
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📋 Input Data Preview", df.head())

        # Predict
        predictions = model.predict(df.drop(columns=['Cohort Day','Media Source']))
        predictions_2 = model_2.predict(df.drop(columns = ['Cohort Day','Media Source']))
        df["Predicted ROAS D30"] = predictions
        df["Predicted ROAS D60"] = predictions_2

        st.write("✅ Prediction Results")
        st.dataframe(df)

        # Optional: Download result
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Result CSV", csv, "predicted_roas.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
