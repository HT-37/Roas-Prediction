import streamlit as st
import pandas as pd
from data_loader import load_games_list, load_csv
from roas_model import roas_prediction_ui
from revenue_model import revenue_prediction_ui

st.set_page_config(page_title="Game Revenue & ROAS Predictor", layout="wide")

# --- Sidebar / User Settings ---
with st.sidebar:
    st.header("⚙️ User Settings")

    # Show developer notes
    show_notes = st.checkbox("Show Developer Notes", value=False)

    # Select Game
    games_df = load_games_list("data/games_list.csv")
    game_name = st.selectbox("Choose Game:", games_df["game_name"].unique())

    # Upload ROAS file
    roas_file = st.file_uploader("Upload ROAS file (CSV)", type=["csv"])

# --- Main Layout ---
st.title("🎮 Game Revenue & ROAS Prediction")

# Tabs for two main tasks
tab1, tab2 = st.tabs(["📊 Cohort ROAS Prediction", "💰 Cash Flow Prediction"])

with tab1:
    if roas_file is not None:
        roas_prediction_ui(roas_file)
    else:
        st.warning("Please upload a ROAS file to continue.")

with tab2:
    FILE_ID = "1N5MDyRjeDsKK1t6aZHGFa-LvHYADF_tC"
    chunk_iter = load_csv(FILE_ID, "data/revenue.csv", chunksize=100000)
    filtered_rows = []
    for chunk in chunk_iter:
        filtered_rows.append(chunk[chunk["product_name"] == selected_product])
    # Combine all filtered chunks
    filtered_df = pd.concat(filtered_rows, ignore_index=True)
    
    revenue_prediction_ui(filtered_df, game_name)

# --- Developer Notes ---
if show_notes:
    st.markdown("---")
    st.subheader("📝 Developer Notes")
    st.write("""
    - **ROAS Prediction**: Based on your uploaded file.
    - **Revenue Prediction**: Uses Prophet model on historical game revenue.
    - Cumulative revenue vs cost is simulated with user input planned cost.
    """)
