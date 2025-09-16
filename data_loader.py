import pandas as pd

def load_games_list(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_revenue_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df
