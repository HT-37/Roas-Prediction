import pandas as pd
import gdown
import os

def load_games_list(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
    
def download_from_gdrive(file_id: str, local_path: str = "data.csv") -> str:
    """
    Download file from Google Drive if not already exists.
    """
    if not os.path.exists(local_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, local_path, quiet=False)
    return local_path


def load_csv(file_id: str, local_path: str = "data.csv", chunksize: int = None):
    """
    Load CSV from Google Drive.
    
    Args:
        file_id (str): Google Drive file ID.
        local_path (str): Path to save the downloaded file.
        chunksize (int, optional): If set, return an iterator that reads the file in chunks.

    Returns:
        pd.DataFrame or Iterator[pd.DataFrame]: Dataframe (full load) or chunks iterator.
    """
    file_path = download_from_gdrive(file_id, local_path)

    if chunksize:
        return pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
    else:
        return pd.read_csv(file_path)
