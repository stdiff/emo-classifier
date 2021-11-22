import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJ_ROOT))

from lib import DATA_DIR

def main():
    base_url = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/"
    file_name = "goemotions_%s.csv"

    file_urls = (f"{base_url}{file_name % i}" for i in range(1,4))
    file_paths = (DATA_DIR / (file_name % i) for i in range(1,4))

    for file_url, file_path in zip(file_urls, file_paths):
        if file_path.exists():
            raise FileExistsError(f"Remove the file before downloading it: path={file_path.absolute()}")

        df = pd.read_csv(file_url)
        print(f"{file_path.name}: {df.shape[0]} x {df.shape[1]}")
        df.to_parquet(file_path, index=False)

if __name__ == "__main__":
    main()