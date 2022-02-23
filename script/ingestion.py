import sys
from pathlib import Path

import pandas as pd

from emo_classifier import setup_logger
from training import LocalPaths

logger = setup_logger(__name__)
local_paths = LocalPaths()
sys.path.append(str(local_paths.project_root))


def download_csv_as_parquet(file_url: str, file_path: Path, **kwargs):
    if file_path.exists():
        logger.info(f"The file exists. Nothing is downloaded. Path = {file_path.relative_to(local_paths.project_root)}")
    else:
        df = pd.read_csv(file_url, **kwargs)
        logger.info(f"{file_path.name}: {df.shape[0]} x {df.shape[1]}")
        df.to_parquet(file_path, index=False)


def download_merged_data_sets():
    base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
    data_set_types = ["train", "dev", "test"]
    file_urls = (f"{base_url}{typ}.tsv" for typ in data_set_types)
    file_paths = (local_paths.dir_datasets / f"{typ}.parquet" for typ in data_set_types)

    for file_url, file_path in zip(file_urls, file_paths):
        download_csv_as_parquet(file_url, file_path, sep="\t", header=None, names=["text", "emotions", "id"])


def download_raw_data_sets():
    base_url = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/"
    file_name = "goemotions_%s"

    file_urls = (f"{base_url}{file_name % i}.csv" for i in range(1, 4))
    file_paths = (local_paths.dir_datasets / f"{file_name % i}.parquet" for i in range(1, 4))

    for file_url, file_path in zip(file_urls, file_paths):
        download_csv_as_parquet(file_url, file_path, sep=",")


def start():
    download_raw_data_sets()
    download_merged_data_sets()
