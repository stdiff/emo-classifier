"""
This module just helps us start a SageMaker training job, this does not accompany the entry point.
Therefore, do not import this module from the training scripts on the sagemaker.
"""
from enum import Enum
from pathlib import Path
import tarfile

from sagemaker.s3 import S3Uploader, S3Downloader

from training import get_logger, PROJ_ROOT, LocalPaths

logger = get_logger(__name__)
region = "eu-central-1"
role = "AmazonSageMaker-ExecutionRole-20210315T231867"
project_root_s3 = "s3://stdiff/sagemaker/emo-classifier"
local_paths = LocalPaths()

class InstanceType(str, Enum):
    """
    SageMaker's Estimator can accept a subset of the instance types listed at the following page.
    https://aws.amazon.com/sagemaker/pricing/
    """

    ml_m5_large = "ml.m5.large"  ## vCPU = 2, RAM =  8GB, no GPU, $0.138 per hour
    ml_m5_xlarge = "ml.m5.xlarge"  # vCPU = 4, RAM = 16 GiB, no GPU, $0.276 per hour
    ml_g4dn_xlarge = "ml.g4dn.xlarge"  ## vCPU = 4, RAM = 16GiB, GPU enabled, $0.921 per hour
    ml_p3_2xlarge = "ml.p3.2xlarge"  ## vCPU = 8, RAM = 61GiB, GPU enabled, $4.779 per hour


def generate_tag_list(**kwargs) -> list[dict[str, str]]:
    """
    converts key-value pairs into a suitable format for a tag for AWS

    :param kwargs: key1=value1, key2=value2, ...
    """
    return [{"Key": key, "Value": value} for key, value in kwargs.items()]


def archive_training_modules(tar_ball_path: Path):
    """
    Create a tar ball for SageMaker training job. The tar ball will include

    - modules training.*
    - emo_classifier/resources/emotions.txt and __init__.py
    - modules emo_classifier.*
    - modules emo_classifier.classifiers.*

    :param tar_ball_path: Path instance of the tar ball to create
    """
    if tar_ball_path.exists():
        logger.warning(f"The tar ball will be overwritten.")

    pkg_dir = PROJ_ROOT / "emo_classifier"

    with tarfile.open(tar_ball_path, "w:gz") as tar_file:

        files_to_add = [
            pkg_dir / "resources/emotions.txt",
            pkg_dir / "resources/__init__.py",
        ]
        files_to_add.extend([file for file in (PROJ_ROOT / "training").iterdir() if file.name.endswith("py")])
        files_to_add.extend([file for file in pkg_dir.iterdir() if file.name.endswith("py")])
        files_to_add.extend([file for file in (pkg_dir / "classifiers").iterdir() if file.name.endswith("py")])

        for file in files_to_add:
            tar_file.add(str(file), file.relative_to(PROJ_ROOT))


def upload_datasets(file_names: list[str]) -> list[str]:
    uploaded_files_in_s3: list[str] = []
    datasets_dir_in_s3 = f"{project_root_s3}/datasets"

    for file_name in file_names:
        file_path = local_paths.dir_datasets / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Not Found: {file_path}")
        uploaded_file_in_s3 = S3Uploader.upload(str(file_path), desired_s3_uri=datasets_dir_in_s3)
        logger.info(f"Uploaded: {file_path} → {uploaded_file_in_s3}")
        uploaded_files_in_s3.append(uploaded_file_in_s3)

    return uploaded_files_in_s3
