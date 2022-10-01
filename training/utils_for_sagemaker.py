"""
This module just helps us start a SageMaker training job, this does not accompany the entry point.
Therefore, do not import this module from the training scripts on the sagemaker.
"""
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import shutil
import json

from sagemaker.s3 import S3Uploader, S3Downloader
from sagemaker.estimator import Estimator

from emo_classifier import setup_logger
from training import LocalPaths

logger = setup_logger(__name__)
region = "eu-central-1"
role = "AmazonSageMaker-ExecutionRole-20210315T231867"
project_root_s3 = "s3://stdiff/sagemaker/emo-classifier"
custom_image_uri = "050266116122.dkr.ecr.eu-central-1.amazonaws.com/smage:py39ml13"
local_paths = LocalPaths()


class InstanceType(str, Enum):
    """
    SageMaker's Estimator can accept a subset of the instance types listed at the following page.
    https://aws.amazon.com/sagemaker/pricing/
    """

    local = "local"
    ml_m5_large = "ml.m5.large"
    """vCPU = 2, RAM =  8GB, no GPU, $0.138 per hour"""
    ml_m5_xlarge = "ml.m5.xlarge"
    """vCPU = 4, RAM = 16 GiB, no GPU, $0.276 per hour"""
    ml_g4dn_xlarge = "ml.g4dn.xlarge"
    """vCPU = 4, RAM = 16GiB, GPU enabled, $0.921 per hour"""
    ml_p3_2xlarge = "ml.p3.2xlarge"
    """vCPU = 8, RAM = 61GiB, GPU enabled, $4.779 per hour"""


def generate_tag_list(**kwargs) -> list[dict[str, str]]:
    """
    converts key-value pairs into a suitable format for a tag for AWS

    :param kwargs: key1=value1, key2=value2, ...
    """
    return [{"Key": key, "Value": value} for key, value in kwargs.items()]


def upload_datasets(*file_names) -> str:
    """
    Upload files under data/

    :param file_names: names of files to upload (as positional arguments)
    :return: S3 path to the datasets directory (SageMaker Estimator will need it.)
    """
    uploaded_files_in_s3: list[str] = []
    datasets_dir_in_s3 = f"{project_root_s3}/datasets"

    for file_name in file_names:
        file_path = local_paths.dir_datasets / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Not Found: {file_path}")
        uploaded_file_in_s3 = S3Uploader.upload(str(file_path), desired_s3_uri=datasets_dir_in_s3)
        logger.info(f"Uploaded: {file_path} → {uploaded_file_in_s3}")
        uploaded_files_in_s3.append(uploaded_file_in_s3)

    return datasets_dir_in_s3


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

    pkg_dir = local_paths.project_root / "emo_classifier"

    with tarfile.open(tar_ball_path, "w:gz") as tar_file:

        files_to_add = [
            pkg_dir / "resources/emotions.txt",
            pkg_dir / "resources/__init__.py",
        ]
        files_to_add.extend(
            [file for file in (local_paths.project_root / "training").iterdir() if file.name.endswith("py")]
        )
        files_to_add.extend([file for file in pkg_dir.iterdir() if file.name.endswith("py")])
        files_to_add.extend([file for file in (pkg_dir / "classifiers").iterdir() if file.name.endswith("py")])

        for file in files_to_add:
            tar_file.add(str(file), file.relative_to(local_paths.project_root))


def upload_sourcedir() -> str:
    """
    create a tar ball for SageMaker training job and upload it to the
    :return: S3 path to the uploaded sourcedir.tar.gz
    """
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        source_tar_ball_path = temp_dir_path / "sourcedir.tar.gz"
        archive_training_modules(source_tar_ball_path)
        sourcedir_s3_uri = S3Uploader.upload(str(source_tar_ball_path), desired_s3_uri=f"{project_root_s3}/code")
        logger.info(f"Tar ball uploaded: {sourcedir_s3_uri}")

    return sourcedir_s3_uri


def generate_base_hyperparameters(entry_point: Path, sourcedir_s3_uri) -> dict[str, str]:
    """
    generate a template hyperparameters dict for SageMaker Estimator.
    This function is convenient if we use a custom Docker image to train a model.

    :return: dict for hyperparameters argument of Estimator
    """
    hyperparameters = {
        "sagemaker_program": str(entry_point.relative_to(local_paths.project_root)),
        "sagemaker_submit_directory": sourcedir_s3_uri,
    }
    return {str(k): json.dumps(v) for k, v in hyperparameters.items()}


def download_sagemaker_outputs_to_local(model_tarball_s3_path: str):
    """
    Download output files (model.tar.gz and output.tar.gz) from S3,
    extract the archive files and move the extracted files under
    suitable directory. More precisely

    - Files in model.tar.gz go to artifact directory
    - Files in output.tar.gz go to resources directory

    :param model_tarball_s3_path: S3 URI of the model.tar.gz
    """
    ## TODO: we need a better way to get the S3 path to output.tar.gz. How about cloudpathlib?
    output_tarball_s3_path = model_tarball_s3_path.replace("model.tar.gz", "output.tar.gz")

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        model_tarball_local_path = temp_dir_path / "model.tar.gz"

        try:
            S3Downloader.download(model_tarball_s3_path, str(model_tarball_local_path.parent))
        except Exception:
            raise FileNotFoundError(f"Download failed: S3 URI = {model_tarball_s3_path}")

        with tarfile.open(model_tarball_local_path, "r:gz") as model_tarball:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(model_tarball, local_paths.dir_artifact)
            logger.info(
                f"The archived files in {model_tarball_local_path.name} is saved under {local_paths.dir_artifact}"
            )

        output_tarball_local_path = temp_dir_path / "output.tar.gz"

        try:
            S3Downloader.download(output_tarball_s3_path, str(output_tarball_local_path.parent))
        except Exception:
            raise FileNotFoundError(f"Download failed: S3 URI = {output_tarball_s3_path}")

        with tarfile.open(output_tarball_local_path, "r:gz") as output_tarball:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(output_tarball, local_paths.dir_resources)
            logger.info(
                f"The archived files in {output_tarball_local_path.name} is saved under {local_paths.dir_resources}"
            )


def copy_artifacts_for_outputs_if_on_sagemaker():
    """
    The artifacts of the training process will be created under artifacts/ or resources/.
    This function copies such files under certain directories only if we are on a SageMaker instance.
    """
    if local_paths.on_sagemaker:
        for file in local_paths.dir_artifact.iterdir():
            if not file.is_dir():
                shutil.copy(file, local_paths.sm_model_dir)
                logger.info(f"COPY: {file} -> {local_paths.sm_model_dir}")
        for file in local_paths.dir_resources.iterdir():
            if not file.is_dir():
                shutil.copy(file, local_paths.sm_output_data_dir)
                logger.info(f"COPY: {file} -> {local_paths.sm_output_data_dir}")


def start_sagemaker_training_job(
    base_job_name: str,
    entry_point: Path,
    tags: list[dict[str, str]],
    instance_type: str = InstanceType.local,
    instance_count: int = 1,
    max_run_in_hour: int = 3,
    **hyperparameters,
):
    datasets_dir_in_s3 = upload_datasets("train.parquet")
    sourcedir_s3_uri = upload_sourcedir()
    hyperparameters4estimator = generate_base_hyperparameters(entry_point, sourcedir_s3_uri)

    for param, value in hyperparameters.items():
        hyperparameters4estimator[param] = value

    estimator = Estimator(
        image_uri=custom_image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        base_job_name=base_job_name,
        output_path=f"{project_root_s3}/output",
        hyperparameters=hyperparameters4estimator,
        use_spot_instances=False if instance_type == InstanceType.local else True,
        max_run=60 * 60 * max_run_in_hour,
        max_wait=60 * 60 * max_run_in_hour,
        tags=tags,
    )

    estimator.fit({"datasets": datasets_dir_in_s3})
    model_tarball_s3_path = estimator.model_data
    download_sagemaker_outputs_to_local(model_tarball_s3_path)
