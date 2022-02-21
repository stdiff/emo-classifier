import logging

from sagemaker.estimator import Estimator

from training import setup_logger, LocalPaths
from training.train_tfidf import start_training_tfidf_model
from training.utils_for_sagemaker import (
    project_root_s3,
    custom_image_uri,
    role,
    InstanceType,
    upload_datasets,
    upload_sourcedir,
    generate_base_hyperparameters,
    generate_tag_list,
    download_sagemaker_outputs_to_local,
)

logging.getLogger().setLevel(logging.INFO)
logger = setup_logger(__name__)
local_paths = LocalPaths()


def train_tfidf_model_on_sagemaker():
    """"""
    datasets_dir_in_s3 = upload_datasets("train.parquet")
    sourcedir_s3_uri = upload_sourcedir()

    entry_point = local_paths.project_root / "training/train_tfidf.py"
    tags = generate_tag_list(Project="SageMakerTest", Owner="hironori.sakai", Env="DEV")
    hyperparameters = generate_base_hyperparameters(entry_point, sourcedir_s3_uri)
    base_job_name = "emo-classifier"

    estimator = Estimator(
        image_uri=custom_image_uri,
        role=role,
        instance_count=1,
        instance_type=InstanceType.local,
        base_job_name=base_job_name,
        output_path=f"{project_root_s3}/output",
        hyperparameters=hyperparameters,
        use_spot_instances=False,
        max_run=60 * 60 * 3,
        max_wait=60 * 60 * 3,
        tags=tags,
    )
    estimator.fit({"datasets": datasets_dir_in_s3})
    model_tarball_s3_path = estimator.model_data
    download_sagemaker_outputs_to_local(model_tarball_s3_path)


def start():
    """
    This function can be executed by command "poetry run train"
    """
    start_training_tfidf_model()


if __name__ == "__main__":
    train_tfidf_model_on_sagemaker()
