import logging
import tarfile
from pathlib import Path

from sagemaker.s3 import S3Uploader, S3Downloader

from training import PROJ_ROOT, DATA_DIR, setup_logger, LocalPaths
from training.train_tfidf import start_training_tfidf_model

logging.getLogger().setLevel(logging.INFO)
logger = setup_logger(__name__)
local_paths = LocalPaths()


def train_tfidf_model_on_sagemaker():
    """"""
    ## 1 upload datasets (no additional preprocessing)
    training_set_path = local_paths.dir_datasets / "train.parquet"

    from training.utils_for_sagemaker import project_root_s3

    training_set_s3_dir = f"{project_root_s3}/datasets"
    training_set_s3_path = S3Uploader.upload(str(training_set_path), desired_s3_uri=training_set_s3_dir)
    logger.info(f"Training set uploaded: {training_set_s3_path}")

    ## 2 archive source_dir
    from tempfile import TemporaryDirectory
    from training.utils_for_sagemaker import archive_training_modules

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        source_tar_ball_path = temp_dir_path / "sourcedir.tar.gz"
        archive_training_modules(source_tar_ball_path)
        sourcedir_s3_uri = S3Uploader.upload(str(source_tar_ball_path), desired_s3_uri=f"{project_root_s3}/code")
        logger.info(f"Tar ball uploaded: {sourcedir_s3_uri}")

    ## 3 fit Estimate instance
    entry_point = PROJ_ROOT / "training/train_tfidf.py"
    """
    in Estimator.fit()

    1. load training set
    2. preprocess the training set → feature matrix
    3. train the model
    4. save the model and TrainingMetrics.json
    """
    import json
    from sagemaker.estimator import Estimator
    from training.utils_for_sagemaker import generate_tag_list

    role = "AmazonSageMaker-ExecutionRole-20210315T231867"
    container_image_uri = "050266116122.dkr.ecr.eu-central-1.amazonaws.com/smage:py39ml13"
    tags = generate_tag_list(Project="SageMakerTest", Owner="hironori.sakai", Env="DEV")

    hyperparameters = {"sagemaker_program": "training/train_tfidf.py", "sagemaker_submit_directory": sourcedir_s3_uri}
    hyperparameters = {str(k): json.dumps(v) for k, v in hyperparameters.items()}

    estimator = Estimator(
        image_uri=container_image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        # instance_type="local",
        base_job_name="custom-container",
        output_path=f"{project_root_s3}/output",
        hyperparameters=hyperparameters,
        use_spot_instances=True,
        max_run=60 * 60 * 3,
        max_wait=60 * 60 * 3,
        tags=tags,
    )
    estimator.fit({"datasets": training_set_s3_dir})

    # 4 download the model, output as results
    model_tarball_s3_path = estimator.model_data
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", model_tarball_s3_path)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", estimator.latest_training_job.name)
    output_tarball_s3_path = model_tarball_s3_path.replace("model.tar.gz", "output.tar.gz") ## any better way?

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        model_tarball_local_path = temp_dir_path / "model.tar.gz"
        output_tarball_local_path = temp_dir_path / "output.tar.gz"
        S3Downloader.download(model_tarball_s3_path, str(model_tarball_local_path.parent))
        S3Downloader.download(output_tarball_s3_path, str(output_tarball_local_path.parent))

        with tarfile.open(model_tarball_local_path, "r:gz") as model_tarball:
            model_tarball.extractall(local_paths.dir_artifact)
            print(f"The archived files in {model_tarball_local_path.name} is saved under {local_paths.dir_artifact}")

        with tarfile.open(output_tarball_local_path, "r:gz") as output_tarball:
            output_tarball.extractall(local_paths.dir_resources)
            print(f"The archived files in {output_tarball_local_path.name} is saved under {local_paths.dir_resources}")


def start():
    """
    This function can be executed by command "poetry run train"
    """
    start_training_tfidf_model()


if __name__ == "__main__":
    train_tfidf_model_on_sagemaker()
