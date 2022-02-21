from pathlib import Path

from sagemaker.s3 import S3Uploader, S3Downloader

from training import PROJ_ROOT, DATA_DIR, setup_logger
from training.train_tfidf import start_training_tfidf_model

logger = setup_logger(__name__)


def train_():
    """"""
    ## 1 upload datasets (no additional preprocessing)
    training_set_path = DATA_DIR / "train.parquet"

    from training.utils_for_sagemaker import project_root_s3

    training_set_s3_dir = f"{project_root_s3}/train"
    training_set_s3_path = S3Uploader.upload(str(training_set_path), desired_s3_uri=training_set_s3_dir)
    logger.info(f"Training set uploaded: {training_set_s3_path}")

    ## 2 archive source_dir
    from tempfile import TemporaryDirectory
    from training.utils_for_sagemaker import archive_training_modules

    entry_point = PROJ_ROOT / "training/train_tfidf.py"

    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        source_tar_ball_path = temp_dir_path / "sourcedir.tar.gz"
        archive_training_modules(entry_point, source_tar_ball_path)
        tar_ball_s3_path = S3Uploader.upload(str(source_tar_ball_path), desired_s3_uri=f"{project_root_s3}/code")
        logger.info(f"Tar ball uploaded: {tar_ball_s3_path}")

    ## 3 fit Estimate instance
    """
    in Estimator.fit()
    
    1. load training set
    2. preprocess the training set → feature matrix
    3. train the model 
    4. save the model and TrainingMetrics.json
    """
    ## 4 download the model, output as results

    ## 5 unpack the model artifact and put it under emo_classifier

    pass


def start():
    """
    This function can be executed by command "poetry run train"
    """
    start_training_tfidf_model()


if __name__ == "__main__":
    start()
