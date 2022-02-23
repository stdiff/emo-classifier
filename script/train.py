import logging

from emo_classifier import setup_logger
from training import LocalPaths
from training.train_tfidf import start_training_tfidf_model
from training.utils_for_sagemaker import InstanceType, generate_tag_list, start_sagemaker_training_job

logging.getLogger().setLevel(logging.INFO)
logger = setup_logger(__name__)
local_paths = LocalPaths()


def train_tfidf_model_on_sagemaker():
    entry_point = local_paths.project_root / "training/train_tfidf.py"
    tags = generate_tag_list(Project="emo-classifier", Owner="hironori.sakai", Env="DEV")
    start_sagemaker_training_job(
        base_job_name="emo-classifier", entry_point=entry_point, tags=tags, instance_type=InstanceType.local
    )


def start():
    """
    This function can be executed by command "poetry run train"
    """
    start_training_tfidf_model()


if __name__ == "__main__":
    train_tfidf_model_on_sagemaker()
