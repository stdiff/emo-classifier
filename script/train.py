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


def train_embedding_bag_model_on_local():
    from training.train_embedding_bag import start_train_embedding_bag_model

    logger.info(f"Start Training an embedding bag model on local machine")
    start_train_embedding_bag_model(embedding_dim=32, max_epoch=1, patience=5)


def train_embedding_bag_model_on_sagemaker():
    logger.info(f"Start Training an embedding bag model on sagemaker")
    entry_point = local_paths.project_root / "training/train_embedding_bag.py"
    tags = generate_tag_list(Project="emo-classifier", Owner="hironori.sakai", Env="DEV")
    start_sagemaker_training_job(
        base_job_name="emo-classifier", entry_point=entry_point, tags=tags, instance_type=InstanceType.ml_m5_xlarge
    )


def start():
    """
    This function can be executed by command "poetry run train"
    """
    # start_training_tfidf_model()
    # train_tfidf_model_on_sagemaker()
    # train_embedding_bag_model_on_local()
    train_embedding_bag_model_on_sagemaker()


if __name__ == "__main__":
    start()
