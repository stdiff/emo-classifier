from typing import Optional
from pathlib import Path
from shutil import rmtree
from abc import ABC, abstractmethod
import os

from emo_classifier import setup_logger
from emo_classifier.metrics import TrainingMetrics
from emo_classifier.model import Model

_PROJ_ROOT = Path(__file__).parents[1]
_DATA_DIR = _PROJ_ROOT / "data"


class LocalPaths:
    """
    Provides a unified API for file/directory locations, regardless of training on sagemaker or local

    cf. https://docs.aws.amazon.com/sagemaker/latest/dg/amazon-sagemaker-toolkits.html
    """

    def __init__(self):
        self.on_sagemaker = True if os.getenv("SM_MODEL_DIR") else False
        self.project_root = Path("/opt/ml/") if self.on_sagemaker else _PROJ_ROOT
        self.code_root = (self.project_root / "code") if self.on_sagemaker else self.project_root

        self.dir_datasets = self.project_root / ("input/data/datasets" if self.on_sagemaker else "data")
        """Directory where training/dev sets are stored."""

        self.dir_artifact = self.code_root / "emo_classifier/artifact"
        """Directory where model artifacts (joblib, pytorch model, etc) should be stored."""

        self.dir_resources = self.code_root / "emo_classifier/resources"
        """Directory where resources (thresholds, metrics, test data) should be stored."""

        self.sm_model_dir = (self.project_root / "model") if self.on_sagemaker else None
        """SageMakers model dir ($SM_MODEL_DIR). The files under this directory are archived in model.tar.gz"""

        self.sm_output_data_dir = (self.project_root / "output/data") if self.on_sagemaker else None
        """SageMakers output dir ($SM_OUTPUT_DATA_DIR). The files under this directory are archived in output.tar.gz"""


local_paths = LocalPaths()


class TrainerBase(ABC):
    def __init__(self, log_file: Optional[Path] = None, classifier: Optional[Model] = None):
        self.logger = setup_logger(type(self).__name__, log_file=log_file)
        self.classifier = classifier
        self.training_metrics: Optional[TrainingMetrics] = None
        self._initialize_artifact_dir()

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self):
        self.classifier.save()
        self.training_metrics.save()

    def _initialize_artifact_dir(self):
        """
        Initialize the artifact directory.
        """
        if local_paths.dir_artifact.exists():
            for file in local_paths.dir_artifact.iterdir():
                if file.is_dir():
                    rmtree(file)
                else:
                    file.unlink()
        else:
            local_paths.dir_artifact.mkdir()
        (local_paths.dir_artifact / "__init__.py").touch()
