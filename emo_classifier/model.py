from typing import BinaryIO
from importlib import resources
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree

from emo_classifier.api import Comment, Prediction
from emo_classifier import ARTIFACT_DIR


class Model(ABC):
    """
    An abstract class for a model class. This provides a united interface for
    - saving a model under ARTIFACT_DIR,
    - loading a model and
    - making a prediction (for REST API)
    """

    artifact_file_name = "model.model"

    @classmethod
    @abstractmethod
    def load_artifact_file(cls, fp: BinaryIO):
        pass

    @classmethod
    def load(cls) -> "Model":
        with resources.open_binary("emo_classifier.artifact", cls.artifact_file_name) as fp:
            model = cls.load_artifact_file(fp)
        print(f"LOADED: {type(model).__name__} instance")
        return model

    @abstractmethod
    def save_artifact_file(self, path: Path):
        pass

    def _initialize_artifact_dir(self):
        if ARTIFACT_DIR.exists():
            for file in ARTIFACT_DIR.iterdir():
                if file.is_dir():
                    rmtree(file)
                else:
                    file.unlink()
        else:
            ARTIFACT_DIR.mkdir()
        (ARTIFACT_DIR / "__init__.py").touch()

    def save(self):
        self._initialize_artifact_dir()
        file_path = ARTIFACT_DIR / self.artifact_file_name
        self.save_artifact_file(file_path)
        print("SAVED:", file_path.absolute())

    @abstractmethod
    def predict(self, comment: Comment) -> Prediction:
        pass
