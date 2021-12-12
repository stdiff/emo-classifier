from typing import BinaryIO
from importlib import resources
from abc import ABC, abstractmethod
from pathlib import Path

from emo_classifier.api import Comment, Prediction
from emo_classifier.artifact import DATA_DIR

class Model(ABC):
    @classmethod
    @abstractmethod
    def load_artifact_file(cls, fp: BinaryIO):
        pass

    @classmethod
    def load(cls) -> "Model":
        file_name = f"{cls.__name__}.joblib"
        with resources.open_binary("emo_classifier.data", file_name) as fp:
            model = cls.load_artifact_file(fp)
        print(f"LOADED: {type(model).__name__} instance")
        return model

    @abstractmethod
    def save_artifact_file(self, path: Path):
        pass

    def save(self):
        file_path = DATA_DIR / f"{type(self).__name__}.joblib"
        self.save_artifact_file(file_path)
        print("SAVED:", file_path.absolute())

    @abstractmethod
    def predict(self, comment: Comment) -> Prediction:
        pass