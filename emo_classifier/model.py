from typing import BinaryIO, Optional, Union
from importlib import resources
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from emo_classifier.api import Comment, Prediction
from emo_classifier.artifact import DATA_DIR, Thresholds
from emo_classifier.text import Tokenizer
from emo_classifier.emotion import load_emotions


class Model(ABC):
    @classmethod
    @abstractmethod
    def load_artifact_file(cls, fp: BinaryIO):
        pass

    @classmethod
    def load(cls) -> "Model":
        file_name = f"{cls.__name__}.json"
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





class TfidfClassifier(Model):
    def __init__(self, tokenizer: Tokenizer, min_df: int, cv: int = 5):
        self.tokenizer = tokenizer
        self.min_df = min_df
        self.cv = cv

        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, min_df=self.min_df, token_pattern=None)
        self.tfidf_vectorizer_name = "tfidf"
        self.model: Optional[GridSearchCV] = None
        self.model_name: Optional[str] = None
        self.param_grid: Optional[dict] = None
        self.cv_results: Optional[pd.DataFrame] = None
        self._thresholds: Optional[pd.Series] = None
        self._emotions: Optional[list[str]] = None

    def set_model(self, name: str, classifier: BaseEstimator, param_grid: dict, n_jobs=-1):
        self.model_name = name
        self.param_grid = {f"estimator__{key}": val for key, val in param_grid.items()}
        ## no Pipeline with TFIDF because of memory issue.
        self.model = GridSearchCV(
            OneVsRestClassifier(classifier), self.param_grid, cv=self.cv, scoring="f1_macro", return_train_score=True,
            n_jobs=n_jobs
        )

    @property
    def thresholds(self) -> Optional[pd.Series]:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds: Thresholds):
        self._thresholds = thresholds.as_series()[self._emotions]

    def fit(self, X: pd.Series, y: pd.DataFrame) -> pd.DataFrame:
        """

        :param X: series of text
        :param y: DataFrame of binarized labels
        :return: result of cross-validation
        """
        if self.model is None:
            raise ValueError("Set model by set_model() in advance.")
        self._emotions = y.columns.tolist()

        X_vectorized = self.tfidf_vectorizer.fit_transform(X)
        self.model.fit(X_vectorized, y)

        cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time"]
        cols.extend([c for c in self.model.cv_results_.keys() if c.startswith("param_")])
        self.cv_results = pd.DataFrame(self.model.cv_results_)[cols].sort_values(by="rank_test_score")

        return self.cv_results

    def predict_proba(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        X_vectorized = self.tfidf_vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)

    def predict_labels(self, X: Union[pd.Series, np.ndarray]):
        if self.thresholds is None:
            raise ValueError("The thresholds are not given.")

        y_prob = self.predict_proba(X)
        return np.where(y_prob > self.thresholds, 1, 0)

    def save_artifact_file(self, path: Path):
        joblib.dump(self, path, compress=3)

    def predict(self, comment: Comment) -> Prediction:
        X = np.array([self.tokenizer(comment.text)])
        y = self.predict_labels(X)[0, :]
        emotions = [emotion for i, emotion in enumerate(self._emotions) if y[i] > 0.5]
        return Prediction(id=comment.id, lables=emotions)

    @classmethod
    def load_artifact_file(cls, fp: BinaryIO) -> "TfidfClassifier":
        return joblib.load(fp)


if __name__ == "__main__":
    pass