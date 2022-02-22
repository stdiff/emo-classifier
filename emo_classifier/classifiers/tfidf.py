from typing import BinaryIO, Optional, Union
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

from emo_classifier.api import Comment, Prediction
from emo_classifier.metrics import Thresholds
from emo_classifier.classifiers.text import SpacyEnglishTokenizer
from emo_classifier.model import Model
from emo_classifier.emotion import load_emotions


class TfidfClassifier(Model):
    """
    responsible for
    - load/save the necessary artifacts (numerator, tfidf model)
    - prediction on the production
    """

    artifact_file_name = "tfidf.joblib"

    def __init__(self, min_df: int, with_lemmatization: bool = False, remove_stopwords: bool = True):
        self.tokenizer = SpacyEnglishTokenizer(with_lemmatization=with_lemmatization, remove_stopwords=remove_stopwords)
        self.min_df = min_df
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, min_df=self.min_df, token_pattern=None)
        self._model: Optional[OneVsRestClassifier] = None
        self._thresholds: Optional[Thresholds] = None
        self._s_thresholds: Optional[pd.Series] = None
        self._dict_thresholds: Optional[dict[str, float]] = None

    @property
    def model(self) -> OneVsRestClassifier:
        if self._model is None:
            raise Exception("No model is set.")
        return self._model

    @model.setter
    def model(self, model: OneVsRestClassifier):
        if not isinstance(model, OneVsRestClassifier):
            raise ValueError("You have to give an OneVsRestClassifier instance.")
        self._model = model

    @property
    def thresholds(self) -> Optional[Thresholds]:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds: Thresholds):
        self._thresholds = thresholds
        self._s_thresholds = thresholds.as_series()[self.emotions]
        self._dict_thresholds = thresholds.as_dict()

    def predict_proba(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        :param X: Series of texts
        :return: array of prediction of shape (#instances, #emotions)
        """
        X_vectorized = self.tfidf_vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)

    def predict_labels(self, X: Union[pd.Series, np.ndarray]):
        if self.thresholds is None:
            raise ValueError("The thresholds are not given.")

        y_prob = self.predict_proba(X)
        return np.where(y_prob > self._s_thresholds.values, 1, 0)

    def save_artifact_file(self, path: Path):
        joblib.dump(self, path, compress=3)
        # self.training_metrics.save()

    def predict(self, comment: Comment) -> Prediction:
        X = np.array([comment.text])
        y = self.predict_labels(X)[0, :]
        emotions = [emotion for i, emotion in enumerate(self.emotions) if y[i] > self._dict_thresholds.get(emotion)]
        return Prediction(id=comment.id, labels=emotions)

    @classmethod
    def load_artifact_file(cls, fp: BinaryIO) -> "TfidfClassifier":
        return joblib.load(fp)
