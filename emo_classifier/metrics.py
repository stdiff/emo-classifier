from typing import Union
from dataclasses import dataclass, asdict
from abc import ABC
from importlib import resources
import json

import pandas as pd

from emo_classifier import RESOURCES_DIR


@dataclass
class JsonArtifact(ABC):
    """NB: As the name suggests, all attributes must be JSON serializable."""

    @classmethod
    def load(cls) -> "JsonArtifact":
        file_name = f"{cls.__name__}.json"
        with resources.open_text("emo_classifier.resources", file_name) as fp:
            data_dict: dict = json.load(fp)

        print(f"LOADED: {type(data_dict)}")
        return cls(**data_dict)

    def save(self):
        file_path = RESOURCES_DIR / f"{type(self).__name__}.json"
        with file_path.open("w") as fp:
            json.dump(asdict(self), fp)
        print("SAVED:", file_path.absolute())

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Thresholds(JsonArtifact):
    admiration: float
    amusement: float
    anger: float
    annoyance: float
    approval: float
    caring: float
    confusion: float
    curiosity: float
    desire: float
    disappointment: float
    disapproval: float
    disgust: float
    embarrassment: float
    excitement: float
    fear: float
    gratitude: float
    grief: float
    joy: float
    love: float
    nervousness: float
    optimism: float
    pride: float
    realization: float
    relief: float
    remorse: float
    sadness: float
    surprise: float
    neutral: float

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, float]]) -> "Thresholds":
        label2threshold = {label: threshold for label, threshold in pairs}
        return Thresholds(**label2threshold)

    def as_series(self) -> pd.Series:
        s_threshold = pd.Series(asdict(self), name="threshold")
        s_threshold.index.name = "label"
        return s_threshold


@dataclass
class TrainingMetrics(JsonArtifact):
    model_class: str
    model_name: str
    best_params: dict[str, Union[str, float, int]]
    validation_score: float
    training_score: float
    training_timestamp: str  ## ISO format with timezone


@dataclass
class DevMetrics(JsonArtifact):
    macro_f1_score: float
    scores: dict[str, dict[str, float]]  # label -> score name -> score


@dataclass
class TestMetrics(JsonArtifact):
    macro_f1_score: float
