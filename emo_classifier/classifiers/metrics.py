from dataclasses import dataclass, asdict
from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class SimpleStats:
    avg: float
    min: float
    max: float

    @classmethod
    def from_array(cls, array: Union[np.ndarray, list[float]]) -> "SimpleStats":
        return cls(avg=float(np.mean(array)), min=float(np.min(array)), max=float(np.max(array)))

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def stats_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> SimpleStats:
    """
    Compute average/max/min of areas under the roc curves

    :param y_true: binary matrix of shape (# instance, # label)
    :param y_score: score matrix of shape (# instance, # label)
    :return:
    """
    if y_true.shape != y_score.shape:
        raise ValueError(f"The shapes do not agree. ({y_true.shape} != {y_score.shape})")

    areas = []
    p = y_true.shape[1]
    for j in range(p):
        try:
            ## The dev set contains a label which is always negative. In this case AUC = 0.
            area = roc_auc_score(y_true[:, j], y_score[:, j])
        except ValueError:
            area = 0
        areas.append(area)

    return SimpleStats.from_array(areas)
