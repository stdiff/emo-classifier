from importlib import resources

import pandas as pd
from scipy.sparse import coo_matrix


def load_emotions() -> list[str]:
    with resources.open_text("emo_classifier.resources", "emotions.txt") as fp:
        return [line.rstrip() for line in fp]


def vectorize_emotions(comma_separated_idx: str, emotions: list[str]) -> pd.Series:
    vector = pd.Series([0] * len(emotions), index=emotions)
    indices = [int(idx) for idx in comma_separated_idx.split(",")]
    for idx in indices:
        vector[idx] = 1
    return vector


def vectorize_series_of_emotions(s_csv: pd.Series, emotions: list[str]) -> pd.DataFrame:
    """
    use this function if you want to process lots of comma separated emotions

    :param s_csv:
    :param emotions:
    :return: DataFrame[admiration, amusement, ..., neutral]
    """
    data, row, col = [], [], []

    for i, vals in enumerate(s_csv):
        for val in vals.split(","):
            data.append(1)
            row.append(i)
            col.append(int(val))

    sparse = coo_matrix((data, (row, col)), shape=(len(s_csv), len(emotions)))
    return pd.DataFrame(sparse.toarray(), columns=emotions, index=s_csv.index)
