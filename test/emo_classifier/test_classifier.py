import pytest
import numpy as np
import pandas as pd

from emo_classifier.classifiers import load_classifier
from emo_classifier.metrics import Thresholds
from emo_classifier.api import Comment, Prediction
from emo_classifier.model import Model
from training import LocalPaths

local_paths = LocalPaths()


@pytest.fixture(scope="module")
def classifier():
    classifier = load_classifier()
    assert isinstance(classifier.thresholds, Thresholds)
    yield classifier


@pytest.fixture(scope="module")
def df_dev():
    dev_set_path = local_paths.dir_datasets / "dev.parquet"
    df_dev = pd.read_parquet(dev_set_path)  ## DataFrame[text, emotions, id]
    assert {"id", "text"} < set(df_dev.columns)
    yield df_dev


@pytest.fixture(scope="module")
def df_dev_pred():
    dev_prediction_path = local_paths.dir_resources / "dev_predictions.parquet"
    df_dev_pred = pd.read_parquet(dev_prediction_path)
    assert df_dev_pred.columns.tolist() == ["id", "label", "target", "probability", "threshold", "prediction"]
    yield df_dev_pred


def test_prediction(classifier):
    comment = Comment(id="abc", text="love love love")
    prediction = classifier.predict(comment)

    assert isinstance(prediction, Prediction)
    assert prediction.id == "abc"
    # assert "love" in prediction.labels


def test_labels(classifier: Model, df_dev_pred: pd.DataFrame):
    label2threshold_in_data = (
        df_dev_pred[["label", "threshold"]].drop_duplicates().set_index("label").to_dict()["threshold"]
    )
    label2threshold_on_api = classifier.thresholds.as_dict()

    for label, threshold_on_api in label2threshold_on_api.items():
        threshold_in_data = label2threshold_in_data[label]
        if threshold_on_api != threshold_in_data:
            raise Exception(f"{label}: api {threshold_on_api} != data {threshold_in_data}")
        assert threshold_on_api == threshold_in_data


def test_prediction_coincidence(classifier: Model, df_dev: pd.DataFrame, df_dev_pred: pd.DataFrame):
    """
    This test checks if the model artifact in local can reproduce the predictions made by the training pipeline.
    """
    # DataFrame[id, label, probability_test]
    df_api_probability = (
        pd.DataFrame(classifier.predict_proba(df_dev["text"]), index=df_dev.index, columns=classifier.emotions)
        .assign(id=df_dev["id"])
        .melt(id_vars="id", var_name="label", value_name="probability_test")
    )

    ## Attach the model prediction to the long table df_dev_pred
    ## If IDs of the both data sets agree, then there is no data lost.
    df_dev_pred = df_dev_pred.merge(df_api_probability, how="outer")
    assert df_dev_pred.isna().sum().sum() == 0

    abs_delta_predictions = (df_dev_pred["probability"] - df_dev_pred["probability_test"]).abs()
    assert np.all(abs_delta_predictions < 1e-4)

    ## The API must return the same prediction as ones we computed in evaluation stage.
    df_predicted_labels = df_dev_pred[df_dev_pred["prediction"] == 1].copy()
    for _, doc in df_dev.iterrows():
        doc_id = doc["id"]
        comment = Comment(id=doc_id, text=doc["text"])
        prediction = classifier.predict(comment)
        predicted_labels_on_api = set(prediction.labels)
        predicted_labels_in_data = set(df_predicted_labels.query("id == @doc_id")["label"])
        assert predicted_labels_on_api == predicted_labels_in_data
