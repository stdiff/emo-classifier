from emo_classifier import load_classifier
from emo_classifier.artifact import Thresholds
from emo_classifier.api import Comment, Prediction


def test_load_classifier():
    classifier = load_classifier()
    assert isinstance(classifier.thresholds, Thresholds)

    comment = Comment(id="abc", text="love love love")
    prediction = classifier.predict(comment)

    assert isinstance(prediction, Prediction)
    assert prediction.id == "abc"
    assert "love" in prediction.labels
