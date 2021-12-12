from emo_classifier.model.tfidf import TfidfClassifier
from emo_classifier.artifact import Thresholds


def load_model():
    return TfidfClassifier.load()


def load_classifier():
    classifier = load_model()
    classifier.thresholds = Thresholds.load()
    return classifier
