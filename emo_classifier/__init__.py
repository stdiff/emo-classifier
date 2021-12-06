from emo_classifier.model import TfidfClassifier
from emo_classifier.artifact import Thresholds


def load_classifier():
    classifier = TfidfClassifier.load()
    classifier.thresholds = Thresholds.load()
    return classifier
