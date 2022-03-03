from emo_classifier.model import Model
from emo_classifier.metrics import Thresholds

# from emo_classifier.classifiers.tfidf import TfidfClassifier
from emo_classifier.classifiers.embedding_bag import EmbeddingBagClassifier


def load_model() -> Model:
    """
    If you want to use a different Model, you have to instantiate it here.

    :return: Model instance you want to use in production
    """
    return EmbeddingBagClassifier.load()
    # return TfidfClassifier.load()


def load_classifier():
    classifier = load_model()
    classifier.thresholds = Thresholds.load()
    return classifier
