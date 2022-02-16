from emo_classifier.classifiers.tfidf import TfidfClassifier, Thresholds


def load_model():
    return TfidfClassifier.load()


def load_classifier():
    classifier = load_model()
    classifier.thresholds = Thresholds.load()
    return classifier
