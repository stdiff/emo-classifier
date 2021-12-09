from sklearn.linear_model import LogisticRegression

from lib import get_logger
from lib.preprocessing import Preprocessor
from emo_classifier.model import TfidfClassifier

logger = get_logger(__name__)


def train_tfidf(preprocessor: Preprocessor):
    X_train, Y_train = preprocessor.get_train_X_and_Y()
    logger.info(f"X_train.shape: {X_train.shape}")
    logger.info(f"Y_train.shape: {Y_train.shape}")

    tfidf_classifier = TfidfClassifier(tokenizer=preprocessor.tokenizer, min_df=43)
    param_grid = {"C": [0.1, 1.0, 10, 100]}
    tfidf_classifier.set_model(
        "plr", LogisticRegression(random_state=0, solver="liblinear", fit_intercept=False), param_grid
    )
    tfidf_classifier.fit(X_train, Y_train)
    logger.info("Training finished")
    tfidf_classifier.save()


def start():
    preprocessor = Preprocessor(with_lemmtatization=False)
    train_tfidf(preprocessor)


if __name__ == "__main__":
    start()
