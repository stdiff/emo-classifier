"""
This script is the entry point of a SageMaker TrainingJob for TFIDF
"""
import shutil
from typing import Optional
from datetime import datetime

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from training import TrainerBase, LocalPaths
from training.preprocessing import Preprocessor
from emo_classifier.classifiers.tfidf import TfidfClassifier
from emo_classifier.metrics import TrainingMetrics

local_paths = LocalPaths()


class TfidfTrainer(TrainerBase):
    def __init__(self, min_df: int = 10):
        super().__init__()
        self.classifier = TfidfClassifier(min_df=min_df)
        self.training_metrics: Optional[TrainingMetrics] = None

    def fit(self, X_train: pd.Series, Y_train: pd.DataFrame):
        X_vectorized = self.classifier.tfidf_vectorizer.fit_transform(X_train)

        plr = GridSearchCV(
            OneVsRestClassifier(LogisticRegression(random_state=0, solver="liblinear", fit_intercept=False)),
            param_grid={"estimator__C": [1.0, 10, 100]},
            cv=5,
            scoring="f1_macro",
            return_train_score=True,
            n_jobs=4,
        )

        plr.fit(X_vectorized, Y_train)
        self.classifier.model = plr.best_estimator_
        self.logger.info("Training finished")

        cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time"]
        df_cv_results = pd.DataFrame(plr.cv_results_)[cols].sort_values(by="rank_test_score")
        s_best_result = df_cv_results.iloc[0, :]

        self.training_metrics = TrainingMetrics(
            model_class=type(self).__name__,
            model_name=type(self.classifier).__name__,
            best_params=plr.best_params_,
            validation_score=s_best_result["mean_test_score"],
            training_score=s_best_result["mean_train_score"],
            training_timestamp=datetime.now().astimezone().isoformat(),
        )
        self.logger.info(f"Best parameter: {self.training_metrics.best_params}")
        self.logger.info(f"CV score: {self.training_metrics.validation_score}")


def start_training_tfidf_model():
    preprocessor = Preprocessor(with_lemmtatization=False)
    X_train, Y_train = preprocessor.get_train_X_and_Y()

    trainer = TfidfTrainer(min_df=43)
    trainer.fit(X_train, Y_train)
    trainer.save_model()

    if local_paths.on_sagemaker:
        for file in local_paths.dir_artifact.iterdir():
            if not file.is_dir():
                shutil.copy(file, local_paths.sm_model_dir)
        for file in local_paths.dir_resources.iterdir():
            if not file.is_dir():
                shutil.copy(file, local_paths.sm_output_data_dir)


if __name__ == "__main__":
    start_training_tfidf_model()
