from datetime import datetime
from typing import Union, Optional

import numpy as np
import pandas as pd
import altair as alt
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

from emo_classifier.classifiers.metrics import SimpleStats, stats_roc_auc
from emo_classifier.metrics import Thresholds, DevMetrics, TestMetrics
from training import LocalPaths
from training.chart import metrics_scatter_plot, positive_rate_scatter_plot, prediction_bar_chart_by_label


local_paths = LocalPaths()


def f1_score(precision: pd.Series, recall: pd.Series):
    return 2 * precision * recall / (precision + recall)


def precision_recall_dataframe(y_true: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    """
    compute metrics for binary classifications (except accuracy) for each threshold

    :param y_true: series of true binary labels (i.e. 0 or 1)
    :param y_prob: series of probabilities of the positive label
    :return: DataFrame[threshold, precision, recall, f1_score]
    """
    precision, recall, threshold = precision_recall_curve(y_true, y_prob)
    df_metrics = pd.DataFrame({"threshold": threshold, "precision": precision[:-1], "recall": recall[:-1]})
    df_metrics["f1_score"] = f1_score(df_metrics["precision"], df_metrics["recall"])
    return df_metrics


class PredictionOnDevSetEvaluator:
    """pick the thresholds and do error analysis"""

    prediction_parquet_path = local_paths.dir_resources / "dev_predictions.parquet"

    def __init__(self, Y_true: pd.DataFrame, Y_prob: Union[pd.DataFrame, np.ndarray], X_text: pd.Series):
        if not isinstance(Y_true, pd.DataFrame):
            raise ValueError("Y_true must be a DataFrame with index = ids, column = labels")
        elif not isinstance(X_text, pd.Series):
            raise ValueError("X_text must be a Series with index = ids")

        self.Y_true = Y_true
        self.labels = Y_true.columns.tolist()
        self.X_text = X_text

        if isinstance(Y_prob, np.ndarray):
            self.Y_prob = pd.DataFrame(Y_prob, columns=self.Y_true.columns, index=self.Y_true.index)
        else:
            if self.labels != Y_prob.columns.tolist():
                raise ValueError("The columns of the given dataframes must be the same.")

            self.Y_prob = Y_prob

        self._best_thresholds: Optional[pd.Series] = None

    @property
    def _metrics_dataframe(self) -> pd.DataFrame:
        """
        :return: DataFrame[label, threshold, precision, recall, f1_score]
        """
        columns = ["label", "threshold", "precision", "recall", "f1_score"]
        df = pd.DataFrame(columns=columns)

        for label in self.labels:
            y_true = self.Y_true[label]
            y_prob = self.Y_prob[label]
            df_pr = precision_recall_dataframe(y_true, y_prob)
            df_pr["label"] = label
            df = df.append(df_pr)

        return df

    @property
    def best_thresholds(self) -> pd.DataFrame:
        """
        DataFrame of metrics with the best threshold by label

        :return: DataFrame[label, threshold, precision, recall, f1_score, positive_rate, actual_positive_rate]
        """

        def pick_best_row(data: pd.DataFrame) -> pd.DataFrame:
            return data.sort_values(by="f1_score", ascending=False).head(1)

        if self._best_thresholds is None:
            self._best_thresholds: pd.DataFrame = (
                self._metrics_dataframe.groupby("label", as_index=False).apply(pick_best_row).reset_index(drop=True)
            )
            label2threshold = {r.label: r.threshold for r in self._best_thresholds.itertuples()}
            positive_rate = [(self.Y_prob[label] > label2threshold[label]).mean() for label in self.labels]
            self._best_thresholds["positive_rate"] = positive_rate
            self._best_thresholds["actual_positive_rate"] = self.Y_true.mean().values

        return self._best_thresholds

    def thresholds(self) -> "Thresholds":
        return Thresholds.from_pairs(list(zip(self.best_thresholds["label"], self.best_thresholds["threshold"])))

    def metrics_scatter_plot(self) -> alt.Chart:
        return metrics_scatter_plot(self.best_thresholds)

    def positive_rate_scatter_plot(self) -> alt.Chart:
        return positive_rate_scatter_plot(self.best_thresholds)

    def macro_f1_score(self) -> float:
        return self.best_thresholds["f1_score"].mean()

    def macro_recall(self) -> float:
        return self.best_thresholds["recall"].mean()

    def macro_precision(self) -> float:
        return self.best_thresholds["precision"].mean()

    def prediction_bar_chart_by_label(self) -> alt.Chart:
        return prediction_bar_chart_by_label(df_prob=self.Y_prob)

    def _wrong_predictions(self, true_label: int, n: int = 3):
        ascending = true_label == 1
        top_wrong_predictions_by_label = []

        for label in self.labels:
            y_true = self.Y_true[label]
            y_prob = self.Y_prob[label][y_true == true_label]
            y_prob = y_prob.sort_values(ascending=ascending).rename("probability")
            top_wrong_predictions_by_label.append(y_prob.to_frame().head(n).assign(label=label))

        top_wrong_predictions = pd.concat(top_wrong_predictions_by_label, axis=0)
        top_wrong_predictions = pd.merge(
            top_wrong_predictions, self.X_text.to_frame(), left_index=True, right_index=True
        )
        labels = self.Y_true.apply(lambda r: ",".join([k for k in r.index if r[k] == 1]), axis=1).rename("true_labels")
        return top_wrong_predictions.merge(labels.to_frame(), how="left", left_index=True, right_index=True)

    def false_positive_by_label(self, n: int = 3) -> pd.DataFrame:
        """
        indices of negative instances with high score by label
        :return:
        """
        return self._wrong_predictions(true_label=0, n=n)

    def false_negative_by_label(self, n: int = 3):
        """
        indices of positive instances with high score by label
        :return:
        """
        return self._wrong_predictions(true_label=1, n=n)

    def save_dev_metrics(self):
        scores = {}
        for label, metrics in self.best_thresholds.set_index("label").iterrows():
            scores[label] = metrics.to_dict()

        from sklearn.metrics import log_loss
        from datetime import datetime

        metric_stats = {}
        for metric in ["f1_score"]:
            metric_stats[metric] = {"avg"}

        dev_metrics = DevMetrics(
            log_loss=log_loss(self.Y_true.values, self.Y_prob.values),
            auc_roc=stats_roc_auc(self.Y_true.values, self.Y_prob.values).as_dict(),
            f1_score=SimpleStats.from_array(self.best_thresholds["f1_score"]).as_dict(),
            precision=SimpleStats.from_array(self.best_thresholds["precision"]).as_dict(),
            recall=SimpleStats.from_array(self.best_thresholds["recall"]).as_dict(),
            timestamp=datetime.now().astimezone().isoformat(),
        )
        dev_metrics.save()

    def save_prediction(self):
        """
        Save the prediction. The schema must be as follows.

        - id: document id
        - label: label (emotion class)
        - target: if the document belongs to the label (0 or 1)
        - probability: predicted probability
        - threshold: threshold of the label
        - prediction: the prediction says if the document belongs to the label (0 or 1)
        """
        df_prediction = (
            self.Y_true.reset_index()
            .melt(id_vars="id", var_name="label", value_name="target")
            .merge(
                self.Y_prob.reset_index().melt(id_vars="id", var_name="label", value_name="probability"),
            )
            .merge(self.thresholds().as_series().reset_index())
        )
        df_prediction["prediction"] = (df_prediction["probability"] > df_prediction["threshold"]).astype(int)
        df_prediction.to_parquet(self.prediction_parquet_path, index=False)

    def save_thresholds_metrics_and_predictions(self):
        self.thresholds().save()
        self.save_dev_metrics()
        self.save_prediction()


class PredictionOnTestSetEvaluator:
    """No threshold choice, no detailed analysis."""

    def __init__(self, Y_true: pd.DataFrame, Y_prob: Union[pd.DataFrame, np.ndarray], thresholds: Thresholds):
        if not isinstance(Y_true, pd.DataFrame):
            raise ValueError("Y_true must be a DataFrame with index = ids, column = labels")
        self.Y_true = Y_true
        self.labels = Y_true.columns.tolist()

        if isinstance(Y_prob, np.ndarray):
            self.Y_prob = pd.DataFrame(Y_prob, columns=self.Y_true.columns, index=self.Y_true.index)
        else:
            if set(self.labels) != set(Y_prob.columns.tolist()):
                raise ValueError("The columns of the given dataframes must be the same.")
            self.Y_prob = Y_prob

        self.thresholds = thresholds
        self._metrics_by_label: Optional[pd.DataFrame] = None

    @property
    def metrics_by_label(self) -> pd.DataFrame:
        """
        label -> (threshold, precision, recall, f1_score)

        :return: DataFrame[label, threshold, precision, recall, f1_score]
        """
        if self._metrics_by_label is None:
            s_threshold = self.thresholds.as_series()[self.labels]  # ensure the order of the labels
            Y_pred = self.Y_prob >= s_threshold
            precision, recall, f1_score, _ = precision_recall_fscore_support(self.Y_true, Y_pred)
            positive_rate = Y_pred.mean()
            actual_positive_rate = self.Y_true.mean()

            self._metrics_by_label = pd.DataFrame(
                {
                    "label": self.labels,
                    "threshold": s_threshold.values,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "positive_rate": positive_rate,
                    "actual_positive_rate": actual_positive_rate,
                }
            ).reset_index(drop=True)

        return self._metrics_by_label

    def metrics_scatter_plot(self) -> alt.Chart:
        return metrics_scatter_plot(self.metrics_by_label)

    def positive_rate_scatter_plot(self) -> alt.Chart:
        return positive_rate_scatter_plot(self.metrics_by_label)

    def macro_f1_score(self) -> float:
        return self.metrics_by_label["f1_score"].mean()

    def save_test_metrics(self):
        test_metrics = TestMetrics(
            auc_roc=stats_roc_auc(self.Y_true.values, self.Y_prob.values).as_dict(),
            f1_score=SimpleStats.from_array(self.metrics_by_label["f1_score"]).as_dict(),
            precision=SimpleStats.from_array(self.metrics_by_label["precision"]).as_dict(),
            recall=SimpleStats.from_array(self.metrics_by_label["recall"]).as_dict(),
            timestamp=datetime.now().astimezone().isoformat(),
        )
        test_metrics.save()
