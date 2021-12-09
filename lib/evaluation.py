from typing import Union, Optional

import numpy as np
import pandas as pd
import altair as alt

from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

from emo_classifier.artifact import Thresholds, DevMetrics, TestMetrics
from lib.chart import metrics_scatter_plot, positive_rate_scatter_plot, prediction_bar_chart_by_label


def f1_score(precision: pd.Series, recall: pd.Series):
    return 2 * precision * recall / (precision + recall)


def precision_recall_dataframe(y_true: pd.Series, y_prob: pd.Series):
    precision, recall, threshold = precision_recall_curve(y_true, y_prob)

    df_metrics = pd.DataFrame({"threshold": threshold, "precision": precision[:-1], "recall": recall[:-1]})
    df_metrics["f1_score"] = f1_score(df_metrics["precision"], df_metrics["recall"])

    return df_metrics


class PredictionOnDevSetEvaluator:
    """pick the thresholds and do error analysis"""

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
        self._metrics_dataframe: Optional[pd.DataFrame] = None

    @property
    def metrics_dataframe(self) -> pd.DataFrame:
        """
        :return: DataFrame[label, threshold, precision, recall, f1_score]
        """
        if self._metrics_dataframe is None:
            columns = ["label", "threshold", "precision", "recall", "f1_score"]
            df = pd.DataFrame(columns=columns)

            for label in self.labels:
                y_true = self.Y_true[label]
                y_prob = self.Y_prob[label]
                df_pr = precision_recall_dataframe(y_true, y_prob)
                df_pr["label"] = label
                df = df.append(df_pr)

            self._metrics_dataframe = df

        return self._metrics_dataframe

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
                self.metrics_dataframe.groupby("label", as_index=False).apply(pick_best_row).reset_index(drop=True)
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

        dev_metrics = DevMetrics(macro_f1_score=float(self.macro_f1_score()), scores=scores)
        dev_metrics.save()


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
        test_metrics = TestMetrics(macro_f1_score=self.macro_f1_score())
        test_metrics.save()
