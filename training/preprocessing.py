from typing import Optional

import pandas as pd
import altair as alt

from emo_classifier import setup_logger
from emo_classifier.emotion import load_emotions, vectorize_series_of_emotions
from emo_classifier.classifiers.text import SpacyEnglishTokenizer
from training import LocalPaths
from training.chart import correlation_heatmap

LazyDataFrame = Optional[pd.DataFrame]
logger = setup_logger(__name__)
local_paths = LocalPaths()


def load_data(file_name: str, emotions: list[str]) -> pd.DataFrame:
    """
    :param file_name: file name (not a path)
    :param emotions: list of target variables
    :return: DataFrame[text, admiration, amusement, ..., neutral]. id is the index.
    """
    df = pd.read_parquet(local_paths.dir_datasets / file_name)
    df = (
        pd.concat([df, vectorize_series_of_emotions(df["emotions"], emotions=emotions)], axis=1)
        .set_index("id")
        .drop("emotions", axis=1)
    )
    logger.info(f"LOADED: file = {file_name}, shape = {df.shape}")
    return df


def _compute_label_proportion(data: pd.DataFrame, emotions: list[str]) -> pd.DataFrame:
    """
    emotion -> positive rate

    :param data: DataFrame[emotion0, ...]. Each column for emotion is binary
    :return: DataFrame[emotion, proportion, count]
    """
    df_label_proportion = (
        pd.concat(
            [
                data[emotions].mean().rename("proportion"),
                data[emotions].sum().rename("count"),
            ],
            axis=1,
        )
        .sort_values(by="count", ascending=False)
        .reset_index()
        .rename({"index": "emotion"}, axis=1)
    )

    return df_label_proportion


def _bar_chart_label_proportion_base(data: pd.DataFrame) -> alt.Chart:
    """
    Only for the training set.
    For dev/test set there is a scatter plot for positive rate

    :param data: DataFrame[emotion, proportion, count]. emotion is PK
    """
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("proportion:Q", axis=alt.Axis(format="%")),
            y=alt.Y("emotion:N", sort="-x"),
            color=alt.Color("emotion:N", legend=None),
            tooltip=["emotion:N", alt.Tooltip("proportion:Q", format="0.2%"), alt.Tooltip("count:Q")],
        )
    )
    return chart


def _count_emotions(data: pd.DataFrame, emotions: list[str]) -> pd.DataFrame:
    """
    Count documents with 1 label, 2 labels, 3 labels and so on.

    :param data: DataFrame[emotion0, ...]
    :param emotions: list of emotions
    :return: DataFrame[n_label, n_text, proportion]
    """
    df_count_emotions = (
        data[emotions]
        .sum(axis=1)
        .value_counts()
        .reset_index()
        .rename({"index": "n_label", 0: "n_text"}, axis=1)
        .assign(proportion=lambda df: df["n_text"] / len(data))
    )
    return df_count_emotions


def _chart_count_emotions(data: pd.DataFrame) -> alt.Chart:
    """
    :param data: DataFrame[n_label, n_text, proportion]
    :return: Bar chart of number of texts by number of labels
    """
    title_n_label = "number of lables"
    title_n_text = "number of texts"

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("n_text:Q", title=title_n_text, scale=alt.Scale(type="sqrt")),
            y=alt.Y("n_label:N", title=title_n_label),
            color=alt.Color("n_label:N", title=title_n_label, legend=None),
            tooltip=[
                alt.Tooltip("n_label", title=title_n_label),
                alt.Tooltip("n_text", title=title_n_text),
                alt.Tooltip("proportion:Q", format="0.2%"),
            ],
        )
    )
    return chart


class Preprocessor:
    """
    Responsible for loading raw data and creating feature matrix (X) and the target (y).
    """

    emotions = load_emotions()

    def __init__(self, with_lemmtatization: bool = False, min_df: Optional[int] = None):
        self.with_lemmatization = with_lemmtatization
        self.min_df = min_df

        self._df_train: LazyDataFrame = None
        self._df_dev: LazyDataFrame = None
        self._df_test: LazyDataFrame = None
        self._df_positive_rate: LazyDataFrame = None
        self.tokenizer = SpacyEnglishTokenizer(with_lemmatization=with_lemmtatization)

    @property
    def df_train(self) -> pd.DataFrame:
        if self._df_train is None:
            self._df_train = load_data("train.parquet", self.emotions)
        return self._df_train

    @property
    def df_dev(self) -> pd.DataFrame:
        if self._df_dev is None:
            self._df_dev = load_data("dev.parquet", self.emotions)
        return self._df_dev

    @property
    def df_test(self) -> pd.DataFrame:
        if self._df_test is None:
            self._df_test = load_data("test.parquet", self.emotions)
        return self._df_test

    @property
    def df_token(self) -> pd.DataFrame:
        """
        Not cached

        :return: DataFrame[id, token]
        """
        df_token = []

        for doc_id, row in self.df_train.iterrows():
            tokens = set(self.tokenizer(row["text"]))
            for token in tokens:
                df_token.append((doc_id, token))

        df_token = pd.DataFrame(df_token, columns=["id", "token"])

        return df_token

    @property
    def df_positive_rate(self) -> pd.DataFrame:
        """
        DataFrame of positive rate P(positive|token in a text)

        :return: DataFrame[token, n_doc, r_pos, label, rank]
        """
        if self._df_positive_rate is None:

            if self.min_df is None:
                self.min_df = int(0.001 * len(self.df_train))
            print(f"Tokens with document frequency < {self.min_df} will be ignored.")

            df_pos = []
            for emotion in self.emotions:
                gb = self.df_token.merge(self.df_train.reset_index()[["id", emotion]]).groupby("token")
                df_tmp = pd.concat(
                    [gb.size().rename("n_doc"), gb.mean()[emotion].rename("r_pos")], axis=1
                ).reset_index()
                df_tmp["label"] = emotion

                df_tmp.sort_values(by="r_pos", ascending=False, inplace=True)
                df_tmp.query("n_doc > @self.min_df", inplace=True)
                df_tmp["rank"] = df_tmp["r_pos"].rank(method="min", ascending=False)
                df_pos.append(df_tmp)

            self._df_positive_rate = pd.concat(df_pos, axis=0)

        return self._df_positive_rate

    def count_emotions(self) -> pd.DataFrame:
        return _count_emotions(self.df_train, self.emotions)

    def chart_count_emotions(self) -> alt.Chart:
        return _chart_count_emotions(_count_emotions(self.df_train, self.emotions))

    def chart_count_emotions_dev(self) -> alt.Chart:
        return _chart_count_emotions(_count_emotions(self.df_dev, self.emotions))

    @property
    def df_label_proportion(self) -> pd.DataFrame:
        """
        positive rate of an emotion / label

        :return: DataFrame[emotion, proportion]
        """
        return _compute_label_proportion(self.df_train, self.emotions)

    @property
    def df_label_proportion_dev(self) -> pd.DataFrame:
        """
        positive rate of an emotion / label

        :return: DataFrame[emotion, proportion]
        """
        return _compute_label_proportion(self.df_dev, self.emotions)

    def bar_chart_label_proportion(self) -> alt.Chart:
        """
        Only for the training set.
        For dev/test set there is a scatter plot for positive rate
        """
        return _bar_chart_label_proportion_base(self.df_label_proportion)

    def bar_chart_label_proportion_dev(self) -> alt.Chart:
        """
        Only for the training set.
        For dev/test set there is a scatter plot for positive rate
        """
        return _bar_chart_label_proportion_base(self.df_label_proportion_dev)

    def bar_chart_count_docs_by_length(self) -> alt.Chart:
        df_doc_length = (
            self.df_train["text"]
            .apply(lambda text: len(self.tokenizer(text)))
            .rename("n_token")
            .to_frame()
            .groupby("n_token")
            .size()
            .rename("n_doc")
            .reset_index()
            .query("n_doc > 1")
        )
        chart = (
            alt.Chart(df_doc_length)
            .mark_bar()
            .encode(
                x=alt.X("n_token:Q", title="# token"),
                y=alt.Y("n_doc:Q", title="# doc"),
                tooltip=[alt.Tooltip("n_token:Q", title="# token"), alt.Tooltip("n_doc:O", title="# doc")],
            )
            .properties(title="text counts by number of tokens")
        )
        return chart

    def heatmap_label_correlation(self) -> alt.Chart:
        return correlation_heatmap(self.df_train.drop("text", axis=1), annot=False)

    def histogram_positive_rate(self) -> alt.Chart:
        df_pos = self.df_positive_rate.assign(r_pos_rounded=(self.df_positive_rate["r_pos"] * 100).astype(int))
        df_pos_count = df_pos.groupby(["label", "r_pos_rounded"]).size().rename("count").reset_index()
        df_pos_count["r_pos_rounded"] = df_pos_count["r_pos_rounded"] * 0.01

        df_pos_count = df_pos_count.merge(self.df_label_proportion, left_on="label", right_on="emotion").rename(
            {"proportion": "r_label"}, axis=1
        )

        chart = (
            alt.Chart(df_pos_count)
            .mark_bar()
            .encode(
                x=alt.X("r_pos_rounded:Q", axis=alt.Axis(format="%")),
                y=alt.Y("count:Q", scale=alt.Scale(type="sqrt")),
                color=alt.condition(
                    alt.datum.r_pos_rounded > alt.datum.r_label, alt.value("LightCoral"), alt.value("DodgerBlue")
                ),
                facet=alt.Facet("label", columns=6),
            )
            .properties(height=100, width=140, title="Histogram of positive rate")
        )
        return chart

    @property
    def df_signal_words(self) -> pd.DataFrame:
        """
        tokens with top 5% positive rate

        :return: DataFrame[token, n_doc, r_pos, label, rank, r_pos_rounded, threshold]
        """
        df_thresholds = self.df_positive_rate.groupby("label")["r_pos"].quantile(0.95).rename("threshold").reset_index()
        df_signal_words = (
            self.df_positive_rate.merge(df_thresholds)
            .assign(is_signal_word=lambda df: df["r_pos"] > df["threshold"])
            .query("is_signal_word")
            .drop("is_signal_word", axis=1)
        )
        return df_signal_words

    def bar_chart_of_top5_signal_words(self) -> alt.Chart:
        chart = (
            alt.Chart(self.df_signal_words.query("rank <=5"))
            .mark_bar()
            .encode(
                x=alt.X("r_pos:Q", axis=alt.Axis(format="%")),
                y=alt.Y("token:N", title=None, sort="-x"),
                facet=alt.Facet("label:N", columns=5, align="all"),
            )
            .resolve_scale(y="independent")
            .properties(height=90, width=90)
        )
        return chart

    def _split_to_X_and_Y(self, data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        return data["text"], data[self.emotions]

    def get_train_X_and_Y(self) -> tuple[pd.Series, pd.DataFrame]:
        return self._split_to_X_and_Y(self.df_train)

    def get_dev_X_and_Y(self) -> tuple[pd.Series, pd.DataFrame]:
        return self._split_to_X_and_Y(self.df_dev)

    def get_test_X_and_Y(self) -> tuple[pd.Series, pd.DataFrame]:
        return self._split_to_X_and_Y(self.df_test)
