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


class Preprocessor:
    """
    Responsible for loading raw data and creating feature matrix (X) and the target (y).
    """

    emotions = load_emotions()

    def __init__(self, with_lemmtatization: bool = False):
        self.with_lemmatization = with_lemmtatization

        self._df_train: LazyDataFrame = None
        self._df_dev: LazyDataFrame = None
        self._df_test: LazyDataFrame = None
        self._df_positive_rate: LazyDataFrame = None
        self._df_positive_rate_dev: LazyDataFrame = None
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

    def _list_token_by_doc(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: DataFrame[text, ...]. Its index must be doc_id
        :return: DataFrame[id, token]
        """
        rows_token = []
        for doc_id, row in data.iterrows():
            tokens = set(self.tokenizer(row["text"]))
            for token in tokens:
                rows_token.append((doc_id, token))

        return pd.DataFrame(rows_token, columns=["id", "token"])

    def _compute_positive_rate(self, data_raw: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame of positive rate P(positive|token in a text) for each token

        :param data_raw: DataFrame[emotion1, emotion2, ...]. Its index must be document id
        :return: DataFrame[token, n_doc, r_pos, label, rank]
        """
        df_pos = []
        df_token = self._list_token_by_doc(data_raw)
        min_df = int(0.001 * len(data_raw))

        for emotion in self.emotions:
            gb = df_token.merge(data_raw.reset_index()[["id", emotion]]).groupby("token")
            df_tmp = pd.concat([gb.size().rename("n_doc"), gb.mean()[emotion].rename("r_pos")], axis=1).reset_index()
            df_tmp["label"] = emotion

            df_tmp.sort_values(by="r_pos", ascending=False, inplace=True)
            df_tmp.query("n_doc > min_df", inplace=True)
            df_tmp["rank"] = df_tmp["r_pos"].rank(method="min", ascending=False)
            df_pos.append(df_tmp)

        return pd.concat(df_pos, axis=0)

    @property
    def df_positive_rate(self) -> pd.DataFrame:
        """
        DataFrame of positive rate P(positive|token in a text) in training set

        :return: DataFrame[token, n_doc, r_pos, label, rank]
        """
        if self._df_positive_rate is None:
            self._df_positive_rate = self._compute_positive_rate(self.df_train)

        return self._df_positive_rate

    @property
    def df_positive_rate_dev(self) -> pd.DataFrame:
        """
        DataFrame of positive rate P(positive|token in a text) in dev set

        :return: DataFrame[token, n_doc, r_pos, label, rank]
        """
        if self._df_positive_rate_dev is None:
            self._df_positive_rate_dev = self._compute_positive_rate(self.df_dev)

        return self._df_positive_rate_dev

    def _count_emotions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Count documents with 1 label, 2 labels, 3 labels and so on.

        :param data: DataFrame[emotion0, ...]
        :return: DataFrame[n_label, n_text, proportion]
        """
        df_count_emotions = (
            data[self.emotions]
            .sum(axis=1)
            .value_counts()
            .reset_index()
            .rename({"index": "n_label", 0: "n_text"}, axis=1)
            .assign(proportion=lambda df: df["n_text"] / len(data))
        )
        return df_count_emotions

    def df_count_emotions(self) -> pd.DataFrame:
        return self._count_emotions(self.df_train)

    def df_count_emotions_dev(self):
        return self._count_emotions(self.df_dev)

    @staticmethod
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

    def chart_count_emotions(self) -> alt.Chart:
        return self._chart_count_emotions(self.df_count_emotions())

    def chart_count_emotions_dev(self) -> alt.Chart:
        return self._chart_count_emotions(self.df_count_emotions_dev())

    def _compute_label_proportion(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        emotion -> positive rate

        :param data: DataFrame[emotion0, ...]. Each column for emotion is binary
        :return: DataFrame[emotion, proportion, count]
        """
        df_label_proportion = (
            pd.concat(
                [
                    data[self.emotions].mean().rename("proportion"),
                    data[self.emotions].sum().rename("count"),
                ],
                axis=1,
            )
            .sort_values(by="count", ascending=False)
            .reset_index()
            .rename({"index": "emotion"}, axis=1)
        )

        return df_label_proportion

    @property
    def df_label_proportion(self) -> pd.DataFrame:
        """
        positive rate of an emotion / label

        :return: DataFrame[emotion, proportion]
        """
        return self._compute_label_proportion(self.df_train)

    @property
    def df_label_proportion_dev(self) -> pd.DataFrame:
        """
        positive rate of an emotion / label

        :return: DataFrame[emotion, proportion]
        """
        return self._compute_label_proportion(self.df_dev)

    @staticmethod
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

    def bar_chart_label_proportion(self) -> alt.Chart:
        """
        Only for the training set.
        For dev/test set there is a scatter plot for positive rate
        """
        return self._bar_chart_label_proportion_base(self.df_label_proportion)

    def bar_chart_label_proportion_dev(self) -> alt.Chart:
        """
        Only for the training set.
        For dev/test set there is a scatter plot for positive rate
        """
        return self._bar_chart_label_proportion_base(self.df_label_proportion_dev)

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

    def chart_label_correlation(self) -> alt.Chart:
        return correlation_heatmap(self.df_train.drop("text", axis=1), annot=False)

    def chart_label_correlation_dev(self) -> alt.Chart:
        return correlation_heatmap(self.df_dev.drop("text", axis=1), annot=False)

    @staticmethod
    def _df_signal_words(data_positive_rate: pd.DataFrame) -> pd.DataFrame:
        """
        tokens with top 5% positive rate

        :return: DataFrame[token, n_doc, r_pos, label, rank, r_pos_rounded, threshold]
        """
        df_thresholds = data_positive_rate.groupby("label")["r_pos"].quantile(0.95).rename("threshold").reset_index()
        df_signal_words = (
            data_positive_rate.merge(df_thresholds)
            .assign(is_signal_word=lambda df: df["r_pos"] > df["threshold"])
            .query("is_signal_word")
            .drop("is_signal_word", axis=1)
        )
        return df_signal_words

    @property
    def df_signal_words(self) -> pd.DataFrame:
        """
        tokens with top 5% positive rate in training set

        :return: DataFrame[token, n_doc, r_pos, label, rank, r_pos_rounded, threshold]
        """
        return self._df_signal_words(self.df_positive_rate)

    @property
    def df_signal_words_dev(self) -> pd.DataFrame:
        """
        tokens with top 5% positive rate in dev set

        :return: DataFrame[token, n_doc, r_pos, label, rank, r_pos_rounded, threshold]
        """
        return self._df_signal_words(self.df_positive_rate_dev)

    @staticmethod
    def _chart_top5_signal_words(data_signal_words: pd.DataFrame) -> alt.Chart:
        df_top5 = (
            data_signal_words.groupby("label", as_index=False)
            .apply(lambda df: df.sort_values(by="rank").head(5).assign(rank=range(1, 6)))
            .reset_index(drop=True)
        )

        chart_base = alt.Chart(df_top5)

        chart_heatmap = chart_base.mark_rect().encode(
            x=alt.X("rank:O", axis=alt.Axis(labelAngle=0)),
            y="label",
            color=alt.Color("r_pos", scale=alt.Scale(scheme="greens", domain=[0, 1])),
            tooltip=["label", "token", alt.Tooltip("r_pos", format="0.2%")],
        )

        chart_text = chart_base.mark_text(align="center", size=14).encode(
            x="rank:O",
            y="label",
            text="token",
            color=alt.condition(alt.datum.r_pos > 0.5, alt.value("white"), alt.value("black")),
            tooltip=["label", "token", alt.Tooltip("r_pos", format="0.2%")],
        )

        return chart_heatmap + chart_text

    def chart_top5_signal_words(self) -> alt.Chart:
        return self._chart_top5_signal_words(self.df_signal_words)

    def chart_top5_signal_words_dev(self) -> alt.Chart:
        return self._chart_top5_signal_words(self.df_signal_words_dev)

    def _split_to_X_and_Y(self, data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        return data["text"], data[self.emotions]

    def get_train_X_and_Y(self) -> tuple[pd.Series, pd.DataFrame]:
        return self._split_to_X_and_Y(self.df_train)

    def get_dev_X_and_Y(self) -> tuple[pd.Series, pd.DataFrame]:
        return self._split_to_X_and_Y(self.df_dev)

    def get_test_X_and_Y(self) -> tuple[pd.Series, pd.DataFrame]:
        return self._split_to_X_and_Y(self.df_test)
