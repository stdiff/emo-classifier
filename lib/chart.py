import altair as alt
import pandas as pd


def correlation_heatmap(data: pd.DataFrame, annot: bool = True) -> alt.Chart:
    df_corr = data.corr().reset_index()

    base = alt.Chart(df_corr).transform_fold(fold=df_corr.columns[1:].tolist(), as_=["variable", "value"])

    heat = base.mark_rect().encode(
        x=alt.X("index:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("variable:N", title=None),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Correlation"),
        tooltip=["index:N", "variable:N", "value:Q"],
    )

    if annot:
        text = base.mark_text(fontSize=16).encode(
            x="index:N",
            y="variable:N",
            text=alt.Text("value:Q", format="0.3f"),
            color=alt.condition(abs(alt.datum.value) > 0.5, alt.value("white"), alt.value("black")),
        )
        return heat + text

    else:
        return heat


def metrics_scatter_plot(metrics_data_by_label: pd.DataFrame) -> alt.Chart:
    """
    :param metrics_data_by_label: DataFrame[label, threshold, precision, recall, f1_score, ...]. label is unique.
    :return: scatter plot of metrics
    """
    chart_base = alt.Chart(metrics_data_by_label)

    tooltip = [
        "label:N",
        alt.Tooltip("precision:Q", format="0.4f"),
        alt.Tooltip("recall:Q", format="0.4f"),
        alt.Tooltip("f1_score:Q", format="0.4f"),
    ]

    chart_scatter = (
        chart_base.mark_point()
        .encode(
            x="precision:Q",
            y="recall:Q",
            color=alt.Color("f1_score:Q", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=tooltip,
        )
        .properties(title="Scatter plot of metrics")
    )

    chart_annotation = chart_base.mark_text(xOffset=5, yOffset=0, align="left").encode(
        x="precision:Q", y="recall:Q", text="label:N", tooltip=tooltip
    )

    chart_diagonal = (
        alt.Chart(alt.sequence(start=0, stop=1, step=0.01, as_="t"))
        .transform_calculate(precision="datum.t", recall="datum.t")
        .mark_line(strokeDash=[5, 5], color="black", size=1)
        .encode(x="precision:Q", y="recall:Q", order="t:Q")
    )
    return chart_diagonal + chart_annotation + chart_scatter


def positive_rate_scatter_plot(data_positive_rate: pd.DataFrame):
    """
    :param data_positive_rate: DataFrame[label, actual_positive_rate, positive_rate, ...]. label is unique.
    :return: scatter plot of positive rates
    """
    chart_base = alt.Chart(data_positive_rate)

    tooltip = ["label", alt.Tooltip("actual_positive_rate", format="0.2%"), alt.Tooltip("positive_rate", format="0.2%")]

    chart_scatter = (
        chart_base.mark_point()
        .encode(
            x=alt.X("actual_positive_rate:Q", scale=alt.Scale(type="log")),
            y=alt.Y("positive_rate:Q", scale=alt.Scale(type="log")),
            color=alt.condition(
                alt.datum.actual_positive_rate > alt.datum.positive_rate, alt.value("SteelBlue"), alt.value("Crimson")
            ),
            tooltip=tooltip,
        )
        .properties(title="Actual positive rate vs positive rate")
    )

    chart_annotation = chart_base.mark_text(xOffset=5, yOffset=0, align="left").encode(
        x="actual_positive_rate:Q", y="positive_rate:Q", text="label", tooltip=tooltip
    )

    min_diagonal = max(data_positive_rate["actual_positive_rate"].min(), data_positive_rate["positive_rate"].min())
    max_diagonal = min(data_positive_rate["actual_positive_rate"].max(), data_positive_rate["positive_rate"].max())
    chart_diagonal = (
        alt.Chart(alt.sequence(start=min_diagonal, stop=max_diagonal + 0.01, step=0.01, as_="t"))
        .transform_calculate(actual_positive_rate="datum.t", positive_rate="datum.t")
        .mark_line(strokeDash=[5, 5], color="black", size=1)
        .encode(x="actual_positive_rate:Q", y="positive_rate:Q", order="t:Q")
    )

    return chart_diagonal + chart_scatter + chart_annotation