import polars as pl
from pfceval.utils import collect, collect_all


def absolute_error(pred_col: str, obs_col: str) -> pl.Expr:
    """
    Calculates the absolute error between predictions and observations.

    Args:
        pred_col (str): Name of the prediction column.
        obs_col (str): Name of the observation column.

    Returns:
        pl.Expr: Polars expression for absolute error.
    """
    return (pl.col(pred_col) - pl.col(obs_col)).abs()


def squared_error(pred_col: str, obs_col: str) -> pl.Expr:
    """
    Calculates the squared error between predictions and observations.

    Args:
        pred_col (str): Name of the prediction column.
        obs_col (str): Name of the observation column.

    Returns:
        pl.Expr: Polars expression for squared error.
    """
    return (pl.col(pred_col) - pl.col(obs_col))**2


def variance(pred_cols: list[str]) -> pl.Expr:
    """
    Calculates the ensemble variance (variance of predictions).

    Args:
        pred_cols (list[str]): List of prediction column names.

    Returns:
        pl.Expr: Polars expression for ensemble variance.
    """
    return pl.concat_list(pred_cols).list.var()


def crps(pred_cols: list[str], obs_col: str) -> pl.Expr:
    """
    Calculates the Continuous Ranked Probability Score (CRPS).

    Args:
        pred_cols (list[str]): List of prediction column names.
        obs_col (str): Name of the observation column.

    Returns:
        pl.Expr: Polars expression for CRPS.
    """
    exX = pl.mean_horizontal((pl.col(pred_cols) - pl.col(obs_col)).abs())
    eXXp = pl.mean_horizontal([
        (pl.col(x) - pl.col(y)).abs() for x in pred_cols for y in pred_cols
    ])
    return exX - 0.5*eXXp


def twcrps(pred_cols: list[str], obs_col: str, th: float) -> pl.Expr:
    """
    Calculates the Threshold-Weighted CRPS (TWCRPS).

    Args:
        pred_cols (list[str]): List of prediction column names.
        obs_col (str): Name of the observation column.
        th (float): The threshold value.

    Returns:
        pl.Expr: Polars expression for TWCRPS.
    """
    clipepd_preds = pl.col(pred_cols).clip(th)
    clipped_obs = pl.col(obs_col).clip(th)
    exX = pl.mean_horizontal((clipepd_preds - clipped_obs).abs())
    eXXp = pl.mean_horizontal([
        (pl.col(x).clip(th) - pl.col(y).clip(th)).abs()
        for x in pred_cols for y in pred_cols
    ])
    return exX - 0.5*eXXp


def brier_score(pred_cols: list[str], obs_col: str, th: float) -> pl.Expr:
    """
    Calculates the Brier Score for a given threshold.

    Args:
        pred_cols (list[str]): List of prediction column names.
        obs_col (str): Name of the observation column.
        th (float): The threshold value.

    Returns:
        pl.Expr: Polars expression for Brier Score.
    """
    probs = pl.mean_horizontal(pl.col(pred_cols).gt(th))
    obs = pl.col(obs_col).gt(th)
    return (probs - obs)**2


def brier_decomposition(
    fc: pl.LazyFrame, pred_cols: list[str], obs_col: str, th: float, engine: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Performs the Brier Score decomposition into reliability, resolution,
    and uncertainty components.

    Args:
        fc (pl.LazyFrame): The forecast data LazyFrame.
        pred_cols (list[str]): List of prediction column names.
        obs_col (str): Name of the observation column.
        th (float): The threshold value.
        engine (str): Polars engine for collection.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - Decomposition components (reliability, resolution, uncertainty).
            - Conditional observed probabilities.
    """
    probs = pl.mean_horizontal(pl.col(pred_cols).gt(th))
    obs = pl.col(obs_col).gt(th)
    mean_obs = collect(fc.select(obs.mean()), engine).item()
    n = collect(fc.select(pl.len()), engine).item()

    temp = fc.select(prob=probs, obs=obs)
    obs_bar = temp.group_by("prob").agg(
        count=pl.col("prob").count(),
        obs_bar=pl.col("obs").mean()
    ).sort("prob")

    decomp = obs_bar.select(
        reliability=(
            (
                pl.col("count")/n
                * ((pl.col("prob") - pl.col("obs_bar"))**2)
            ).sum()
        ),
        resolution=(
            (pl.col("count")/n * (pl.col("obs_bar") - mean_obs)**2).sum()
        ),
        uncertainty=mean_obs * (1-mean_obs),
    )
    return collect_all([decomp, obs_bar], engine)


def group_brier_decomposition(
    fc: pl.LazyFrame,
    preds_cols: list[str],
    obs_col: str,
    groupby_cols: str | list[str],
    th: float,
    engine: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Performs Brier Score decomposition grouped by specified columns.

    Args:
        fc (pl.LazyFrame): The forecast data LazyFrame.
        preds_cols (list[str]): List of prediction column names.
        obs_col (str): Name of the observation column.
        groupby_cols (str | list[str]): Column(s) to group by.
        th (float): The threshold value.
        engine (str): Polars engine for collection.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: A tuple containing:
            - Grouped decomposition components.
            - Grouped conditional observed probabilities.
    """
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    probs = pl.mean_horizontal(pl.col(preds_cols).ge(th)) # TODO: Make it gt(th)
    obs = pl.col(obs_col).ge(th)
    group_info = (
        fc
        .group_by(groupby_cols)
        .agg(
            n_group=pl.len(),
            mean_group=obs.mean()
        )
    )

    temp = fc.with_columns(prob=probs, obs=obs)
    obs_bar = (
        temp
        .group_by(["prob"] + groupby_cols)
        .agg(
            count=pl.col("prob").count(),
            obs_bar=pl.col("obs").mean(),
        )
        .sort(groupby_cols + ["prob"])
        .join(group_info, on=groupby_cols)
    )

    decomp = (
        obs_bar
        .group_by(groupby_cols)
        .agg(
            reliability=(
                (
                    pl.col("count")/pl.col("n_group")
                    * ((pl.col("prob") - pl.col("obs_bar"))**2)
                ).sum()
            ),
            resolution=(
                (
                    pl.col("count")/pl.col("n_group")
                    * (pl.col("obs_bar") - pl.col("mean_group"))**2
                ).sum()
            ),
            uncertainity=(
                (pl.col("mean_group") * (1-pl.col("mean_group"))).mean()
            ),
        )
        .sort(groupby_cols)
    )

    return collect_all([decomp, obs_bar], engine)