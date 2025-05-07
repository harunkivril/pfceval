import polars as pl
from .utils import collect, collect_all


def absolute_error(pred_col, obs_col):
    return (pl.col(pred_col) - pl.col(obs_col)).abs()


def squared_error(pred_col, obs_col):
    return (pl.col(pred_col) - pl.col(obs_col))**2


def spread(pred_cols):
    return pl.concat_list(pred_cols).list.std()


def crps(pred_cols, obs_col):
    exX = pl.mean_horizontal((pl.col(pred_cols) - pl.col(obs_col)).abs())
    eXXp = pl.mean_horizontal([
        (pl.col(x) - pl.col(y)).abs() for x in pred_cols for y in pred_cols
    ])
    return exX - 0.5*eXXp


def twcrps(pred_cols, obs_col, th):
    clipepd_preds = pl.col(pred_cols).clip(th)
    clipped_obs = pl.col(obs_col).clip(th)
    exX = pl.mean_horizontal((clipepd_preds - clipped_obs).abs())
    eXXp = pl.mean_horizontal([
        (pl.col(x).clip(th) - pl.col(y).clip(th)).abs()
        for x in pred_cols for y in pred_cols
    ])
    return exX - 0.5*eXXp


def brier_score(pred_cols, obs_col, th):
    probs = pl.mean_horizontal(pl.col(pred_cols).gt(th))
    obs = pl.col(obs_col).gt(th)
    return (probs - obs)**2


def brier_decomposition(data, pred_cols, obs_col, th, engine):
    probs = pl.mean_horizontal(pl.col(pred_cols).gt(th))
    obs = pl.col(obs_col).gt(th)
    mean_obs = collect(data.select(obs.mean()), engine).item()
    n = collect(data.select(pl.len()), engine).item()

    temp = data.select(prob=probs, obs=obs)
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
        uncertainity=mean_obs * (1-mean_obs),
    )

    return pl.collect_all([obs_bar, decomp])


def group_brier_decomposition(
        data, pred_cols, obs_col, th, groupby_cols, engine, lazy=False
):
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    probs = pl.mean_horizontal(pl.col(pred_cols).gt(th))
    obs = pl.col(obs_col).gt(th)
    group_info = (
        data
        .group_by(groupby_cols)
        .agg(
            n_group=pl.len(),
            mean_group=obs.mean()
        )
    )

    temp = data.with_columns(prob=probs, obs=obs)
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

    if lazy:
        return decomp, obs_bar

    return collect_all([decomp, obs_bar], engine)
