import polars as pl

from .utils import collect, collect_all
from .metrics import group_brier_decomposition


def bootstrap_metric_groupby(
        data, metric_f, n_samples, CI, groupby_cols, engine, **kwargs
):
    size = collect(data.select(pl.len()))
    samples = [
        data
        .select(pl.all().sample(n=size, with_replacement=True, seed=i))
        .group_by(groupby_cols)
        .agg([metric_f(**kwargs).alias(f"{i}")])
        for i in range(n_samples)
    ]

    lq = (1 - CI) / 2
    uq = 1 - lq

    fname = metric_f.__name__
    res = (
        pl.concat(samples, how="align")
        .unpivot(index=groupby_cols, value_name=fname)
        .group_by(groupby_cols)
        .agg(
            pl.col(fname).mean().alias(f"{fname}_mean"), 
            pl.col(fname).quantile(uq).alias(f"{fname}_q{round(uq*100):03}"),
            pl.col(fname).quantile(lq).alias(f"{fname}_q{round(lq*100):03}"),
        )
    )
    return collect(res.sort(groupby_cols), engine)


def bootstrap_brier_decomp(
        data, n_iterations, CI, pred_cols, obs_col, th, groupby_cols, engine, lazy
):

    data = collect(data, "auto")

    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    obs_bars = []
    decomps = []
    size = collect(data.select(pl.len()))
    for i in range(n_iterations):
        sample = data.select(
            pl.all().sample(n=size, with_replacement=True, seed=i)
        )
        decomp, obs_bar = group_brier_decomposition(
            sample, pred_cols, obs_col, th, groupby_cols, engine, lazy
        )

        decomps.append(decomp)
        obs_bars.append(obs_bar)

    decomps = pl.concat(decomps)
    obs_bars = pl.concat(obs_bars)

    lq = (1 - CI) / 2
    uq = 1 - lq

    decomps = decomps.group_by(groupby_cols).agg(
            pl.exclude("iter").mean().name.suffix("_mean"), 
            pl.exclude("iter").quantile(uq).name.suffix("_uq"), 
            pl.exclude("iter").quantile(lq).name.suffix("_lq"),
    ).sort(groupby_cols)

    groupby_cols = ["prob"] + groupby_cols
    obs_bars = obs_bars.group_by(groupby_cols).agg(
            pl.col("obs_bar").mean().name.suffix("_mean"), 
            pl.col("obs_bar").quantile(uq).name.suffix("_uq"), 
            pl.col("obs_bar").quantile(lq).name.suffix("_lq"),
    )

    obs_bars = obs_bars.join(
        obs_bar.select(pl.exclude(["obs_bar"])), on=groupby_cols
    ).sort(groupby_cols)

    return collect_all([decomps, obs_bars], engine)
