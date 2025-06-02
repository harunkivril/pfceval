import polars as pl
import numpy as np

from .data import Forecast
from . import metrics
from .utils import collect, collect_all


class Calculator:
    def __init__(self, forecast: Forecast, index_cols):

        self.seed = 3136
        index_cols = set(index_cols + [forecast.obs_col, "_bootstrap"])
        self.forecast = forecast
        self.metrics_df = forecast.select(index_cols)
        self.added_metrics = []

        self.unique_bootstrap = collect(
            forecast.select(pl.col("_bootstrap").unique()))
        self.n_bootstrap = collect(
            self.unique_bootstrap.select(pl.len()), forecast.engine
        )

    def add_metric(self, name, expression):
        if name not in self.added_metrics:
            self.metrics_df = pl.concat([
                self.metrics_df, self.forecast.select(expression.alias(name))
            ], how="horizontal")
            self.added_metrics.append(name)

    def add_absolute_error(self):
        expression = metrics.absolute_error(
            self.forecast.deterministic_col, self.forecast.obs_col
        )
        self.add_metric("absolute_error", expression)

    def add_squared_error(self):
        expression = metrics.squared_error(
            self.forecast.deterministic_col, self.forecast.obs_col
        )
        self.add_metric("squared_error", expression)

    def add_rmse(self):
        if "squared_error" not in self.added_metrics:
            self.add_squared_error()

    def add_mae(self):
        if "absolute_error" not in self.added_metrics:
            self.add_absolute_error()

    def add_spread(self):
        expression = metrics.spread(self.forecast.pred_cols)
        self.add_metric("spread", expression)

    def add_crps(self):
        expression = metrics.crps(
            self.forecast.pred_cols, self.forecast.obs_col)
        self.add_metric("crps", expression)

    def add_twcrps(self, th):
        expression = metrics.twcrps(
            self.forecast.pred_cols, self.forecast.obs_col, th=th)
        self.add_metric(f"twcrps_th:{th}", expression)

    def add_brier(self, th):
        expression = metrics.brier_score(
            self.forecast.pred_cols, self.forecast.obs_col, th=th)
        self.add_metric(f"brier_th:{th}", expression)

    def get_rank_histogram(self, n_bins, groupby_cols=None):
        cols_to_rank =  [self.forecast.obs_col] + self.forecast.pred_cols
        rank_exp = (pl.concat_list(cols_to_rank)
            .list.eval(pl.element().rank())
            .list.first()
            .alias("counts") - 1)
        if groupby_cols:
            ranks = self.forecast.fc.select(groupby_cols, rank_exp)
        else:
            ranks =  self.forecast.fc.select(rank_exp)

        n_ens = len(self.forecast.pred_cols)
        step = n_ens/n_bins
        bins = (
            [-1e-5] + list(x*step - 1e-5 for x  in range(1, n_bins)) + [n_ens]
        )
        labels = np.arange(0, len(bins)) * step
        if groupby_cols is None:
            ranks = ranks.with_columns(group=pl.lit("all"))
            groupby_cols = "group"
        
        hist_vals = (
            ranks
            .group_by(groupby_cols)
            .agg(pl.col("counts").hist(bins=bins))
        )
        
        return hist_vals, labels

    def get_brier_decomp(self, th, groupby_cols=None):
        if groupby_cols is None:
            return metrics.brier_decomposition(
                self.forecast.fc,
                self.forecast.pred_cols,
                self.forecast.obs_col,
                th,
                self.forecast.engine
            )

        return metrics.group_brier_decomposition(
                self.forecast.fc,
                self.forecast.pred_cols,
                self.forecast.obs_col,
                groupby_cols,
                th,
                self.forecast.engine,
        )

    def get_bootstrapped_brier_decomp(self, n_iter, th, groupby_cols, CI=0.9):

        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]

        all_decomps, all_obs_bars = [], []
        for iteration in range(n_iter):
            selected = self.unique_bootstrap.select(
                pl.col("_bootstrap").sample(
                    n=self.n_bootstrap,
                    with_replacement=True,
                    seed=self.seed
                )
            )

            temp = (
                self.forecast.fc
                .lazy()
                .filter(pl.col("_bootstrap").is_in(selected))
            )
            decomp, obs_bar = metrics.group_brier_decomposition(
                temp,
                self.forecast.pred_cols,
                self.forecast.obs_col,
                groupby_cols,
                th,
                self.forecast.engine,
            )

            decomp = decomp.with_columns(iteration=iteration)
            obs_bar = obs_bar.with_columns(iteration=iteration)
            all_decomps.append(decomp)
            all_obs_bars.append(obs_bar)
            self.seed += 1

        uq = (1 + CI)/2
        lq = (1 - CI)/2
        uq_suffix = f"_q{round(uq*100):03}"
        lq_suffix = f"_q{round(lq*100):03}"
        decomp_cols = ["reliability", "resolution", "uncertainity"]

        all_decomps = (
            pl.concat(all_decomps)
            .group_by(groupby_cols)
            .agg(
                pl.col(decomp_cols).quantile(uq).name.suffix(uq_suffix),
                pl.col(decomp_cols).mean().name.suffix("_mean"),
                pl.col(decomp_cols).quantile(lq).name.suffix(lq_suffix),
            )
            .sort(groupby_cols)
        )

        all_obs_bars = (
            pl.concat(all_obs_bars)
            .group_by(["prob"] + groupby_cols)
            .agg(
                pl.col("obs_bar").quantile(uq).name.suffix(uq_suffix),
                pl.col("obs_bar").mean().name.suffix("_mean"),
                pl.col("obs_bar").quantile(lq).name.suffix(lq_suffix),
                pl.col("count").mean().round(),
                pl.col("mean_group").mean(),
            )
            .sort(["prob"] + groupby_cols)
        )

        all_decomps, all_obs_bars = collect_all(
            [all_decomps, all_obs_bars], engine=self.forecast.engine)

        col_order = sorted(
            x for x in all_decomps.columns if x not in groupby_cols
        )
        all_decomps = all_decomps.select(pl.col(groupby_cols + col_order))
        return all_decomps, all_obs_bars

    def get_metrics(self, groupby_cols=None):
        if groupby_cols is None:
            return collect(
                self.metrics_df
                .select(pl.col(self.added_metrics).mean())
                .rename(lambda x: x.replace("absolute_error", "mae"))
                .rename(lambda x: x.replace("squared_error", "mse")),
                self.forecast.engine
            )
        return collect(
            self.metrics_df
            .group_by(groupby_cols)
            .agg(pl.col(self.added_metrics).mean())
            .sort(groupby_cols)
            .rename(lambda x: x.replace("absolute_error", "mae"))
            .rename(lambda x: x.replace("squared_error", "mse")),
            self.forecast.engine
        )
    
    def get_station_meta(self, station_id_col):
        return collect(
            self.forecast.fc
            .select(pl.col([station_id_col, "latitude", "longitude"]))
            .group_by(station_id_col).first()
        )

    def bootstrap_metrics(self, n_iter, groupby_cols, CI=0.9):
        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]

        self.metrics_df = collect(self.metrics_df, engine=self.forecast.engine)
        all_iter = []
        for iteration in range(n_iter):
            selected = self.unique_bootstrap.select(
                pl.col("_bootstrap").sample(
                    n=self.n_bootstrap,
                    with_replacement=True,
                    seed=self.seed
                )
            )
            lazy_df = (
                self.metrics_df
                .lazy()
                .with_columns(iteration=iteration)
                .filter(pl.col("_bootstrap").is_in(selected))
                .group_by(groupby_cols)
                .agg(pl.col(self.added_metrics).mean())
            )
            all_iter.append(lazy_df)
            self.seed += 1

        uq = (1 + CI)/2
        lq = (1 - CI)/2
        uq_suffix = f"_q{round(uq*100):03}"
        lq_suffix = f"_q{round(lq*100):03}"

        all_iter = (
            pl.concat(all_iter)
            .group_by(groupby_cols)
            .agg(
                pl.col(self.added_metrics).quantile(uq).name.suffix(uq_suffix),
                pl.col(self.added_metrics).mean().name.suffix("_mean"),
                pl.col(self.added_metrics).quantile(lq).name.suffix(lq_suffix),
            )
            .sort(groupby_cols)
            .rename(lambda x: x.replace("absolute_error", "mae"))
            .rename(lambda x: x.replace("squared_error", "mse"))
        )

        all_iter = collect(all_iter, engine=self.forecast.engine)
        col_order = sorted(
            x for x in all_iter.columns if x not in groupby_cols
        )
        all_iter = all_iter.select(pl.col(groupby_cols + col_order))
        return all_iter
