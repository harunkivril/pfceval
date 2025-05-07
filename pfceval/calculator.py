import polars as pl
from .data import Forecast
from .metrics import (
    absolute_error, squared_error, spread, crps, twcrps, brier_score
)
from .utils import collect


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

        self.special_metric_data = None

    def add_metric(self, name, expression):
        if name not in self.added_metrics:
            self.metrics_df = pl.concat([
                self.metrics_df, self.forecast.select(expression.alias(name))
            ], how="horizontal")
            self.added_metrics.append(name)

    def add_absolute_error(self):
        expression = absolute_error(
            self.forecast.deterministic_col, self.forecast.obs_col
        )
        self.add_metric("absolute_error", expression)

    def add_squared_error(self):
        expression = squared_error(
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
        expression = spread(self.forecast.pred_cols)
        self.add_metric("spread", expression)

    def add_crps(self):
        expression = crps(self.forecast.pred_cols, self.forecast.obs_col)
        self.add_metric("crps", expression)

    def add_twcrps(self, th):
        expression = twcrps(
            self.forecast.pred_cols, self.forecast.obs_col, th=th)
        self.add_metric(f"twcrps_th_{th}", expression)

    def add_brier(self, th):
        expression = brier_score(
            self.forecast.pred_cols, self.forecast.obs_col, th=th)
        self.add_metric(f"brier_th_{th}", expression)

    def get_metrics(self, groupby_cols):
        agg_metrics = (
            self.metrics_df
            .group_by(groupby_cols)
            .agg(pl.col(self.added_metrics).mean())
            .sort(groupby_cols)
            .rename(lambda x: x.replace("absolute_error", "mae"))
            .rename(lambda x: x.replace("squared_error", "mse"))
        )
        return collect(agg_metrics, self.forecast.engine)

    def bootstrap_all(self, n_iter, groupby_cols, CI=0.9):
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
