import polars as pl
import logging
from .utils import collect


class Forecast:

    def __init__(
        self,
        fc_path: str,
        ensemble_prefix: str,
        obs_col: str,
        bootstrap_cols: str | list[str],
        load_to_memory: bool,
        deterministic_col: str = "mean_pred",
        engine: str = "auto",
    ):

        self.engine = engine

        if load_to_memory:
            self.fc = pl.read_parquet(fc_path)
            self.columns = self.fc.columns
            self.lazy = False
        else:
            self.fc = pl.scan_parquet(fc_path)
            self.columns = self.fc.collect_schema().names()
            self.lazy = True

        if isinstance(bootstrap_cols, str):
            bootstrap_cols = [bootstrap_cols]
        # Column checks
        self.pred_cols = [x for x in self.columns if ensemble_prefix in x]
        logging.info(f"Found {len(self.pred_cols)} ensembles.")
        assert self.check_cols(bootstrap_cols)
        assert self.check_cols(obs_col)
        assert not self.check_cols("_bootstrap")

        self.obs_col = obs_col
        self.deterministic_col = deterministic_col

        # Generate a single bootstrap column
        self.fc = self.fc.with_columns(
            mean_pred=pl.mean_horizontal(pl.col(self.pred_cols)),
            _bootstrap=pl.concat_str(
                (pl.col(col) for col in bootstrap_cols),
                separator="-_-",
            )
        )

    def check_cols(self, cols: str | list[str]):
        if isinstance(cols, str):
            cols = [cols]
        return all(col in self.columns for col in cols)

    def select(self, *args, **kwargs):
        return self.fc.select(*args, **kwargs)

    def collect(self):
        self.fc = collect(self.fc, self.engine)
        self.lazy = False
