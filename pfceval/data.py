import polars as pl
import logging
from pfceval.utils import collect
from copy import copy

class Forecast:
    """
    Manages forecast data. Supports in-memory/lazy Polars ops for
    loading, processing, and filtering. Creates unified bootstrap ID.

    Attributes:
        engine (str): Polars execution engine.
        fc (pl.DataFrame | pl.LazyFrame): Underlying forecast data.
        columns (list[str]): All column names.
        lazy (bool): True if LazyFrame, False if DataFrame.
        pred_cols (list[str]): Ensemble prediction col names.
        obs_col (str): Observed values col name.
        deterministic_col (str): Deterministic prediction col (default:
                                 "mean_pred").
    """

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
        """
        Initializes Forecast object by loading data.

        Args:
            fc_path (str): Path to Parquet forecast data.
            ensemble_prefix (str): Prefix for ensemble prediction cols.
            obs_col (str): Column name for observed values.
            bootstrap_cols (str | list[str]): Column(s) for bootstrap ID.
            load_to_memory (bool): If True, loads data into memory.
            deterministic_col (str, optional): Name for deterministic
                                               pred. Defaults to "mean_pred".
            engine (str, optional): Polars engine. Defaults to "auto".
        Raises:
            AssertionError: If cols not found or "_bootstrap" exists.
        """

        self.engine = engine

        if load_to_memory:
            self.forecast = pl.read_parquet(fc_path)
            self.columns = self.forecast.columns
            self.lazy = False
        else:
            self.forecast = pl.scan_parquet(fc_path)
            self.columns = self.forecast.collect_schema().names()
            self.lazy = True

        if isinstance(bootstrap_cols, str):
            bootstrap_cols = [bootstrap_cols]
        # Column checks
        self.pred_cols = [x for x in self.columns if ensemble_prefix in x]
        logging.info(f"Found {len(self.pred_cols)} ensembles.")
        assert self.check_cols(bootstrap_cols), \
            f"Bootstrap col(s) {bootstrap_cols} not found."
        assert self.check_cols(obs_col), \
            f"Obs col '{obs_col}' not found."
        assert not self.check_cols("_bootstrap"), \
            "Col '_bootstrap' already exists. Rename or remove."

        self.obs_col = obs_col
        self.deterministic_col = deterministic_col

        # Generate a single bootstrap column and mean prediction
        self.forecast = self.forecast.with_columns(
            pl.mean_horizontal(pl.col(self.pred_cols)).alias(deterministic_col),
            _bootstrap=pl.concat_str(
                (pl.col(col).cast(str) for col in bootstrap_cols),
                separator="-_-", # Unique separator for bootstrap ID
            )
        )

    def check_cols(self, cols: str | list[str]) -> bool:
        """
        Checks if all specified column(s) exist in forecast data.

        Args:
            cols (str | list[str]): Col name(s) to check.

        Returns:
            bool: True if all cols exist, False otherwise.
        """
        if isinstance(cols, str):
            cols = [cols]
        return all(col in self.columns for col in cols)

    def select(self, *args, **kwargs) -> pl.DataFrame | pl.LazyFrame:
        """
        Proxies Polars `select` method for column selection.

        Returns:
            pl.DataFrame | pl.LazyFrame: Data with selected cols.
        """
        return self.forecast.select(*args, **kwargs)

    def collect(self):
        """
        Materializes Polars LazyFrame to DataFrame.
        Sets `self.lazy` to False after collection.
        """
        if self.lazy:
            self.forecast = collect(self.forecast, self.engine)
            self.lazy = False

    def filter(self, *args, **kwargs) -> 'Forecast':
        """
        Proxies Polars `filter`, returns new Forecast object with
        filtered data. Original object unchanged.

        Returns:
            Forecast: New Forecast instance with filtered data.
        """
        sub_fc = self.forecast.filter(*args, **kwargs)
        filtered = self.copy() # Shallow copy
        filtered.forecast = sub_fc
        return filtered

    def copy(self) -> 'Forecast':
        """
        Creates a shallow copy of the Forecast object. Useful for
        retaining original object when modifying data.

        Returns:
            Forecast: Shallow copy of current Forecast instance.
        """
        return copy(self)

    def nrows(self):
        """
        Get number of rows in the forecast dataframe.

        Returns:
            int: Number of rows
        """
        return collect(self.forecast.select(pl.len())).item()
