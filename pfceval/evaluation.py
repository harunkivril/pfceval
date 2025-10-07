import os
import polars as pl
import json
import logging

from pfceval.calculator import Calculator
from pfceval.utils import ensure_duration_lead_time

class Evaluation:
    """
    Manages and stores forecast evaluation results. It can save results
    to and load from JSON, supporting Polars DataFrames within the JSON.

    Attributes:
        experiment_name (str): Name of the evaluation experiment.
        results (dict): Dictionary storing evaluation tables and their
                        metadata.
        lead_time_col (str): Column name for lead time.
        location_id_col (str): Column name for location identifier.
    """

    def __init__(
            self, experiment_name: str, lead_time_col: str,
            location_id_col: str, results: dict, data_size: int | None = None
    ):
        """
        Initializes an Evaluation object.

        Args:
            experiment_name (str): Name of the experiment.
            lead_time_col (str): Column for lead time.
            location_id_col (str): Column for location ID.
            results (dict): Initial dictionary of results.
        """
        self.experiment_name = experiment_name
        self.results = results
        self.lead_time_col = lead_time_col
        self.location_id_col = location_id_col
        self.data_size = data_size

    def save_results(self, base_path: str):
        """
        Saves evaluation results. Metadata is saved to a JSON file,
        and each Polars DataFrame is saved as a separate Parquet file
        within a dedicated subdirectory.

        Args:
            base_path (str): The base directory for saving.
                             E.g., "my_experiment_results" will create
                             "my_experiment_results/" folder
                             "my_experiment_results/metadata.json" and a folder
                             "my_experiment_results/data/".
        """
        # Determine paths for the manifest file and data directory
        metadata_path = os.path.join(base_path, "metadata.json")
        data_dir_path = os.path.join(base_path, "data")
        os.makedirs(data_dir_path, exist_ok=True)

        serializable_results = {
            "__class_meta": {
                "experiment_name": self.experiment_name,
                "lead_time_col": self.lead_time_col,
                "location_id_col": self.location_id_col,
                "data_size": self.data_size,
            }
        }

        for table_name, table_data in self.results.items():
            current_metadata = table_data["metadata"].copy()
            values = table_data["values"]

            if isinstance(values, (pl.DataFrame, pl.LazyFrame)):
                data_file_name = f"{table_name}.parquet"
                data_full_path = os.path.join(data_dir_path, data_file_name)
                
                values.write_parquet(data_full_path)
                
                # Store relative path to data file and its type in metadata
                current_metadata["_data_file"] = data_file_name
                current_metadata["_data_type"] = "polars_parquet"
                serializable_results[table_name] = {"metadata": current_metadata}
            else:
                # For non-DataFrame items, store them directly in the manifest
                serializable_results[table_name] = table_data


        with open(metadata_path, "w", encoding="utf8") as file:
            json.dump(serializable_results, file, indent=4)
        logging.info(
            f"Evaluation results saved to {base_path}")


    @classmethod
    def load_report(cls, base_path: str) -> 'Evaluation':
        """
        Loads evaluation results from a JSON manifest and associated Parquet files.

        Args:
            base_path (str): The base directory or file prefix used for saving.
                             E.g., "my_experiment_results" to load from
                             "my_experiment_results/metadata.json" and
                             "my_experiment_results/data/".

        Returns:
            Evaluation: A new Evaluation object with loaded results.
        
        Raises:
            FileNotFoundError: If the metedata file is not found.
        """
        metadata_path = os.path.join(base_path, "metadata.json")
        data_dir_path = os.path.join(base_path, "data")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at: {metadata_path}")

        with open(metadata_path, "r", encoding="utf8") as file:
            serializable_results = json.load(file)

        class_meta = serializable_results.pop("__class_meta")

        results = {}
        for table_name, table_info in serializable_results.items():
            metadata = table_info.get("metadata", {})
            data_file = metadata.pop("_data_file", None)
            data_type = metadata.pop("_data_type", None)

            if data_file and data_type == "polars_parquet":
                full_data_file_path = os.path.join(data_dir_path, data_file)
                if os.path.exists(full_data_file_path):
                    # Load as LazyFrame by default for efficiency
                    values = pl.read_parquet(full_data_file_path)
                    results[table_name] = {
                        "values": values, "metadata": metadata}
                else:
                    logging.warning(
                        f"Data file '{full_data_file_path}' for table"
                        f" '{table_name}' not found. Skipping this table."
                    )
            elif "values" in table_info:
                 # For non-DataFrame items stored directly
                 results[table_name] = table_info
            else:
                logging.warning(
                    f"Skipping table '{table_name}': Neither a data file nor "
                    "direct values found in manifest."
                )

        obj = cls(results=results, **class_meta)
        logging.info(f"Evaluation results loaded from {base_path}.")
        return obj

    @classmethod
    def fill_evaluation(
        cls, calculator: Calculator, experiment_name: str,
        lead_time_col: str, location_id_col: str,
        bootstrap: bool = False, n_iter: int = 300, CI: float = 0.9,
        location_metrics: bool = True
    ) -> 'Evaluation':
        """
        Fills an Evaluation object with common metrics and tables
        using a Calculator instance.

        Args:
            calculator (Calculator): An initialized Calculator object.
            experiment_name (str): Name of the experiment.
            lead_time_col (str): Column for lead time.
            location_id_col (str): Column for location ID.
            bootstrap (bool, optional): If True, performs bootstrapped
                                        metrics. Defaults to False.
            n_iter (int, optional): Number of bootstrap iterations.
                                    Defaults to 300.
            CI (float, optional): Confidence interval level for bootstrap.
                                  Defaults to 0.9.
            location_metrics (bool, optional): If True, computes
                                               location-specific metrics.
                                               Defaults to True.

        Returns:
            Evaluation: A new Evaluation object populated with results.
        """
        available_metrics = [
            x.replace("absolute_error", "mae").replace("squared_error", "mse")
            for x in calculator.added_metrics
        ]
        obj = cls(
            experiment_name, 
            lead_time_col, 
            location_id_col, 
            results={},
            data_size=calculator.forecast.nrows()
        )
        obj.add_table(
            "overall_metrics",
            calculator.get_metrics(),
            {"metrics": available_metrics}
        )
        # Metrics per group
        obj.add_table(
            "lead_time_metrics",
            calculator.get_metrics(lead_time_col),
            {"metrics": available_metrics, "groupby": lead_time_col}
        )
        
        if location_metrics:
            groupby = [lead_time_col, location_id_col]
            obj.add_table(
                "lead_time_location_metrics",
                calculator.get_metrics(groupby),
                {"metrics": available_metrics, "groupby": groupby}
            )

            obj.add_table(
                "station_meta",
                calculator.get_station_meta(location_id_col),
                {"on": location_id_col} # Metadata about join key
            )

        if bootstrap:
            obj.add_table(
                "bootstraped_lead_time_metrics",
                calculator.bootstrap_metrics(n_iter, lead_time_col, CI),
                {"metrics": available_metrics, "groupby": lead_time_col,
                 "n_iter": n_iter, "CI": CI}
            )
        return obj

    def add_table(self, table_name: str, values: pl.DataFrame | pl.LazyFrame,
                  metadata: dict):
        """
        Adds a table (Polars DataFrame/LazyFrame) and its associated
        metadata to the evaluation results.

        Args:
            table_name (str): Unique name for the table.
            values (pl.DataFrame | pl.LazyFrame): The Polars data to store.
            metadata (dict): A dictionary of metadata for the table.
        """
        values = ensure_duration_lead_time(values, self.lead_time_col)
        if "data_size" not in metadata:
            metadata["data_size"] = self.data_size
        self.results[table_name] = {"values": values, "metadata": metadata}

    def add_brier_decomp(self, calculator: Calculator, n_iter: int,
                         th: float, CI: float):
        """
        Adds bootstrapped Brier decomposition results to the evaluation.

        Args:
            calculator (Calculator): An initialized Calculator object.
            n_iter (int): Number of bootstrap iterations.
            th (float): Threshold for Brier score.
            CI (float): Confidence interval level.
        """
        decomp, obs_bar = calculator.get_bootstrapped_brier_decomp(
            n_iter=n_iter, th=th, groupby_cols=self.lead_time_col, CI=CI
        )
        metadata = {
            "metrics": ["reliability", "resolution", "uncertainity"],
            "groupby": self.lead_time_col,
            "th": th,
            "n_iter": n_iter,
            "CI": CI
        }

        self.add_table(
            f"bootstrapped_brier_decomp_th:{th}",
            decomp,
            metadata
        )

        metadata = metadata.copy() # Copy to avoid modifying original dict
        metadata["metrics"] = ["obs_bar"]

        self.add_table(
            f"bootstrapped_obs_bar_th:{th}",
            obs_bar,
            metadata,
        )
    
    def add_rank_histogram(self, calculator: Calculator, n_bins: int,
                           groupby_cols: str | list[str] = None):
        """
        Adds rank histogram results to the evaluation.

        Args:
            calculator (Calculator): An initialized Calculator object.
            n_bins (int): Number of bins for the histogram.
            groupby_cols (str | list[str]): Column(s) to group by.
        """
        if groupby_cols is None:
            groupby_cols = [self.lead_time_col]

        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]

        counts, bins = calculator.get_rank_histogram(
            n_bins=n_bins,
            groupby_cols=groupby_cols,
        )
        metadata = {
            "metrics": ["counts"],
            "bins": bins,
            "groupby": groupby_cols,
        }
        self.add_table("lead_time_rank_histogram", counts, metadata)

    def __getitem__(self, key: str) -> dict:
        """
        Allows dictionary-like access to stored results.

        Args:
            key (str): The name of the table to retrieve.

        Returns:
            dict: The dictionary containing 'values' and 'metadata' for
                  the requested table.
        """
        return self.results[key]

    def tables(self) -> list[str]:
        """
        Returns a list of all table names stored in the evaluation.

        Returns:
            list[str]: Names of all stored tables.
        """
        return list(self.results.keys())
    
    def extend(self, other: 'Evaluation', right_prefix: str = None):
        """
        Extends the current Evaluation object with results from another.
        Ensures consistency in lead/location columns.

        Args:
            other (Evaluation): The other Evaluation object to extend from.
            right_prefix (str, optional): Prefix to add to table names
                                          from 'other' to avoid conflicts.
                                          If None, ensures no name conflicts.
        Raises:
            AssertionError: If lead_time_col or location_id_col differ,
                            or if conflicts exist without a prefix.
        """
        assert self.lead_time_col == other.lead_time_col, \
            "Lead time columns must match for extension."
        assert self.location_id_col == other.location_id_col, \
            "Location ID columns must match for extension."

        if not right_prefix:
            assert not set(self.tables()).intersection(set(other.tables())),\
                "Table names conflict. Use 'right_prefix' or rename."
            self.results = {**self.results, **other.results}
        else:
            for table in other.tables():
                # Copying each item to ensure independence
                self.results[f"{right_prefix}_{table}"] = other[table].copy()
