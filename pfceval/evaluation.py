import io
import polars as pl
import json

from ast import literal_eval

class Evaluation:

    def __init__(
            self, experiment_name, lead_time_col, location_id_col, results: dict
    ):
        self.experiment_name = experiment_name
        self.results = results
        self.lead_time_col = lead_time_col
        self.location_id_col = location_id_col

    def save_results(self, path):
        serializable_results = {
            "__class_meta": {
                "experiment_name": self.experiment_name,
                "lead_time_col": self.lead_time_col,
                "location_id_col": self.location_id_col,
            }
        }

        for key, item in self.results.items():
            item = item.copy()
            values = item["values"]
            if isinstance(values, (pl.DataFrame, pl.LazyFrame)):
                item["values"] = values.serialize(format="json")
                serializable_results[f"{key}_polars"] = item
            else:
                serializable_results[key] = item

        with open(path, "w", encoding="utf8") as file:
            json.dump(serializable_results, file)

    @classmethod
    def load_report(cls, path):
        with open(path, "r", encoding="utf8") as file:
            serializable_results = json.load(file)

        class_meta = serializable_results.pop("__class_meta")
        results = {}
        for key, item in serializable_results.items():
            if key.endswith("_polars"):
                item["values"] = pl.DataFrame.deserialize(
                    io.BytesIO(bytes(literal_eval(item["values"])))
                )
            results[key.replace("_polars", "")] = item

        return cls(results=results, **class_meta)

    @classmethod
    def fill_evaluation(
        cls, calculator, experiment_name, lead_time_col, location_id_col,
        bootstrap=False, n_iter=300, CI=0.9, location_metrics=True
    ):
        available_metrics = [
            x.replace("absolute_error", "mae").replace("squared_error", "mse")
            for x in calculator.added_metrics
        ]
        obj = cls(experiment_name, lead_time_col, location_id_col, results={})
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
                {"on": location_id_col}
            )

        if bootstrap:
            obj.add_table(
                "bootstraped_lead_time_metrics",
                calculator.bootstrap_metrics(n_iter, lead_time_col, CI),
                {"metrics": available_metrics, "groupby": lead_time_col,
                 "n_iter": n_iter, "CI": CI}
            )
        return obj

    def add_table(self, table_name, values, metadata):
        self.results[table_name] = {"values": values, "metadata": metadata}


    def add_brier_decomp(self, calculator, n_iter, th, CI):
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

        metadata = metadata.copy()
        metadata["metrics"] = ["obs_bar"]

        self.add_table(
            f"bootstrapped_obs_bar_th:{th}",
            obs_bar,
            metadata,
        )

    def __getitem__(self, key):
        return self.results[key]

    def tables(self):
        return list(self.results.keys())
    
    def extend(self, other, right_prefix=None):
        assert self.lead_time_col == other.lead_time_col
        assert self.location_id_col == other.location_id_col

        if not right_prefix:
            assert not set(self.tables()).intersection(set(other.tables()))
            self.results = {**self.results, **other.results}
        else:
            for table in other.tables():
                self.results[f"{right_prefix}_{table}"] = other[table].copy()
