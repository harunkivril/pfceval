import polars as pl
import json

class EvaluationReport:

    def __init__(self, results: dict):
        self.results = results
        self.metrics = list(results.keys())

    def save_results(self, path):
        serializable_results = {}

        for key, item in self.results.items():
            values = item["values"]
            if isinstance(values, (pl.DataFrame, pl.LazyFrame)):
                values = values.serialize(format="json")
                item["values"] = values
                serializable_results[f"{key}_polars"] = item
            else:
                serializable_results[key] = item

        with open(path, "w", encoding="utf8") as file:
            json.dump(serializable_results, file)

    @classmethod
    def get_report(cls, path):
        with open(path, "w", encoding="utf8") as file:
            serializable_results = json.load(file)

        results = {}
        for key, item in serializable_results:
            if key.endswith("_polars"):
                item["values"] = pl.DataFrame.deserialize(item["values"])
            results[key] = item

        return cls(results)
    
    def add_metric(self, metric_name, values, metadata):
        self.results[metric_name] = {"values": values, "metadata": metadata}
