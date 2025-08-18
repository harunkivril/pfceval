import pytest
import polars as pl
import json

from polars.testing import assert_frame_equal
from pfceval.evaluation import Evaluation
from pathlib import Path

class MockCalculator:
    """A mock of the Calculator class for predictable test inputs."""
    def __init__(self):
        self.added_metrics = ["absolute_error", "spread"]

    def get_metrics(self, groupby_cols=None):
        if groupby_cols is None:
            return pl.DataFrame({"mae": [0.5], "spread": [0.8]})
        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]
        metrics = {col: [1] for col in groupby_cols}
        metrics["mae"] = [0.5]
        metrics["spread"] = [0.8]
        return pl.DataFrame(metrics)

    def bootstrap_metrics(self, n_iter, groupby_cols, CI):
        return pl.DataFrame({groupby_cols: [1], "mae_mean": [0.5], "spread_mean": [0.8]})
    
    def get_station_meta(self, station_id_col):
        return pl.DataFrame({station_id_col: ["A", "B"], "lat": [0, 1]})

    def get_bootstrapped_brier_decomp(self, n_iter, th, groupby_cols, CI):
        decomp = pl.DataFrame({groupby_cols: [1], "reliability": [0.1]})
        obs_bar = pl.DataFrame({groupby_cols: [1], "obs_bar": [0.2]})
        return decomp, obs_bar
    
    def get_rank_histogram(self, n_bins, groupby_cols):
        counts = {col: [1] for col in groupby_cols}
        counts["counts"] = [[10, 20]]
        counts = pl.DataFrame(counts)
        bins = [0, 1, 2]
        return counts, bins
    

@pytest.fixture
def mock_calculator():
    """Provides a mock Calculator instance."""
    return MockCalculator()

@pytest.fixture
def sample_evaluation():
    """Provides a sample Evaluation instance for testing."""
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    results = {
        "metrics_table": {"values": df, "metadata": {"source": "test"}},
        "config_table": {"values": {"param": 10}, "metadata": {"type": "scalar"}}
    }
    return Evaluation("test_exp", "lt", "sid", results)


def test_init_and_properties(sample_evaluation):
    """Tests basic initialization and property/dunder methods."""
    assert sample_evaluation.experiment_name == "test_exp"
    assert "metrics_table" in sample_evaluation.tables()
    # Test __getitem__
    assert "source" in sample_evaluation["metrics_table"]["metadata"]
    assert isinstance(sample_evaluation["metrics_table"]["values"], pl.DataFrame)

def test_save_and_load_report(sample_evaluation, tmp_path: Path):
    """Tests the critical save and load functionality."""
    # 1. Save the results to a temporary directory
    sample_evaluation.save_results(str(tmp_path))

    # 2. Verify the file structure
    metadata_path = tmp_path / "metadata.json"
    data_dir_path = tmp_path / "data"
    parquet_path = data_dir_path / "metrics_table.parquet"

    assert metadata_path.exists()
    assert data_dir_path.is_dir()
    assert parquet_path.exists()

    # 3. Verify the contents of the metadata file
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    assert metadata["__class_meta"]["experiment_name"] == "test_exp"
    assert metadata["metrics_table"]["metadata"]["_data_file"] == "metrics_table.parquet"
    assert metadata["config_table"]["values"]["param"] == 10 # Non-DF data is stored directly

    # 4. Load the report back
    loaded_eval = Evaluation.load_report(str(tmp_path))
    assert isinstance(loaded_eval, Evaluation)
    assert loaded_eval.experiment_name == "test_exp"
    assert "metrics_table" in loaded_eval.tables()
    assert "config_table" in loaded_eval.tables()

    # 5. Verify the loaded data is correct
    assert_frame_equal(loaded_eval["metrics_table"]["values"], sample_evaluation["metrics_table"]["values"])
    assert loaded_eval["config_table"]["values"]["param"] == 10

def test_load_report_file_not_found(tmp_path: Path):
    """Tests that loading from a non-existent path raises an error."""
    with pytest.raises(FileNotFoundError):
        Evaluation.load_report(str(tmp_path / "non_existent"))

def test_fill_evaluation(mock_calculator):
    """Tests the classmethod for populating an Evaluation object."""
    # Test with all options enabled
    eval_obj = Evaluation.fill_evaluation(
        mock_calculator,
        experiment_name="fill_test",
        lead_time_col="lt",
        location_id_col="sid",
        bootstrap=True,
        location_metrics=True
    )
    expected_tables = {
        "overall_metrics",
        "lead_time_metrics",
        "lead_time_location_metrics",
        "station_meta",
        "bootstraped_lead_time_metrics"
    }
    assert expected_tables.issubset(set(eval_obj.tables()))
    assert eval_obj["overall_metrics"]["metadata"]["metrics"] == ["mae", "spread"]

def test_add_brier_and_rank(mock_calculator):
    """Tests adding Brier decomposition and rank histogram tables."""
    eval_obj = Evaluation("test_exp", "lt", "sid", {})
    
    # Add Brier
    eval_obj.add_brier_decomp(mock_calculator, n_iter=100, th=0.5, CI=0.9)
    assert "bootstrapped_brier_decomp_th:0.5" in eval_obj.tables()
    assert "bootstrapped_obs_bar_th:0.5" in eval_obj.tables()
    assert eval_obj["bootstrapped_brier_decomp_th:0.5"]["metadata"]["th"] == 0.5

    # Add Rank Histogram
    eval_obj.add_rank_histogram(mock_calculator, n_bins=10)
    assert "lead_time_rank_histogram" in eval_obj.tables()
    assert eval_obj["lead_time_rank_histogram"]["metadata"]["bins"] == [0, 1, 2]

def test_extend(sample_evaluation):
    """Tests merging two Evaluation objects."""
    # Create another evaluation object
    other_results = {"other_table": {"values": pl.DataFrame({"c": [5]}), "metadata": {}}}
    other_eval = Evaluation("test_exp", "lt", "sid", other_results)

    # Case 1: Extend without conflict
    base_eval_copy = Evaluation(
        sample_evaluation.experiment_name, 
        sample_evaluation.lead_time_col, 
        sample_evaluation.location_id_col, 
        sample_evaluation.results.copy()
    )
    base_eval_copy.extend(other_eval)
    assert "other_table" in base_eval_copy.tables()
    assert "metrics_table" in base_eval_copy.tables()

    # Case 2: Extend with a prefix
    base_eval_copy = Evaluation(
        sample_evaluation.experiment_name, 
        sample_evaluation.lead_time_col, 
        sample_evaluation.location_id_col, 
        sample_evaluation.results.copy()
    )
    base_eval_copy.extend(other_eval, right_prefix="new")
    assert "new_other_table" in base_eval_copy.tables()

    # Case 3: Conflict without prefix should raise an error
    conflicting_eval = Evaluation(
        "test_exp", "lt", "sid", {"metrics_table": other_eval["other_table"]})
    with pytest.raises(AssertionError, match="Table names conflict"):
        sample_evaluation.extend(conflicting_eval)
        
    # Case 4: Mismatched lead_time_col should raise an error
    mismatched_eval = Evaluation("test_exp", "different_lt", "sid", other_results)
    with pytest.raises(AssertionError, match="Lead time columns must match"):
        sample_evaluation.extend(mismatched_eval)