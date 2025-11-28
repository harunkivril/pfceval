import pytest
import polars as pl

from types import SimpleNamespace
from pfceval.calculator import Calculator
from pfceval import metrics

class MockForecast:
    """A mock of the Forecast class for predictable test inputs."""
    def __init__(self, df):
        self.fc = df
        self.obs_col = "observed"
        self.deterministic_col = "mean_pred"
        self.pred_cols = [c for c in df.columns if "ens_" in c]
        self.engine = "auto"

    def select(self, *args, **kwargs):
        return self.fc.select(*args, **kwargs)

@pytest.fixture
def sample_forecast_object() -> MockForecast:
    """Creates a mock Forecast object with sample data."""
    df = pl.DataFrame({
        "station_id": ["A", "A", "B", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "observed": [10, 12, 15, 11],
        "mean_pred": [9.5, 12.5, 15.5, 10.5],
        "_bootstrap": ["b1", "b2", "b3", "b4"], # Unique bootstrap IDs
        "ens_1": [9, 11, 14, 10],
        "ens_2": [10, 14, 17, 11],
    })
    return MockForecast(df)

@pytest.fixture
def calculator_instance(sample_forecast_object) -> Calculator:
    """Creates a Calculator instance for testing."""
    return Calculator(sample_forecast_object, index_cols=["station_id", "date"])


def test_init(calculator_instance, sample_forecast_object):
    """Tests the successful initialization of the Calculator."""
    calc = calculator_instance
    assert calc.seed == 3136
    assert calc.n_bootstrap == 4
    assert len(calc.added_metrics) == 0
    # Expected columns: index_cols + obs_col + _bootstrap
    expected_cols = {"station_id", "date", "observed", "_bootstrap"}
    assert set(calc.metrics_df.columns) == expected_cols
    assert calc.unique_bootstrap["_bootstrap"].n_unique() == 4

def test_add_metric(calculator_instance):
    """Tests adding a metric, including idempotency."""
    calc = calculator_instance
    initial_col_count = len(calc.metrics_df.columns)

    # 1. Add a new metric
    calc.add_metric("variance", metrics.variance(calc.forecast.pred_cols))
    assert "variance" in calc.metrics_df.columns
    assert "variance" in calc.added_metrics
    assert len(calc.metrics_df.columns) == initial_col_count + 1
    assert calc.metrics_df["variance"].to_list() == pytest.approx([0.5, 4.5, 4.5, 0.5])

    # 2. Try adding the same metric again; nothing should change
    calc.add_metric("variance", metrics.variance(calc.forecast.pred_cols))
    assert len(calc.added_metrics) == 1
    assert len(calc.metrics_df.columns) == initial_col_count + 1

def test_add_specific_metrics(calculator_instance):
    """Tests the specific 'add_*' helper methods."""
    calc = calculator_instance
    calc.add_absolute_error()
    assert "absolute_error" in calc.added_metrics
    assert "absolute_error" in calc.metrics_df.columns
    # mean_pred: [9.5, 12.5, 15.5, 10.5], observed: [10, 12, 15, 11]
    # abs_error: [0.5, 0.5, 0.5, 0.5]
    assert calc.metrics_df["absolute_error"].to_list() == [0.5, 0.5, 0.5, 0.5]

    calc.add_mae() # Should do nothing since absolute_error is already there
    assert len(calc.added_metrics) == 1

    calc.add_squared_error()
    assert "squared_error" in calc.added_metrics
    # sq_error: [0.25, 0.25, 0.25, 0.25]
    assert calc.metrics_df["squared_error"].to_list() == [0.25, 0.25, 0.25, 0.25]

def test_get_metrics_no_grouping(calculator_instance):
    """Tests getting aggregated metrics without grouping."""
    calc = calculator_instance
    calc.add_absolute_error() # mean is 0.5
    calc.add_squared_error() # mean is 0.25
    calc.add_variance()      # variance is 5.5

    metrics_result = calc.get_metrics()
    assert metrics_result.shape == (1, 3)
    # Check that columns were renamed correctly
    assert "mae" in metrics_result.columns
    assert "mse" in metrics_result.columns
    assert "variance" in metrics_result.columns
    # Check values
    assert metrics_result["mae"].item() == pytest.approx(0.5)
    assert metrics_result["mse"].item() == pytest.approx(0.25)
    assert metrics_result["variance"].item() == pytest.approx(2.5)

def test_get_metrics_with_grouping(calculator_instance):
    """Tests getting aggregated metrics with grouping."""
    calc = calculator_instance
    calc.add_absolute_error()
    calc.add_crps()

    metrics_result = calc.get_metrics(groupby_cols=["station_id"])
    assert metrics_result.shape == (2, 3) # 2 stations, 3 columns (id, mae, crps)
    assert "station_id" in metrics_result.columns
    assert "mae" in metrics_result.columns
    assert "crps" in metrics_result.columns

    # Station A and B have the same values in the mock data
    station_a_metrics = metrics_result.filter(pl.col("station_id") == "A")
    assert station_a_metrics["mae"].item() == pytest.approx(0.5)
    assert station_a_metrics["crps"].item() == pytest.approx(0.5)

def test_bootstrap_metrics(calculator_instance):
    """Tests the bootstrap functionality for deterministic results."""
    calc = calculator_instance
    calc.add_variance()
    calc.add_crps()

    # Since the metrics are constant, the mean and quantiles should be the same
    # regardless of the bootstrap samples. This is a great way to test the
    # structure of the output without complex calculations.
    bootstrap_result = calc.bootstrap_metrics(n_iter=10, groupby_cols="station_id", CI=0.95)

    assert bootstrap_result.shape == (2, 7) # 2 stations, 1 group col + 2 metrics * 3 stats
    # Check for correct column naming with CI=0.95 -> q97.5 and q02.5
    expected_cols = [
        "station_id", "crps_q003", "crps_mean", "crps_q097",
        "variance_q003", "variance_mean", "variance_q097"
    ]
    assert sorted(bootstrap_result.columns) == sorted(expected_cols)

    # Check values for one group
    station_b_metrics = bootstrap_result.filter(pl.col("station_id") == "B")
    assert station_b_metrics["variance_mean"].item() >= station_b_metrics["variance_q003"].item()
    assert station_b_metrics["variance_mean"].item() <= station_b_metrics["variance_q097"].item()
    assert station_b_metrics["crps_mean"].item() >= station_b_metrics["crps_q003"].item()
    assert station_b_metrics["crps_mean"].item() <= station_b_metrics["crps_q097"].item()