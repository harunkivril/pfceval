import pytest
import polars as pl
from pathlib import Path

from pfceval.data import Forecast

@pytest.fixture
def parquet_file(tmp_path: Path) -> str:
    """
    This fixture creates a temporary Parquet file with sample data for testing.
    'tmp_path' is a built-in pytest fixture that provides a temporary directory.
    """
    df = pl.DataFrame({
        "station_id": ["A", "A", "B", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "observed": [10, 12, 15, 11],
        "ens_1": [9, 11, 14, 11],
        "ens_2": [10, 13, 16, 12],
        "ens_3": [11, 12, 15, 13],
    })
    file_path = tmp_path / "test_forecast.parquet"
    df.write_parquet(file_path)
    return str(file_path)

@pytest.fixture
def parquet_file_with_bootstrap(tmp_path: Path) -> str:
    """
    Creates a dummy parquet file that already has a '_bootstrap' column
    to test the assertion that prevents overwriting it.
    """
    df = pl.DataFrame({
        "station_id": ["A"],
        "observed": [10],
        "ens_1": [9],
        "_bootstrap": ["pre-existing"],  # This should cause an error
    })
    file_path = tmp_path / "test_forecast_bootstrap.parquet"
    df.write_parquet(file_path)
    return str(file_path)


def test_init_eager(parquet_file):
    """Tests successful initialization when loading data directly into memory."""
    fc = Forecast(
        fc_path=parquet_file,
        ensemble_prefix="ens",
        obs_col="observed",
        bootstrap_cols=["station_id", "date"],
        load_to_memory=True
    )
    # 1. Check object attributes
    assert not fc.lazy
    assert isinstance(fc.forecast, pl.DataFrame)
    assert fc.pred_cols == ["ens_1", "ens_2", "ens_3"]
    assert fc.obs_col == "observed"
    assert fc.deterministic_col == "mean_pred"

    # 2. Check calculated columns
    expected_mean = [10.0, 12.0, 15.0, 12.0]
    expected_bootstrap = ["A-_-2024-01-01", "A-_-2024-01-02", "B-_-2024-01-01", "B-_-2024-01-02"]

    assert fc.forecast["mean_pred"].to_list() == pytest.approx(expected_mean)
    assert fc.forecast["_bootstrap"].to_list() == expected_bootstrap
    assert "mean_pred" in fc.forecast.columns
    assert "_bootstrap" in fc.forecast.columns


def test_init_lazy(parquet_file):
    """Tests successful initialization in lazy mode."""
    fc = Forecast(
        fc_path=parquet_file,
        ensemble_prefix="ens",
        obs_col="observed",
        bootstrap_cols="station_id",  # Test with a single string
        load_to_memory=False
    )
    # 1. Check object attributes
    assert fc.lazy
    assert isinstance(fc.forecast, pl.LazyFrame)
    assert fc.pred_cols == ["ens_1", "ens_2", "ens_3"]

    # 2. Collect the lazy frame to verify results
    fc.collect()  # This should modify the object in place
    assert not fc.lazy  # State should change after collect
    assert isinstance(fc.forecast, pl.DataFrame)

    expected_mean = [10.0, 12.0, 15.0, 12.0]
    expected_bootstrap = ["A", "A", "B", "B"]

    assert fc.forecast["mean_pred"].to_list() == pytest.approx(expected_mean)
    assert fc.forecast["_bootstrap"].to_list() == expected_bootstrap


def test_init_assertions(parquet_file, parquet_file_with_bootstrap):
    """Tests that the expected assertions are raised for invalid inputs."""
    # Test missing bootstrap column
    with pytest.raises(AssertionError, match="Bootstrap col\\(s\\) \\['non_existent'\\] not found."):
        Forecast(fc_path=parquet_file, ensemble_prefix="ens", obs_col="observed",
                 bootstrap_cols="non_existent", load_to_memory=True)

    # Test missing observation column
    with pytest.raises(AssertionError, match="Obs col 'non_existent' not found."):
        Forecast(fc_path=parquet_file, ensemble_prefix="ens", obs_col="non_existent",
                 bootstrap_cols="station_id", load_to_memory=True)

    # Test pre-existing _bootstrap column
    with pytest.raises(AssertionError, match="Col '_bootstrap' already exists. Rename or remove."):
        Forecast(fc_path=parquet_file_with_bootstrap, ensemble_prefix="ens", obs_col="observed",
                 bootstrap_cols="station_id", load_to_memory=True)


def test_check_cols(parquet_file):
    """Tests the 'check_cols' utility method."""
    fc = Forecast(fc_path=parquet_file, ensemble_prefix="ens", obs_col="observed",
                  bootstrap_cols="station_id", load_to_memory=True)
    assert fc.check_cols("station_id") is True
    assert fc.check_cols(["station_id", "observed"]) is True
    assert fc.check_cols("non_existent_col") is False
    assert fc.check_cols(["station_id", "non_existent_col"]) is False


def test_select(parquet_file):
    """Tests that the 'select' method correctly proxies to Polars."""
    # Eager mode
    fc_eager = Forecast(fc_path=parquet_file, ensemble_prefix="ens", obs_col="observed",
                        bootstrap_cols="station_id", load_to_memory=True)
    selected_eager = fc_eager.select("station_id", "observed")
    assert isinstance(selected_eager, pl.DataFrame)
    assert selected_eager.columns == ["station_id", "observed"]

    # Lazy mode
    fc_lazy = Forecast(fc_path=parquet_file, ensemble_prefix="ens", obs_col="observed",
                       bootstrap_cols="station_id", load_to_memory=False)
    selected_lazy = fc_lazy.select("station_id", "observed")
    assert isinstance(selected_lazy, pl.LazyFrame)
    assert selected_lazy.collect_schema().names() == ["station_id", "observed"]


def test_filter(parquet_file):
    """Tests that filtering creates a new, correct Forecast object."""
    fc_orig = Forecast(fc_path=parquet_file, ensemble_prefix="ens", obs_col="observed",
                       bootstrap_cols="station_id", load_to_memory=True)
    
    # Filter the data
    fc_filtered = fc_orig.filter(pl.col("station_id") == "A")

    # 1. Check that a new object is returned
    assert isinstance(fc_filtered, Forecast)
    assert id(fc_orig) != id(fc_filtered)
    assert id(fc_orig.forecast) != id(fc_filtered.forecast)

    # 2. Check original object is unchanged
    assert fc_orig.forecast.shape[0] == 4

    # 3. Check filtered object has correct data
    assert fc_filtered.forecast.shape[0] == 2
    assert fc_filtered.forecast["station_id"].unique().to_list() == ["A"]


def test_copy(parquet_file):
    """Tests that the shallow copy method works as expected."""
    fc_orig = Forecast(fc_path=parquet_file, ensemble_prefix="ens", obs_col="observed",
                       bootstrap_cols="station_id", load_to_memory=True)
    
    fc_copy = fc_orig.copy()

    # 1. It's a shallow copy: the object is new, but the data ('fc' attribute)
    # points to the same underlying Polars DataFrame object.
    assert isinstance(fc_copy, Forecast)
    assert id(fc_orig) != id(fc_copy)
    assert id(fc_orig.forecast) == id(fc_copy.forecast)

    # 2. Prove that methods like 'filter', which reassign the 'fc' attribute,
    # don't affect the original object.
    fc_filtered_from_copy = fc_copy.filter(pl.col("station_id") == "A")
    assert id(fc_orig.forecast) != id(fc_filtered_from_copy.forecast)
    assert fc_orig.forecast.shape[0] == 4
    assert fc_filtered_from_copy.forecast.shape[0] == 2