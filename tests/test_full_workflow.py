import pytest
import polars as pl
from pathlib import Path

from pfceval.data import Forecast
from pfceval.calculator import Calculator
from pfceval.evaluation import Evaluation

# You can reuse the fixture from test_forecast.py to create this file
@pytest.fixture
def parquet_file(tmp_path: Path) -> str:
    df = pl.DataFrame({
        "station_id": ["A", "B"], "lead_time": [12, 24],
        "observed": [10, 15], "ens_1": [9, 14], "ens_2": [11, 16],
        "latitude": [52.1, 52.2], "longitude": [5.1, 5.2]
    })
    file_path = tmp_path / "integration_test.parquet"
    df.write_parquet(file_path)
    return str(file_path)

def test_end_to_end_workflow(parquet_file, tmp_path: Path):
    """
    Tests the full workflow from data loading to saving an evaluation report.
    """
    # 1. Forecast: Load the data
    fc = Forecast(
        fc_path=parquet_file,
        ensemble_prefix="ens",
        obs_col="observed",
        bootstrap_cols=["station_id", "lead_time"],
        load_to_memory=True
    )

    # 2. Calculator: Initialize and add metrics
    # This now uses your REAL metrics, not mocks
    calc = Calculator(forecast=fc, index_cols=["lead_time", "station_id"])
    calc.add_absolute_error()
    calc.add_spread()

    # 3. Evaluation: Fill, save, and load a report
    evaluation = Evaluation.fill_evaluation(
        calculator=calc,
        experiment_name="integration_test",
        lead_time_col="lead_time",
        location_id_col="station_id"
    )
    
    # Save to a temporary directory
    save_path = tmp_path / "integration_results"
    evaluation.save_results(str(save_path))

    # Load it back
    loaded_evaluation = Evaluation.load_report(str(save_path))

    # 4. Assert: Check if the final result is as expected
    assert loaded_evaluation.experiment_name == "integration_test"
    assert "overall_metrics" in loaded_evaluation.tables()
    
    overall_metrics = loaded_evaluation["overall_metrics"]["values"]
    # Based on the sample data, the absolute error is 0 for both rows.
    # The deterministic_col (mean_pred) would be 10 and 15.
    # The absolute error would be abs(10-10)=0 and abs(15-15)=0. MAE is 0.
    assert overall_metrics["mae"].item() == pytest.approx(0.0) 