import polars as pl
import pytest
import numpy as np
import properscoring as ps
from pfceval.metrics import (
    absolute_error,
    squared_error,
    variance,
    crps,
    brier_score,
    brier_decomposition
)

@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "pred1": [1.0, 2.0, 3.0],
        "pred2": [1.5, 2.5, 3.5],
        "obs": [1.2, 2.1, 2.8]
    })


def test_absolute_error(sample_df):
    expr = absolute_error("pred1", "obs")
    result = sample_df.select(expr.alias("abs_err"))
    expected = [abs(1.0 - 1.2), abs(2.0 - 2.1), abs(3.0 - 2.8)]
    assert result["abs_err"].to_list() == pytest.approx(expected)


def test_squared_error(sample_df):
    expr = squared_error("pred2", "obs")
    result = sample_df.select(expr.alias("sq_err"))
    expected = [(1.5 - 1.2)**2, (2.5 - 2.1)**2, (3.5 - 2.8)**2]
    assert result["sq_err"].to_list() == pytest.approx(expected)


def test_variance(sample_df):
    expr = variance(["pred1", "pred2"])
    result = sample_df.select(expr.alias("variance"))

    preds = np.vstack(
        [sample_df["pred1"].to_numpy(), sample_df["pred2"].to_numpy()]).T
    expected = np.var(preds, axis=1, ddof=1)  # ddof=1 for sample var

    assert result["variance"].to_list() == pytest.approx(expected.tolist(), rel=1e-6)

def test_crps(sample_df):
    expr = crps(["pred1", "pred2"], "obs")
    result = sample_df.select(expr.alias("crps"))

    # Prepare predictions as arrays per row
    preds = np.vstack(
        [sample_df["pred1"].to_numpy(), sample_df["pred2"].to_numpy()]).T
    obs = sample_df["obs"].to_numpy()

    expected = ps.crps_ensemble(obs, preds)
    assert result["crps"].to_list() == pytest.approx(expected, rel=1e-6)


def test_brier_score(sample_df):
    threshold = 2.0
    expr = brier_score(["pred1", "pred2"], "obs", threshold)
    result = sample_df.select(expr.alias("brier"))

    preds = np.vstack(
        [sample_df["pred1"].to_numpy(), sample_df["pred2"].to_numpy()]).T
    obs = (sample_df["obs"] > threshold).to_numpy()

    # Convert ensemble forecasts to probabilities of exceedance
    probs = np.mean(preds > threshold, axis=1)
    expected = (probs - obs) ** 2  # Brier score definition

    assert result["brier"].to_list() == pytest.approx(expected, rel=1e-6)


def test_brier_decomposition_identity(sample_df):
    pred_cols = ["pred1", "pred2"]
    obs_col = "obs"
    threshold = 3
    engine="auto"

    # Compute Brier score (mean over all rows)
    brier_expr = brier_score(pred_cols, obs_col, threshold).alias("brier")
    brier_df = sample_df.select(brier_expr)
    brier_value = brier_df["brier"].mean()

    # Compute decomposition
    decomp_df, _ = brier_decomposition(
        sample_df, pred_cols, obs_col, threshold, engine)
    reliability = decomp_df["reliability"][0]
    resolution = decomp_df["resolution"][0]
    uncertainty = decomp_df["uncertainty"][0]

    # Check if the decomposition sums to Brier score within tolerance
    assert brier_value == pytest.approx(reliability - resolution + uncertainty, rel=1e-6)
