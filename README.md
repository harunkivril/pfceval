# pfceval

![Project Status: Alpha](https://img.shields.io/badge/status-alpha-orange)
![Downloads](https://img.shields.io/pypi/dm/pfceval)
![Python](https://img.shields.io/badge/python-3.10+-blue)

**pfceval** (Polars/Probabilistic Forecast Evaluation) is a robust Python library designed to streamline the evaluation of probabilistic forecasts, with particular emphasis on ensemble-based prediction systems. Leveraging the high-performance Polars data processing framework, pfceval efficiently handles large-scale datasets, enabling rapid computation of a comprehensive suite of forecast verification metrics. The library supports advanced evaluation workflows, including bootstrapped confidence intervals, threshold-based scoring methods, and detailed reliability diagnostics, empowering users to rigorously assess and interpret forecast performance with precision and clarity.

> ⚠️ **Warning**: This package is in early alpha. Expect breaking changes and incomplete documentation.

## Features

- Forecast evaluation via a modular `Forecast` and `Calculator` interface  
- Support for ensemble forecasts with observation comparison  
- Metrics:  
  - **Deterministic**: Absolute Error, Squared Error  
  - **Probabilistic**: Spread, CRPS, Threshold-weighted CRPS, Brier Score  
  - **Diagnostics**: Rank Histograms, Brier Score Decomposition  
- Bootstrapped confidence intervals  
- Subgroup evaluation (e.g., by location or "unseen" subsets)  
- Plotting utilities for evaluations  

## Main Components

### `pfceval.Forecast`

Handles input forecast data:  
- Loads ensemble predictions and corresponding observations  
- Supports filtering and in-memory operations  
- Can be initialized from Parquet or other supported formats  

**Example Forecast Data Format**
Your input data should include:
 - One observation column
 - Multiple ensemble/Single prediction columns (e.g. pred_q1, pred_q2, …)
 - Optional metadata (e.g., location SID, forecast step, run_id)

*Example Data Format:*
| SID | step | run_id   | pred_q1 | pred_q2 | pred_q3 | pred_q4 | observation |
|-----|------|----------|---------|---------|---------|---------|-------------|
| 101 | 6    | 20230801 | 2.1     | 2.3     | 1.9     | 2.0     | 2.2         |
| 101 | 12   | 20230801 | 3.0     | 3.2     | 2.9     | 3.1     | 3.1         |
| 102 | 6    | 20230801 | 1.1     | 1.2     | 1.0     | 1.3     | 1.2         |

You can use any column names for your ensemble members — just set the correct ensemble_prefix when loading with Forecast.

### `pfceval.Calculator`

Computes forecast evaluation metrics:  
- `add_absolute_error()`  
- `add_squared_error()`  
- `add_spread()`  
- `add_crps()`  
- `add_twcrps(threshold)`  
- `add_brier(threshold)`  

### `pfceval.Evaluation`

Aggregates results into structured evaluation reports:  
- Uses a `Calculator` to compute metrics  
- Supports bootstrapping with `n_iter` and confidence intervals (`CI`)  
- Allows report extension and saving to JSON  
- Adds diagnostic plots like rank histograms and Brier decomposition 

```python
from pfceval import Forecast, Calculator, Evaluation

# Load forecast data
forecast = Forecast(
    fc_path="path/to/forecast.parquet",
    ensemble_prefix="pred_q",
    obs_col="observation",
    bootstrap_cols="run_id",
    load_to_memory=True,
    engine="auto"
)

# Initialize calculator
calc = Calculator(forecast, index_cols=["SID", "step"])
calc.add_absolute_error()
calc.add_crps()
calc.add_brier(threshold=0.5)

# Create evaluation report
report = Evaluation.fill_evaluation(
    experiment_name="model_trial_0",
    calculator=calc,
    lead_time_col="step",
    location_id_col="SID",
    bootstrap=True,
    n_iter=1000,
    CI=0.9,
    location_metrics=True
)

# Add diagnostics
report.add_rank_histogram(calculator=calc, nbins=20)
report.add_brier_decomp(calculator=calc, th=0.5, n_iter=1000, CI=0.9)

# Save results
report.save_results("evaluation_report.json")
```

## Plotting

The `pfceval.plotting` module provides visualization functions to explore and communicate forecast evaluation results effectively. These functions help visualize metrics across lead times, locations, and thresholds and include diagnostic plots for forecast quality.

### Plot Types

- **Lead Time Metrics Plot**  
  Displays metrics such as CRPS or Brier Score across different forecast lead times, optionally with confidence intervals and multiple experiments comparison.

- **Location Metrics Plot**  
  Geographic scatterplots of forecast metrics at a specific lead time, visualized using Cartopy for map projections.

- **Reliability Diagram**  
  Calibration plots showing observed event frequency versus predicted probabilities, including Brier score decomposition components and predicted probability histograms.

- **Spread vs RMSE Plot**  
  Diagnostic plot comparing ensemble spread to the Root Mean Square Error (RMSE) over forecast steps.

- **Rank Histogram**  
  Histogram of ensemble ranks of the observations, useful for checking ensemble reliability and calibration.

```python
from pfceval import Evaluation
from pfceval.plotting import (
    plot_lead_time_metrics,
    plot_location_metrics,
    plot_reliability_diagram,
    plot_spread_rmse,
    plot_rank_histogram
)

# Load evaluation results
eval1 = Evaluation.load("eval_exp1")
eval2 = Evaluation.load("eval_exp2")

# Plot metrics over lead time for multiple experiments
plot_lead_time_metrics(eval1, eval2, metrics=["crps", "brier_score"])

# Plot spatial distribution of a metric at lead time 24 hours
plot_location_metrics(step=24, evaluation=eval1)

# Reliability diagram for threshold 9 at 24 hours lead time
plot_reliability_diagram(eval1, step=24, table_name="bootstrapped_obs_bar_th:9", nbins=10)

# Plot spread vs RMSE for a single experiment
plot_spread_rmse(eval1)

# Rank histogram at 24 hours lead time
plot_rank_histogram(eval1, step=24)

```

## Project Status

**Status: Alpha**

This library is currently in early development. Interfaces and APIs may change, and features are being actively added.  
We welcome early adopters and contributors, but recommend caution when using it in production.

### Contributing

Contributions are welcome! Please open an issue to discuss your plans.

### Acknowledgements

This package builds on the work of the scientific forecasting and open-source data community. We thank contributors to properscoring, polars, and various ensemble verification tools.