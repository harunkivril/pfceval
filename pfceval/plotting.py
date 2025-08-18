import polars as pl
import matplotlib.pyplot as plt

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pfceval.evaluation import Evaluation

# Apply a specific matplotlib style for consistent plotting.
plt.style.use("seaborn-v0_8-notebook")


def stack_overall_metrics(
    *evals: Evaluation,
    table_name: str = "overall_metrics",
    metrics: list[str] | None = None,
):
    """
    Stacks overall metrics from multiple evaluation experiments.

    This function takes one or more `Evaluation` objects, extracts
    a specified table (e.g., "overall_metrics"), and concatenates
    their metric values. It ensures that only common metrics across
    all evaluations are considered if no specific metrics are
    provided.

    Args:
        *evals (Evaluation): Variable number of Evaluation objects.
        table_name (str, optional): Name of the table to extract
                                     metrics from. Defaults to
                                     "overall_metrics".
        metrics (list[str] | None, optional): List of specific
                                               metrics to stack. If
                                               None, common metrics
                                               are used. Can also be
                                               a single string.

    Returns:
        pl.DataFrame: A Polars DataFrame with stacked metrics and an
                      "experiment" column.

    Raises:
        AssertionError: If specified metrics aren't in the first
                        evaluation's metadata.
    """
    meta = evals[0][table_name]["metadata"]

    if metrics is None:
        metrics = get_common_metrics(evals, table_name)
    else:
        metrics = [metrics] if isinstance(metrics, str) else metrics
        assert all(metric in meta["metrics"] for metric in metrics)

    stacked = []
    for ev in evals:
        table = (
            ev[table_name]["values"]
            .select(metrics)
            .insert_column(0, pl.lit(ev.experiment_name).alias("experiment"))
        )
        stacked.append(table)

    return pl.concat(stacked)


def get_common_metrics(evals: list[Evaluation], table_name: str):
    """
    Identifies common metrics across multiple evaluation experiments.

    Args:
        evals (list[Evaluation]): A list of Evaluation objects.
        table_name (str): The table name to check for metrics.

    Returns:
        list[str]: A list of metric names common to all provided
                   evaluations for the specified table.
    """
    metric_set = set(evals[0][table_name]["metadata"]["metrics"])
    for ev in evals:
        metric_set = metric_set.intersection(
            ev[table_name]["metadata"]["metrics"])
    return list(metric_set)


def plot_lead_time_metrics(
    *evals: Evaluation,
    table_name: str = "lead_time_metrics",
    metrics: list[str] | None = None,
):
    """
    Generates plots for lead time metrics from multiple experiments.

    Visualizes how various metrics change with lead time, supporting
    plotting with or without bootstrapped confidence intervals.

    Args:
        *evals (Evaluation): Variable number of Evaluation objects.
        table_name (str, optional): Table name with lead time metrics.
                                     Defaults to "lead_time_metrics".
        metrics (list[str] | None, optional): List of specific metrics
                                               to plot. If None, common
                                               metrics are used. Can be
                                               a single string.

    Raises:
        AssertionError: If specified metrics aren't in the first
                        evaluation's metadata.
    """
    meta = evals[0][table_name]["metadata"]

    if metrics is None:
        metrics = get_common_metrics(evals, table_name)
    else:
        metrics = [metrics] if isinstance(metrics, str) else metrics
        assert all(metric in meta["metrics"] for metric in metrics)

    for metric in metrics:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.grid(True)

        for ev in evals:
            table = ev[table_name]["values"]
            table = table.with_columns(pl.col(ev.lead_time_col).dt.total_hours())

            label_name = (
                f"{ev.experiment_name}: " +
                " ".join(x.upper() for x in metric.split("_"))
            )
            bootstrap_status = ""

            if meta.get("n_iter"):
                lq, uq = sorted(x for x in table.columns if f"{metric}_q" in x)
                ax.plot(
                    table[ev.lead_time_col],
                    table[f"{metric}_mean"],
                    marker="o",
                    label=label_name,
                )
                ax.fill_between(
                    table[ev.lead_time_col],
                    table[lq],
                    table[uq],
                    alpha=0.2,
                    label=f"{meta['CI']}% CI ({label_name})",
                )
                bootstrap_status = f" Bootstrapped N:{meta['n_iter']}"
            else:
                ax.plot(
                    table[ev.lead_time_col],
                    table[metric],
                    marker="o",
                    label=label_name,
                )

        metric_name = " ".join(x.upper() for x in metric.split("_"))
        ax.set_title(
            f"{metric_name} by {evals[0].lead_time_col.capitalize()}"
            f"{bootstrap_status}"
        )
        ax.set_xlabel(evals[0].lead_time_col.capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.legend()

        fig.show()


def plot_location_metrics(
    step: int,
    evaluation: Evaluation,
    compare_with: Evaluation = None,
    table_name: str = "lead_time_location_metrics",
    metrics: str | None = None,
    cmap: str = "viridis",
    tol: float = 0.01,
    dot_size: int = 10,
):
    """
    Plots metrics at specific geographic locations for a given step.

    Visualizes location-based metrics on a map using Cartopy. Can
    compare metrics between two experiments or show a single one.

    Args:
        step (int): The lead time step (hours) to plot.
        evaluation (Evaluation): The primary Evaluation object.
        compare_with (Evaluation, optional): Evaluation for comparison.
                                            If provided, plots metric
                                            differences. Defaults to None.
        table_name (str, optional): Table name with location metrics.
                                     Defaults to
                                     "lead_time_location_metrics".
        metrics (str | None, optional): Specific metric to plot. If None,
                                         all available metrics are plotted.
                                         Must be a single string.
        cmap (str, optional): Colormap for plotting. Defaults to "viridis".
        tol (float, optional): Tolerance for map boundaries. Defaults to 0.01.
        dot_size (int, optional): Size of scatter plot markers. Defaults to 10.

    Raises:
        AssertionError: If specified metrics aren't in the evaluation's
                        metadata.
    """

    meta = evaluation[table_name]["metadata"]
    table = evaluation[table_name]["values"].filter(
        step=pl.duration(hours=step)
    )
    station_locations = evaluation["station_meta"]["values"]
    table = table.join(station_locations, on=evaluation.location_id_col)

    if compare_with:
        metrics = get_common_metrics([evaluation, compare_with], table_name)
        table = table.join(
            (
                compare_with[table_name]["values"]
                .filter(step=pl.duration(hours=step))
                .select(pl.col(evaluation.location_id_col, *metrics))
            ),
            on=evaluation.location_id_col,
        )

    if metrics is None:
        metrics = meta["metrics"]
    else:
        metrics = [metrics] if isinstance(metrics, str) else metrics
        assert all(metric in meta["metrics"] for metric in metrics)

    for metric in metrics:
        fig, ax = plt.subplots(
            figsize=(15, 8), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        ax.set_global()
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        metric_name = " ".join(x.upper() for x in metric.split("_"))
        if compare_with:
            value = (
                table.select(pl.col(metric) - pl.col(f"{metric}_right"))
                .to_numpy()
                .squeeze()
            )
            metric_name = (
                f"{metric_name} ({evaluation.experiment_name}"
                f"- {compare_with.experiment_name})"
            )
        else:
            value = table.select(pl.col(metric)).to_numpy().squeeze()

        lat = table.select(pl.col("latitude")).to_numpy().squeeze()
        lon = table.select(pl.col("longitude")).to_numpy().squeeze()

        ax.coastlines()
        ax.set_extent(
            [lon.min() - tol, lon.max() + tol, lat.min() - tol, lat.max() + tol],
            crs=ccrs.PlateCarree(),
        )

        vmin, vmax = np.quantile(value, 0.01), np.quantile(value, 0.99)
        if compare_with:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

        img = ax.scatter(
            lon, lat, s=dot_size, c=value, vmax=vmax, vmin=vmin, 
            alpha=0.7, cmap=cmap
        )
        cbar = fig.colorbar(img)
        cbar.set_label(metric)

        ax.gridlines(draw_labels=True)

        ax.set_title(f"{metric_name} Step:{step} ", fontdict={"size": 16})
        fig.tight_layout()


def rebin_obs_bar(obs_bar: pl.DataFrame, nbins: int | None):
    """
    Rebins observed frequencies (obs_bar) to a specified number of bins.

    Used for reliability diagrams to group probabilities into fewer,
    fixed bins, recalculating mean observed frequency and total count.

    Args:
        obs_bar (pl.DataFrame): DataFrame with "prob" (predicted)
                                and "count" (observations).
        nbins (int | None): Desired number of bins. If None, no
                            rebinning.

    Returns:
        pl.DataFrame: Rebinned observed frequencies, sorted by
                      probability.
    """
    if not nbins:
        return obs_bar

    new_probs = 1 / nbins * (np.arange(nbins) + 1 / 2)
    assigned = np.array(
        [np.argmin(abs(new_probs - p)) for p in obs_bar["prob"]]
    )
    obs_bar = obs_bar.with_columns(prob=new_probs[assigned])

    obs_bar = obs_bar.group_by("prob").agg(
        (pl.exclude("count") * (pl.col("count") / pl.col("count").sum())).sum(),
        count=pl.col("count").sum(),
    )
    return obs_bar.sort("prob")


def prep_reliability_ax():
    """
    Prepares the figure and axes for plotting a reliability diagram.

    Sets up a matplotlib figure with a grid layout, allocating space
    for the reliability curve and a histogram.

    Returns:
        tuple[matplotlib.gridspec.GridSpec, matplotlib.figure.Figure,
              matplotlib.axes.Axes]:
            - gs (matplotlib.gridspec.GridSpec): The GridSpec object.
            - fig (matplotlib.figure.Figure): The matplotlib figure.
            - rel_ax (matplotlib.axes.Axes): Axes for reliability curve.
    """
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(4, 3)
    rel_ax = fig.add_subplot(gs[:2, :3])
    return gs, fig, rel_ax


def plot_reliability_diagram(
    *evals: Evaluation, step: int, table_name: str, nbins: int | None = None
):
    """
    Plots a reliability diagram for one or more evaluation experiments.

    Assesses the calibration of probabilistic forecasts, showing
    observed frequency vs. predicted probability. Includes a histogram
    of predicted probabilities and Brier score decomposition.

    Args:
        *evals (Evaluation): Variable number of Evaluation objects. Max
                             3 supported.
        step (int): Lead time step (hours) for the diagram.
        table_name (str): Table name with observed frequencies (e.g.,
                          "obs_bar:<threshold>").
        nbins (int | None, optional): Number of bins to rebin
                                       probabilities into. If None, no
                                       rebinning.

    Raises:
        ValueError: If more than 3 experiments are provided.
    """
    if len(evals) > 3:
        raise ValueError("Too many experiments to plot. Max 3 supported.")

    decomp_table = table_name.replace("obs_bar", "brier_decomp")
    th = int(table_name.split(":")[-1])
    color_list = plt.colormaps["tab10"].colors

    gs, fig, rel_ax = prep_reliability_ax()

    rel_ax.plot([0, 1], [0, 1], linestyle="--", c="black")
    hist_ax = fig.add_subplot(gs[2, :3])

    for i, ev in enumerate(evals):
        obs_bar = (
            ev[table_name]["values"]
            .filter(step=pl.duration(hours=step))
            .select(pl.exclude("step"))
        )

        rel, res, unc = (
            ev[decomp_table]["values"]
            .filter(step=pl.duration(hours=step))
            .select(pl.selectors.ends_with("_mean"))
            .to_numpy()
            .squeeze()
            .round(3)
        )
        bs = round(rel - res + unc, 3)

        obs_bar = rebin_obs_bar(obs_bar, nbins)
        label_name = (
            f"{ev.experiment_name}|Step:{step}|Rel:{rel}|Res:{res}|BS:{bs}"
        )
        meta = ev[table_name]["metadata"]

        lq, uq = sorted(x for x in obs_bar.columns if "obs_bar_q" in x)
        rel_ax.plot(
            obs_bar["prob"],
            obs_bar["obs_bar_mean"],
            marker=".",
            label=label_name,
            c=color_list[i],
        )
        rel_ax.fill_between(
            obs_bar["prob"],
            obs_bar[lq],
            obs_bar[uq],
            alpha=0.2,
            color=color_list[i],
        )
        rel_ax.grid(True)

        hist_ax.hist(
            obs_bar["prob"],
            bins=len(obs_bar),
            weights=obs_bar["count"],
            align="mid",
            histtype="step",
            linewidth=2,
            log=True,
            color=color_list[i],
            fill=True,
            alpha=0.5,
        )
        hist_ax.set_ylabel("Counts")
        hist_ax.set_xlim(rel_ax.get_xlim())
        hist_ax.set_xlabel("Predicted Probability") 

    pos_class_ratio = obs_bar["mean_group"].first()
    hist_ax.set_title(f"Positive Class Ratio: {pos_class_ratio:.4f}")

    rel_ax.legend()
    rel_ax.set_title(
        f"Reliability Diagram | Th:{th} | CI:{meta['CI']} N:{meta['n_iter']}"
    )
    rel_ax.set_ylabel("Observed Frequency")
    rel_ax.set_xlabel("Predicted Probability")
    fig.tight_layout()


def plot_spread_rmse(
    ev: Evaluation, table_name: str = "bootstraped_lead_time_metrics"
):
    """
    Plots spread and RMSE metrics over lead time for an experiment.

    Visualizes the relationship between forecast spread and Root Mean
    Squared Error (RMSE), often used for ensemble reliability. Can
    display bootstrapped confidence intervals.

    Args:
        ev (Evaluation): The Evaluation object with metrics.
        table_name (str, optional): Table name with metrics. Defaults to
                                     "bootstraped_lead_time_metrics".
    Raises:
        AssertionError: If "spread" or "mse" metrics aren't in the
                        table's metadata.
    """

    table = ev[table_name]["values"]
    meta = ev[table_name]["metadata"]
    metrics = meta["metrics"]
    assert "spread" in metrics, "The table must contain 'spread' metric."
    assert "mse" in metrics, "The table must contain 'mse' metric."

    table = (
        table
        .with_columns(pl.col(ev.lead_time_col).dt.total_hours())
        .with_columns(
            pl.col("^mse_.*$").sqrt().name
            .map(lambda x: x.replace("mse_", "rmse_"))
        )
    )

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.grid(True)

    for metric in ("rmse", "spread"):
        label_name = (
            f"{ev.experiment_name}: " +
            " ".join(x.upper() for x in metric.split("_"))
        )
        bootstrap_status = ""

        if meta.get("n_iter"):
            lq, uq = sorted(x for x in table.columns if f"{metric}_q" in x)
            ax.plot(
                table[ev.lead_time_col],
                table[f"{metric}_mean"],
                marker="o",
                label=label_name,
            )
            ax.fill_between(
                table[ev.lead_time_col],
                table[lq],
                table[uq],
                alpha=0.2,
                label=f"{meta['CI']}% CI ({label_name})",
            )
            bootstrap_status = f" Bootstrapped N:{meta['n_iter']}"
        else:
            ax.plot(
                table[ev.lead_time_col],
                table[metric],
                marker="o",
                label=label_name,
            )

        metric_name = " ".join(x.upper() for x in metric.split("_"))
        ax.set_title(
            f"{metric_name} by {ev.lead_time_col.capitalize()}"
            f"{bootstrap_status}"
        )
        ax.set_xlabel(ev.lead_time_col.capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.legend()

        fig.show()


def plot_rank_histogram(
    ev: Evaluation, step: int, table_name: str = "lead_time_rank_histogram"
):
    """
    Plots a rank histogram for a given lead time step.

    A rank histogram assesses the reliability of an ensemble forecast
    by showing the distribution of the observed rank within the ensemble.

    Args:
        ev (Evaluation): The Evaluation object containing the data.
        step (int): The lead time step (in hours) to plot for.
        table_name (str, optional): The name of the table containing the
                                     rank histogram data. Defaults to
                                     "lead_time_rank_histogram".
    """
    table = ev[table_name]["values"]
    meta = ev[table_name]["metadata"]

    bins = meta["bins"]

    # Get the counts for the given lead time step.
    counts = table.filter(
        pl.col(ev.lead_time_col) == pl.duration(hours=step)
    )["counts"].first()

    mean_counts = np.mean(counts)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.grid(True)
    ax.hist(bins[:-1], bins, weights=counts)
    ax.hlines(mean_counts, min(bins), max(bins), color="black", linestyle="--")
    ax.set_title(f"Rank Histogram | Step:{step}")

    fig.show()
