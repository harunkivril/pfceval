import polars as pl
import matplotlib.pyplot as plt

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from .evaluation import Evaluation
plt.style.use("seaborn-v0_8-notebook")


def stack_overall_metrics(
        *evals: Evaluation,
        table_name: str = "overall_metrics",
        metrics: list[str] | None = None,
):
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

    stacked = pl.concat(stacked)
    return stacked


def get_common_metrics(evals: list[Evaluation], table_name: str):
    metric_set = set(evals[0][table_name]["metadata"]['metrics'])
    for ev in evals:
        metric_set = metric_set.intersection(
            ev[table_name]["metadata"]['metrics'])
    return list(metric_set)


def plot_lead_time_metrics(
    *evals: Evaluation,
    table_name: str = "lead_time_metrics",
    metrics: list[str] | None = None,
):
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
            table = table.with_columns(
                pl.col(ev.lead_time_col).dt.total_hours()
            )

            label_name = (
                f"{ev.experiment_name}: "
                + " ".join(x.upper() for x in metric.split("_"))
            )
            bootstrap_status = ""

            if meta.get("n_iter"):
                lq, uq = sorted(x for x in table.columns if f"{metric}_q" in x)
                ax.plot(
                    table[ev.lead_time_col],
                    table[f"{metric}_mean"],
                    marker="o",
                    label=label_name
                )
                ax.fill_between(
                    table[ev.lead_time_col],
                    table[lq],
                    table[uq],
                    alpha=0.2,
                    label=f"{meta['CI']}% CI ({label_name})"
                )
                bootstrap_status = f" Bootstrapped N:{meta['n_iter']}"
            else:
                ax.plot(
                    table[ev.lead_time_col],
                    table[metric],
                    marker="o",
                    label=label_name
                )

        metric_name = " ".join(x.upper() for x in metric.split("_"))
        ax.set_title(
            f"{metric_name} by {evals[0].lead_time_col.capitalize()}"
            + f"{bootstrap_status}"
        )
        ax.set_xlabel(evals[0].lead_time_col.capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.legend()

        fig.show()


def plot_location_metrics(
        step,
        evaluation: Evaluation,
        compare_with: Evaluation = None,
        table_name="lead_time_location_metrics",
        metrics: str | None = None,
        cmap: str = "viridis",
        tol: float = 0.01,
        dot_size: int = 10
):

    meta = evaluation[table_name]["metadata"]
    table = evaluation[table_name]["values"].filter(
        step=pl.duration(hours=step))
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
            on=evaluation.location_id_col
        )

    if metrics is None:
        metrics = meta["metrics"]
    else:
        metrics = [metrics] if isinstance(metrics, str) else metrics
        assert all(metric in meta["metrics"] for metric in metrics)

    for metric in metrics:
        fig, ax = plt.subplots(
            figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        plt.set_cmap(cmap)
        # Add basic features
        ax.set_global()
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        metric_name = " ".join(x.upper() for x in metric.split("_"))
        if compare_with:
            value = (
                table.select(pl.col(metric) - pl.col(f"{metric}_right"))
                .to_numpy().squeeze()
            )
            metric_name = (
                f"{metric_name} ({evaluation.experiment_name}" 
                + f"- {compare_with.experiment_name})"
            )
        else:
            value = table.select(pl.col(metric)).to_numpy().squeeze()
        lat = table.select(pl.col("latitude")).to_numpy().squeeze()
        lon = table.select(pl.col("longitude")).to_numpy().squeeze()

        # Add coastlines
        ax.coastlines()
        ax.set_extent(
            [lon.min()-tol, lon.max()+tol, lat.min()-tol, lat.max()+tol],
            crs=ccrs.PlateCarree()
        )
        vmin, vmax = np.quantile(value, 0.01), np.quantile(value, 0.99)
        img = ax.scatter(
            lon, lat, s=dot_size, c=value, vmax=vmax, vmin=vmin, alpha=0.7)
        cbar = fig.colorbar(img)
        cbar.set_label(metric)

        # Add gridlines
        ax.gridlines(draw_labels=True)

        # Add a title
        ax.set_title(f"{metric_name} Step:{step} ", fontdict={"size": 16})
        fig.tight_layout()


def rebin_obs_bar(obs_bar: pl.DataFrame, nbins: int | None):
    if not nbins:
        return obs_bar

    new_probs = 1/nbins * (np.arange(nbins) + 1/2)
    assigned = np.array(
        [np.argmin(abs(new_probs - p)) for p in obs_bar["prob"]]
    )
    obs_bar = obs_bar.with_columns(prob=new_probs[assigned])
    obs_bar = obs_bar.group_by("prob").agg(
        (pl.exclude("count") * (pl.col("count")/pl.col("count").sum())).sum(),
        count=pl.col("count").sum()
    )
    return obs_bar.sort("prob")


def prep_reliability_ax():
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(4, 3)
    rel_ax = fig.add_subplot(gs[:2, :3])
    return gs, fig, rel_ax


def plot_reliability_diagram(
        *evals: Evaluation,
        step: int,
        table_name: str,
        nbins: int | None = None
):
    if len(evals) > 3:
        raise ValueError("Too many experiments to plot. Suported max is 3.")

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
            .to_numpy().squeeze().round(3)
        )
        bs = round(rel - res + unc, 3)

        obs_bar = rebin_obs_bar(obs_bar, nbins)
        label_name = f"{ev.experiment_name}|Rel:{rel}|Res:{res}|BS:{bs}"
        meta = ev[table_name]["metadata"]
        lq, uq = sorted(x for x in obs_bar.columns if "obs_bar_q" in x)
        rel_ax.plot(
            obs_bar["prob"],
            obs_bar["obs_bar_mean"],
            marker=".",
            label=label_name,
            c=color_list[i]
        )
        rel_ax.fill_between(
            obs_bar["prob"],
            obs_bar[lq],
            obs_bar[uq],
            alpha=0.2,
            color=color_list[i]
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
            alpha=0.5
        )
        hist_ax.set_ylabel("Counts")
        hist_ax.set_xlim(rel_ax.get_xlim())

    pos_class_ratio = obs_bar["mean_group"].first()
    hist_ax.set_title(f"Positive Class Ratio: {pos_class_ratio:.4f}")

    rel_ax.legend()
    rel_ax.set_title(
        f"Reliability Diagram | Th:{th} | CI:{meta['CI']} N:{meta['n_iter']}")
    rel_ax.set_ylabel("Observed Frequency")
    rel_ax.set_xlabel("Predicted Probability")
    fig.tight_layout()


def plot_spread_rmse(ev, table_name="bootstraped_lead_time_metrics"):

    table = ev[table_name]["values"]
    meta = ev[table_name]["metadata"]
    metrics = meta["metrics"]
    assert "spread" in metrics
    assert "mse" in metrics

    table = table.with_columns(
        pl.col(ev.lead_time_col).dt.total_hours(),
    )
    table = table.with_columns(
        pl.col("^mse_.*$").sqrt().name.map(lambda x: x.replace("mse_", "rmse_"))
    )

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.grid(True)

    for metric in ("rmse", "spread"):
        label_name = (
            f"{ev.experiment_name}: "
            + " ".join(x.upper() for x in metric.split("_"))
        )
        bootstrap_status = ""

        if meta.get("n_iter"):
            lq, uq = sorted(x for x in table.columns if f"{metric}_q" in x)
            ax.plot(
                table[ev.lead_time_col],
                table[f"{metric}_mean"],
                marker="o",
                label=label_name
            )
            ax.fill_between(
                table[ev.lead_time_col],
                table[lq],
                table[uq],
                alpha=0.2,
                label=f"{meta['CI']}% CI ({label_name})"
            )
            bootstrap_status = f" Bootstrapped N:{meta['n_iter']}"
        else:
            ax.plot(
                table[ev.lead_time_col],
                table[metric],
                marker="o",
                label=label_name
            )

        metric_name = " ".join(x.upper() for x in metric.split("_"))
        ax.set_title(
            f"{metric_name} by {ev.lead_time_col.capitalize()}"
            + f"{bootstrap_status}"
        )
        ax.set_xlabel(ev.lead_time_col.capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.legend()

        fig.show()