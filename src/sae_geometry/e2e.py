import pandas as pd
import tqdm
import wandb
from wandb.apis.public import Run
from wandb.apis.public.runs import Runs

def get_df_gpt2() -> pd.DataFrame:
    api = wandb.Api()
    project = "sparsify/gpt2"
    runs = api.runs(project)

    d_resid = 768

    df = create_run_df(runs)

    assert df["model_name"].nunique() == 1

    # Ignore runs that have an L0 bigger than d_resid
    df = df.loc[df["L0"] <= d_resid]
    return df

def create_run_df(
    runs: Runs, per_layer_metrics: bool = True, use_run_name: bool = False, grad_norm: bool = True
) -> pd.DataFrame:
    run_info = []
    for run in tqdm.tqdm(runs, total=len(runs), desc="Processing runs"):
        if run.state != "finished":
            print(f"Run {run.name} is not finished, skipping")
            continue
        sae_pos = run.config["saes"]["sae_positions"]
        if isinstance(sae_pos, list):
            if len(sae_pos) > 1:
                raise ValueError("More than one SAE position found")
            sae_pos = sae_pos[0]
        sae_layer = int(sae_pos.split(".")[1])

        kl_coeff = None
        in_to_orig_coeff = None
        out_to_in_coeff = None
        if "logits_kl" in run.config["loss"] and run.config["loss"]["logits_kl"] is not None:
            kl_coeff = run.config["loss"]["logits_kl"]["coeff"]
        if "in_to_orig" in run.config["loss"] and run.config["loss"]["in_to_orig"] is not None:
            in_to_orig_coeff = run.config["loss"]["in_to_orig"]["total_coeff"]
        if "out_to_in" in run.config["loss"] and run.config["loss"]["out_to_in"] is not None:
            out_to_in_coeff = run.config["loss"]["out_to_in"]["coeff"]

        if use_run_name:
            run_type = _get_run_type_using_names(run.name)
        else:
            run_type = _get_run_type(kl_coeff, in_to_orig_coeff, out_to_in_coeff)

        explained_var_layers = {}
        explained_var_ln_layers = {}
        recon_loss_layers = {}
        if per_layer_metrics:
            # The out_to_in in the below is to handle the e2e+recon loss runs which specified
            # future layers in the in_to_orig but not the output of the SAE at the current layer
            # (i.e. at hook_resid_post). Note that now if you leave in_to_orig as None, it will
            # default to calculating in_to_orig at all layers at hook_resid_post.
            # The explained variance at each layer
            explained_var_layers = _extract_per_layer_metrics(
                run=run,
                metric_key="loss/eval/in_to_orig/explained_variance",
                metric_name_prefix="explained_var_layer",
                sae_layer=sae_layer,
                sae_pos=sae_pos,
            )

            explained_var_ln_layers = _extract_per_layer_metrics(
                run=run,
                metric_key="loss/eval/in_to_orig/explained_variance_ln",
                metric_name_prefix="explained_var_ln_layer",
                sae_layer=sae_layer,
                sae_pos=sae_pos,
            )

            recon_loss_layers = _extract_per_layer_metrics(
                run=run,
                metric_key="loss/eval/in_to_orig",
                metric_name_prefix="recon_loss_layer",
                sae_layer=sae_layer,
                sae_pos=sae_pos,
            )

        if "dict_size_to_input_ratio" in run.config["saes"]:
            ratio = float(run.config["saes"]["dict_size_to_input_ratio"])
        else:
            # local runs didn't store the ratio in the config for these runs
            ratio = float(run.name.split("ratio-")[1].split("_")[0])

        out_to_in = None
        explained_var = None
        explained_var_ln = None
        if f"loss/eval/out_to_in/{sae_pos}" in run.summary_metrics:
            out_to_in = run.summary_metrics[f"loss/eval/out_to_in/{sae_pos}"]
            explained_var = run.summary_metrics[f"loss/eval/out_to_in/explained_variance/{sae_pos}"]
            try:
                explained_var_ln = run.summary_metrics[
                    f"loss/eval/out_to_in/explained_variance_ln/{sae_pos}"
                ]
            except KeyError:
                explained_var_ln = None

        try:
            kl = run.summary_metrics["loss/eval/logits_kl"]
        except KeyError:
            kl = None

        mean_grad_norm = None
        if grad_norm:
            # Check if "mean_grad_norm" is in the run summary, if not, we need to calculate it
            if "mean_grad_norm" in run.summary:
                mean_grad_norm = run.summary["mean_grad_norm"]
            else:
                grad_norm_history = run.history(keys=["grad_norm"], samples=2000)
                # Get the mean of grad norms after the first 10000 steps
                mean_grad_norm = grad_norm_history.loc[
                    grad_norm_history["_step"] > 10000, "grad_norm"
                ].mean()

                run.summary["mean_grad_norm"] = mean_grad_norm
                run.summary.update()

        run_info.append(
            {
                "name": run.name,
                "id": run.id,
                "sae_pos": sae_pos,
                "model_name": run.config["tlens_model_name"],
                "run_type": run_type,
                "layer": sae_layer,
                "seed": run.config["seed"],
                "n_samples": run.config["n_samples"],
                "lr": run.config["lr"],
                "ratio": ratio,
                "sparsity_coeff": run.config["loss"]["sparsity"]["coeff"],
                "in_to_orig_coeff": in_to_orig_coeff,
                "kl_coeff": kl_coeff,
                "out_to_in": out_to_in,
                "L0": run.summary_metrics[f"sparsity/eval/L_0/{sae_pos}"],
                "explained_var": explained_var,
                "explained_var_ln": explained_var_ln,
                "CE_diff": run.summary_metrics["performance/eval/difference_ce_loss"],
                "CELossIncrease": -run.summary_metrics["performance/eval/difference_ce_loss"],
                "alive_dict_elements": run.summary_metrics[
                    f"sparsity/alive_dict_elements/{sae_pos}"
                ],
                "mean_grad_norm": mean_grad_norm,
                **explained_var_layers,
                **explained_var_ln_layers,
                **recon_loss_layers,
                "sum_recon_loss": sum(recon_loss_layers.values()),
                "kl": kl,
            }
        )
    df = pd.DataFrame(run_info)
    return df

def _extract_per_layer_metrics(
    run: Run, metric_key: str, metric_name_prefix: str, sae_layer: int, sae_pos: str
) -> dict[str, float]:
    """Extract the per layer metrics from the run summary metrics.

    Note that the layer indices correspond to those collected before that layer. E.g. those from
    hook_resid_pre rather than hook_resid_post.

    Args:
        run: The run to extract the metrics from.
        metric_key: The key to use to extract the metrics from the run summary.
        metric_name_prefix: The prefix to use for the metric names in the returned dictionary.
        sae_layer: The layer number of the SAE.
        sae_pos: The position of the SAE.
    """

    results: dict[str, float] = {}
    for key, value in run.summary_metrics.items():
        if not key.startswith(f"{metric_key}/blocks"):
            # We don't care about the other metrics
            continue
        layer_num_str, hook_pos = key.split(f"{metric_key}/blocks.")[1].split(".")
        if "pre" in hook_pos:
            layer_num = int(layer_num_str)
        elif "post" in hook_pos:
            layer_num = int(layer_num_str) + 1
        else:
            raise ValueError(f"Unknown hook position: {hook_pos}")
        results[f"{metric_name_prefix}-{layer_num}"] = value

    # Overwrite the SAE layer with the out_to_in for that layer. This is so that we get the
    # reconstruction/variance at the output of the SAE rather than the input
    out_to_in_prefix = metric_key.replace("in_to_orig", "out_to_in")
    results[f"{metric_name_prefix}-{sae_layer}"] = run.summary_metrics[
        f"{out_to_in_prefix}/{sae_pos}"
    ]
    return results


def _get_run_type(
    kl_coeff: float | None, in_to_orig_coeff: float | None, out_to_in_coeff: float | None
) -> str:
    if (
        kl_coeff is not None
        and in_to_orig_coeff is not None
        and kl_coeff > 0
        and in_to_orig_coeff > 0
    ):
        if out_to_in_coeff is not None and out_to_in_coeff > 0:
            return "downstream_all"
        else:
            return "downstream"
    if (
        kl_coeff is not None
        and out_to_in_coeff is not None
        and in_to_orig_coeff is None
        and kl_coeff > 0
        and out_to_in_coeff > 0
    ):
        return "e2e_local"
    if (
        kl_coeff is not None
        and kl_coeff > 0
        and (out_to_in_coeff is None or out_to_in_coeff == 0)
        and in_to_orig_coeff is None
    ):
        return "e2e"
    return "local"


def _get_run_type_using_names(run_name: str) -> str:
    if "logits-kl-1.0" in run_name and "in-to-orig" not in run_name:
        return "e2e"
    if "logits-kl-" in run_name and "in-to-orig" in run_name:
        return "downstream"
    return "local"

import os
from pathlib import Path

REPO_ROOT = (
    Path(os.environ["GITHUB_WORKSPACE"]) if os.environ.get("CI") else Path(__file__).parent.parent
)

import logging
from logging.config import dictConfig
from pathlib import Path

DEFAULT_LOGFILE = Path(__file__).resolve().parent.parent / "logs" / "logs.log"


def setup_logger(logfile: Path = DEFAULT_LOGFILE) -> logging.Logger:
    """Setup a logger to be used in all modules in the library.

    Sets up logging configuration with a console handler and a file handler.
    Console handler logs messages with INFO level, file handler logs WARNING level.
    The root logger is configured to use both handlers.

    Returns:
        logging.Logger: A configured logger object.

    Example:
        >>> logger = setup_logger()
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
    """
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)

    logging_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(logfile),
                "formatter": "default",
                "level": "WARNING",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    }

    dictConfig(logging_config)
    return logging.getLogger()


logger = setup_logger()


from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

def plot_per_layer_metric(
    df: pd.DataFrame,
    run_ids: Mapping[int, Mapping[str, str]],
    metric: str,
    final_layer: int,
    run_types: Sequence[str],
    out_file: str | Path | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    legend_label_cols_and_precision: list[tuple[str, int]] | None = None,
    legend_title: str | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    horz_layout: bool = False,
    show_ax_titles: bool = True,
    save_svg: bool = True,
) -> None:
    """
    Plot the per-layer metric (explained variance or reconstruction loss) for different run types.

    Args:
        df: DataFrame containing the filtered data for the specific layer.
        run_ids: The run IDs to use. Format: {layer: {run_type: run_id}}.
        metric: The metric to plot ('explained_var' or 'recon_loss').
        final_layer: The final layer to plot up to.
        run_types: The run types to include in the plot.
        out_file: The filename which the plot will be saved as.
        ylim: The y-axis limits.
        legend_label_cols_and_precision: Columns in df that should be used for the legend, along
            with their precision. Added in addition to the run type.
        legend_title: The title of the legend.
        styles: The styles to use.
        horz_layout: Whether to use a horizontal layout for the subplots. Requires sae_layers to be
            exactly [2, 6, 10]. Ignores legend_label_cols_and_precision if True.
        show_ax_titles: Whether to show titles for each subplot.
        save_svg: Whether to save the plot as an SVG file in addition to PNG. Default is True.
    """
    metric_names = {
        "explained_var": "Explained Variance",
        "explained_var_ln": "Explained Variance\nof Normalized Activations",
        "recon_loss": "Reconstruction MSE",
    }
    metric_name = metric_names.get(metric, metric)

    sae_layers = list(run_ids.keys())
    n_sae_layers = len(sae_layers)

    if horz_layout:
        assert sae_layers == [2, 6, 10]
        fig, axs = plt.subplots(
            1, n_sae_layers, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2, 1.2]}
        )
        legend_label_cols_and_precision = None
    else:
        fig, axs = plt.subplots(n_sae_layers, 1, figsize=(5, 3.5 * n_sae_layers))
    axs = np.atleast_1d(axs)  # type: ignore

    def plot_metric(
        ax: plt.Axes,
        plot_df: pd.DataFrame,
        sae_layer: int,
        xs: NDArray[np.signedinteger[Any]],
    ) -> None:
        for _, row in plot_df.iterrows():
            run_type = row["run_type"]
            assert isinstance(run_type, str)
            legend_label = styles[run_type]["label"] if styles is not None else run_type
            if legend_label_cols_and_precision is not None:
                assert all(
                    col in row for col, _ in legend_label_cols_and_precision
                ), f"Legend label cols not found in row: {row}"
                metric_strings = [
                    f"{col}={format(row[col], f'.{prec}f')}"
                    for col, prec in legend_label_cols_and_precision
                ]
                legend_label += f" ({', '.join(metric_strings)})"
            ys = [row[f"{metric}_layer-{i}"] for i in range(sae_layer, final_layer + 1)]
            kwargs = styles[run_type] if styles is not None else {}
            ax.plot(xs, ys, **kwargs)

    for i, sae_layer in enumerate(sae_layers):
        layer_df = df.loc[df["id"].isin(list(run_ids[sae_layer].values()))]

        ax = axs[i]

        xs = np.arange(sae_layer, final_layer + 1)
        for run_type in run_types:
            plot_metric(ax, layer_df.loc[layer_df["run_type"] == run_type], sae_layer, xs)

        if show_ax_titles:
            ax.set_title(f"SAE Layer {sae_layer}", fontweight="bold")
        ax.set_xlabel("Model Layer")
        if (not horz_layout) or i == 0:
            ax.legend(title=legend_title, loc="best")
            ax.set_ylabel(metric_name)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_ylim(ylim)

    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file, dpi=400)
        logger.info(f"Saved to {out_file}")
        if save_svg:
            plt.savefig(Path(out_file).with_suffix(".svg"))
    plt.close()

def plot_facet(
    df: pd.DataFrame,
    xs: Sequence[str],
    y: str,
    facet_by: str,
    line_by: str,
    line_by_vals: Sequence[str] | None = None,
    sort_by: str | None = None,
    xlabels: Sequence[str | None] | None = None,
    ylabel: str | None = None,
    suptitle: str | None = None,
    facet_vals: Sequence[Any] | None = None,
    xlims: Sequence[Mapping[Any, tuple[float | None, float | None]] | None] | None = None,
    xticks: Sequence[tuple[list[float], list[str]] | None] | None = None,
    yticks: tuple[list[float], list[str]] | None = None,
    ylim: Mapping[Any, tuple[float | None, float | None]] | None = None,
    styles: Mapping[Any, Mapping[str, Any]] | None = None,
    title: Mapping[Any, str] | None = None,
    legend_title: str | None = None,
    legend_pos: str = "lower right",
    axis_formatter: Callable[[Sequence[plt.Axes]], None] | None = None,
    out_file: str | Path | None = None,
    plot_type: Literal["line", "scatter"] = "line",
    save_svg: bool = True,
) -> None:
    """Line plot with multiple x-axes and one y-axis between them. One line for each run type.

    Args:
        df: DataFrame containing the data.
        xs: The variables to plot on the x-axes.
        y: The variable to plot on the y-axis.
        facet_by: The variable to facet the plot by.
        line_by: The variable to draw lines for.
        line_by_vals: The values to draw lines for. If None, all unique values will be used.
        sort_by: The variable governing how lines are drawn between points. If None, lines will be
            drawn based on the y value.
        title: The title of the plot.
        xlabel: The labels for the x-axes.
        ylabel: The label for the y-axis.
        out_file: The filename which the plot will be saved as.
        run_types: The run types to include in the plot.
        xlims: The x-axis limits for each x-axis for each layer.
        xticks: The x-ticks for each x-axis.
        yticks: The y-ticks for the y-axis.
        ylim: The y-axis limits for each layer.
        styles: The styles to use for each line. If None, default styles will be used.
        title: The title for each row of the plot.
        legend_title: The title for the legend.
        axis_formatter: A function to format the axes, e.g. to add "better" labels.
        out_file: The filename which the plot will be saved as.
        plot_type: The type of plot to create. Either "line" or "scatter".
        save_svg: Whether to save the plot as an SVG file in addition to png. Default is True.
    """

    num_axes = len(xs)
    if facet_vals is None:
        facet_vals = sorted(df[facet_by].unique())
    if sort_by is None:
        sort_by = y

    # TODO: For some reason the title is not centered at x=0.5. Fix
    xtitle_pos = 0.513

    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#f5f6fc"})
    fig_width = 4 * num_axes
    fig = plt.figure(figsize=(fig_width, 4 * len(facet_vals)), constrained_layout=True)
    subfigs = fig.subfigures(len(facet_vals))
    subfigs = np.atleast_1d(subfigs)  # type: ignore

    # Get all unique line values from the entire DataFrame
    all_line_vals = df[line_by].unique()
    if line_by_vals is not None:
        assert all(
            val in all_line_vals for val in line_by_vals
        ), f"Invalid line values: {line_by_vals}"
        sorted_line_vals = line_by_vals
    else:
        sorted_line_vals = sorted(all_line_vals, key=str if df[line_by].dtype == object else float)

    colors = sns.color_palette("tab10", n_colors=len(sorted_line_vals))
    for subfig, facet_val in zip(subfigs, facet_vals, strict=False):
        axs = subfig.subplots(1, num_axes)
        facet_df = df.loc[df[facet_by] == facet_val]
        for line_val, color in zip(sorted_line_vals, colors, strict=True):
            data = facet_df.loc[facet_df[line_by] == line_val]
            line_style = {
                "label": line_val,
                "marker": "o",
                "linewidth": 1.1,
                "color": color,
                "linestyle": "-" if plot_type == "line" else "None",
            }  # default
            line_style.update(
                {} if styles is None else styles.get(line_val, {})
            )  # specific overrides
            if not data.empty:
                # draw the lines between points based on the y value
                data = data.sort_values(sort_by)
                for i in range(num_axes):
                    if plot_type == "scatter":
                        axs[i].scatter(data[xs[i]], data[y], **line_style)
                    elif plot_type == "line":
                        axs[i].plot(data[xs[i]], data[y], **line_style)
                    else:
                        raise ValueError(f"Unknown plot type: {plot_type}")
            else:
                # Add empty plots for missing line values to ensure they appear in the legend
                for i in range(num_axes):
                    axs[i].plot([], [], **line_style)

        if facet_val == facet_vals[-1]:
            axs[0].legend(title=legend_title or line_by, loc=legend_pos)

        for i in range(num_axes):
            if xlims is not None and xlims[i] is not None:
                xmin, xmax = xlims[i][facet_val]  # type: ignore
                axs[i].set_xlim(xmin=xmin, xmax=xmax)
            if ylim is not None:
                ymin, ymax = ylim[facet_val]
                axs[i].set_ylim(ymin=ymin, ymax=ymax)

        # Set a title above the subplots to show the layer number
        row_title = title[facet_val] if title is not None else None
        subfig.suptitle(row_title, fontweight="bold", x=xtitle_pos)
        for i in range(num_axes):
            axs[i].set_xlabel(xlabels[i] if xlabels is not None else xs[i])
            if i == 0:
                axs[i].set_ylabel(ylabel or y)
            if xticks is not None and xticks[i] is not None:
                ticks, labels = xticks[i]  # type: ignore
                axs[i].set_xticks(ticks, labels=labels)
            if yticks is not None:
                axs[i].set_yticks(yticks[0], yticks[1])

        if axis_formatter is not None:
            axis_formatter(axs)

    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold", x=xtitle_pos)

    if out_file is not None:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=400)
        logger.info(f"Saved to {out_file}")
        if save_svg:
            plt.savefig(Path(out_file).with_suffix(".svg"))

    plt.close(fig)


import matplotlib.pyplot as plt

# Runs with constant CE loss increase for each layer. Values represent wandb run IDs.
SIMILAR_CE_RUNS = {
    2: {"local": "ue3lz0n7", "e2e": "ovhfts9n", "downstream": "visi12en"},
    6: {"local": "1jy3m5j0", "e2e": "zgdpkafo", "downstream": "2lzle2f0"},
    10: {"local": "m2hntlav", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}
# Runs with similar L0 loss increase for each layer. Values represent wandb run IDs.
SIMILAR_L0_RUNS = {
    2: {"local": "6vtk4k51", "e2e": "bst0prdd", "downstream": "e26jflpq"},
    6: {"local": "jup3glm9", "e2e": "tvj2owza", "downstream": "2lzle2f0"},
    10: {"local": "5vmpdgaz", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}
# Runs with similar alive dictionary elements. Values represent wandb run IDs.
SIMILAR_ALIVE_ELEMENTS_RUNS = {
    2: {"local": "6vtk4k51", "e2e": "0z98g9pf", "downstream": "visi12en"},
    6: {"local": "h9hrelni", "e2e": "tvj2owza", "downstream": "p9zmh62k"},
    10: {"local": "5vmpdgaz", "e2e": "vnfh4vpi", "downstream": "f2fs7hk3"},
}

SIMILAR_RUN_INFO = {
    "CE": SIMILAR_CE_RUNS,
    "l0": SIMILAR_L0_RUNS,
    "alive_elements": SIMILAR_ALIVE_ELEMENTS_RUNS,
}

STYLE_MAP = {
    "local": {"marker": "^", "color": "#f0a70a", "label": "local"},
    "e2e": {"marker": "o", "color": "#518c31", "label": "e2e"},
    "downstream": {"marker": "X", "color": plt.get_cmap("tab20b").colors[2], "label": "e2e+ds"},  # type: ignore[reportAttributeAccessIssue]
}