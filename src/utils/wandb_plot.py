"""Line, bar and grouped-bar plots of wandb runs, pulled straight from the wandb API.

Style: a figstyle rc + `save` callback (thesis, via render_plot.py `style: thesis`), or a legacy
tueplots bundle + `save_to` (YAMLs without `style`, bytes pinned). A list of WandbSeries inside
`series` stitches sequential (resumed) runs into one line. YAML schema: render_plot.py.
Auth: `wandb login` or $WANDB_API_KEY.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from dotenv import load_dotenv
from matplotlib.figure import Figure

from say_no.utils.figstyle import get_style, save_pinned  # noqa: F401  (get_style: re-exported)

load_dotenv()  # picks up WANDB_API_KEY from .env in cwd or any parent dir


@dataclass
class WandbSeries:
    """One line on a plot, sourced from a wandb run."""
    run: str                       # "entity/project/run_id"
    metric: str                    # e.g. "val/loss"
    label: str | None = None       # legend label; defaults to metric name
    x: str = "_step"               # x-axis column ("_step", "_runtime", "epoch", ...)
    smooth: int = 1                # rolling-mean window; 1 disables smoothing
    color: str | None = None       # hex; None = the style's colour cycle
    ls: str = "-"                  # line style; tells series apart in black-and-white print


@dataclass
class WandbBarItem:
    """One bar on a bar plot, sourced from a wandb run's summary scalar."""
    run: str                       # "entity/project/run_id"
    metric: str                    # summary key, e.g. "val/q_value_difference_lang"
    label: str | None = None       # x-tick label; defaults to run id
    color: str | None = None       # hex; None = the style's colour cycle


SeriesOrGroup = WandbSeries | Sequence[WandbSeries]


def fetch_series(spec: WandbSeries, samples: int = 10_000) -> pd.DataFrame:
    """Fetch a single metric history from wandb. Returns DataFrame with columns ['x','y']."""
    api = wandb.Api()
    run = api.run(spec.run)
    df = run.history(keys=[spec.metric], x_axis=spec.x, samples=samples, pandas=True)
    df = df.dropna(subset=[spec.metric])
    if df.empty:
        raise ValueError(f"No data for metric {spec.metric!r} in run {spec.run!r}.")
    if spec.smooth > 1:
        df[spec.metric] = (
            df[spec.metric].rolling(spec.smooth, min_periods=1, center=True).mean()
        )
    return df.rename(columns={spec.x: "x", spec.metric: "y"})[["x", "y"]]


def fetch_summary(spec: WandbBarItem) -> float:
    """Fetch a single scalar from a run's summary. Errors loudly if the key is missing."""
    api = wandb.Api()
    run = api.run(spec.run)
    val = run.summary.get(spec.metric)
    if val is None:
        raise ValueError(
            f"No summary value for metric {spec.metric!r} in run {spec.run!r}. "
            f"Available summary keys: {sorted(run.summary.keys())[:20]} ..."
        )
    return float(val)


def stitch(frames: Sequence[pd.DataFrame], gap: int = 0) -> pd.DataFrame:
    """Concatenate sequential runs, offsetting x so each continues from the prior end."""
    if not frames:
        raise ValueError("Need at least one frame to stitch.")
    out, offset = [], 0
    for df in frames:
        d = df.copy()
        d["x"] = d["x"] + offset
        out.append(d)
        offset = int(d["x"].iloc[-1]) + gap
    return pd.concat(out, ignore_index=True)


def _rc(rc: dict | None, bundle: str, font_scale: float, bundle_kwargs: dict | None) -> dict:
    """The caller's rc (a figstyle profile), or the legacy bundle + font_scale of a YAML."""
    if rc is None:
        return get_style(bundle, font_scale=font_scale, **(bundle_kwargs or {}))
    if bundle != "icml2024" or font_scale != 1.0 or bundle_kwargs:
        raise ValueError("pass either rc or bundle/font_scale/bundle_kwargs, not both")
    return rc


def _save(fig: Figure, save: Callable[[Figure], None] | None, save_to, **legacy_kwargs) -> None:
    """`save(fig)` (thesis: figstyle.save_figure) or the legacy pinned write to `save_to`."""
    if save is not None and save_to is not None:
        raise ValueError("pass either save or save_to, not both")
    if save is not None:
        save(fig)
    elif save_to is not None:
        save_pinned(fig, save_to, **legacy_kwargs)


def line_plot(
    series: Iterable[SeriesOrGroup],
    *,
    xlabel: str = "Training step",
    ylabel: str = "Value",
    title: str | None = None,
    save_to: str | Path | None = None,
    bundle: str = "icml2024",
    bundle_kwargs: dict | None = None,
    font_scale: float = 1.0,
    rc: dict | None = None,
    save: Callable[[Figure], None] | None = None,
    sci_x: bool = True,
    figsize: tuple[float, float] | None = None,
    grid: bool = True,
    show_diff: bool = False,
    ylim: tuple[float, float] | None = None,
    hline: tuple[float, str] | None = None,
    x_thousands: bool = False,
    log_y: bool = False,
    legend: bool = True,
    y_decimals: int | None = None,
) -> tuple[plt.Figure, plt.Axes | tuple[plt.Axes, plt.Axes]]:
    """Render a line plot. Each entry in `series` becomes one line.

    Pass a list-of-WandbSeries inside the iterable to stitch sequential runs into one line.
    Set `show_diff=True` (requires exactly two non-stitched series) to add a difference subplot.
    `ylim` fixes the y-range (a zoomed auto-range can make noise look like a trend), `hline` =
    (y, colour) draws a dotted reference line such as chance, `x_thousands` labels steps as 50k.
    """
    if x_thousands and sci_x:
        raise ValueError("x_thousands and sci_x both format the x-axis; set sci_x=False")
    if show_diff and (ylim or hline or x_thousands or log_y or y_decimals is not None or not legend):
        raise ValueError("ylim / hline / x_thousands / log_y / y_decimals / legend are single-panel options")
    rc = _rc(rc, bundle, font_scale, bundle_kwargs)

    resolved: list[tuple[pd.DataFrame, str, str | None]] = []
    for entry in series:
        if isinstance(entry, WandbSeries):
            df = fetch_series(entry)
            label, color, ls = entry.label or entry.metric, entry.color, entry.ls
        else:
            frames = [fetch_series(s) for s in entry]
            df = stitch(frames)
            label, color, ls = entry[0].label or entry[0].metric, entry[0].color, entry[0].ls
        resolved.append((df, label, color, ls))

    with plt.rc_context(rc):
        if show_diff:
            if len(resolved) != 2:
                raise ValueError("show_diff=True requires exactly two series.")
            fig, (ax_main, ax_diff) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            for df, label, color, ls in resolved:
                ax_main.plot(df["x"], df["y"], label=label, color=color, ls=ls)
            ax_main.set_ylabel(ylabel)
            if title:
                ax_main.set_title(title)
            if any(lbl for _, lbl, _, _ in resolved):
                ax_main.legend()
            if grid:
                ax_main.grid(True, alpha=0.3)

            (df_a, _, _, _), (df_b, _, _, _) = resolved
            common_x = pd.merge(df_a, df_b, on="x", suffixes=("_a", "_b"))
            ax_diff.plot(common_x["x"], common_x["y_a"] - common_x["y_b"])
            ax_diff.set_xlabel(xlabel)
            ax_diff.set_ylabel(f"{ylabel} (diff)")
            if grid:
                ax_diff.grid(True, alpha=0.3)
            if sci_x:
                ax_diff.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
            axes = (ax_main, ax_diff)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            for df, label, color, ls in resolved:
                ax.plot(df["x"], df["y"], label=label, color=color, ls=ls)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title:
                ax.set_title(title)
            if legend and any(lbl for _, lbl, _, _ in resolved):
                ax.legend()
            if grid:
                ax.grid(True, alpha=0.3)
            if sci_x:
                ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
            if x_thousands:
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
            if log_y:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(*ylim)
            if y_decimals is not None:
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter(f"%.{y_decimals}f"))
            if hline is not None:
                ax.axhline(hline[0], color=hline[1], linestyle=":", linewidth=1.0, zorder=0)
            axes = ax

        _save(fig, save, save_to)
        return fig, axes


def _headroom_for_annotations(ax, log_y: bool) -> None:
    """Lift the top of the y-axis so the value printed on the tallest bar still fits.

    tueplots turns constrained layout on, which clips rather than grows the figure, so an
    annotation that overflows the axes is simply lost.
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 2.2 if log_y else hi * 1.15)


def bar_plot(
    items: Iterable[WandbBarItem],
    *,
    xlabel: str = "",
    ylabel: str = "Value",
    title: str | None = None,
    save_to: str | Path | None = None,
    bundle: str = "icml2024",
    bundle_kwargs: dict | None = None,
    font_scale: float = 1.0,
    rc: dict | None = None,
    save: Callable[[Figure], None] | None = None,
    figsize: tuple[float, float] | None = None,
    grid: bool = True,
    log_y: bool = False,
    rotate_xticks: float = 0.0,
    annotate: bool = True,
    annotate_fmt: str = "{:.2f}",
) -> tuple[plt.Figure, plt.Axes]:
    """Render a vertical bar plot. Each entry in `items` becomes one bar.

    Each bar's height is fetched from `run.summary[metric]`. Use this for
    sweep-style ablations (e.g. metric vs hyperparameter setting).
    """
    rc = _rc(rc, bundle, font_scale, bundle_kwargs)
    items = list(items)
    if not items:
        raise ValueError("bar_plot requires at least one item.")
    values = [fetch_summary(i) for i in items]
    labels = [i.label or i.run.split("/")[-1] for i in items]
    colors = [i.color for i in items]
    if any(c is None for c in colors) and any(c is not None for c in colors):
        raise ValueError("bar_plot: give every bar a colour or none")

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(labels, values, color=None if colors[0] is None else colors)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, axis="y", alpha=0.3)
        if rotate_xticks:
            ax.tick_params(axis="x", rotation=rotate_xticks)
        if annotate:
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    annotate_fmt.format(v),
                    ha="center", va="bottom", fontsize=plt.rcParams["xtick.labelsize"],
                )
            _headroom_for_annotations(ax, log_y)
        _save(fig, save, save_to, bbox_inches="tight")
        return fig, ax


def grouped_bar_plot(
    groups: Iterable[Sequence[WandbBarItem]],
    *,
    group_labels: Sequence[str],
    bar_labels: Sequence[str],
    bar_colors: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "Value",
    title: str | None = None,
    save_to: str | Path | None = None,
    bundle: str = "icml2024",
    bundle_kwargs: dict | None = None,
    font_scale: float = 1.0,
    rc: dict | None = None,
    save: Callable[[Figure], None] | None = None,
    figsize: tuple[float, float] | None = None,
    grid: bool = True,
    log_y: bool = False,
    rotate_xticks: float = 0.0,
    annotate: bool = True,
    annotate_fmt: str = "{:.2f}",
    bar_width_total: float = 0.8,
    legend_above: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a grouped bar plot. Each group becomes a cluster of N bars on the x-axis.

    `groups` is a sequence of groups; each group is a sequence of `WandbBarItem` of equal
    length N. `group_labels` labels the x-axis ticks (one per group); `bar_labels` labels
    the legend entries (one per bar within a group, shared across groups). Bar colours
    default to the active matplotlib cycle so the legend matches the rest of the figures
    in the thesis.
    """
    import numpy as np

    rc = _rc(rc, bundle, font_scale, bundle_kwargs)
    groups_list = [list(g) for g in groups]
    if not groups_list:
        raise ValueError("grouped_bar_plot requires at least one group.")
    n_per_group = len(groups_list[0])
    if any(len(g) != n_per_group for g in groups_list):
        raise ValueError("All groups must have the same number of bars.")
    if len(group_labels) != len(groups_list):
        raise ValueError("group_labels length must equal number of groups.")
    if len(bar_labels) != n_per_group:
        raise ValueError("bar_labels length must equal bars per group.")

    values: list[list[float]] = [[fetch_summary(item) for item in g] for g in groups_list]

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if bar_colors is None:
            bar_colors = [cycle_colors[i % len(cycle_colors)] for i in range(n_per_group)]
        x_centers = np.arange(len(groups_list))
        bar_w = bar_width_total / n_per_group
        for i in range(n_per_group):
            offsets = x_centers + (i - (n_per_group - 1) / 2) * bar_w
            heights = [values[g_i][i] for g_i in range(len(groups_list))]
            bars = ax.bar(offsets, heights, bar_w, label=bar_labels[i], color=bar_colors[i])
            if annotate:
                for bar, v in zip(bars, heights):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        annotate_fmt.format(v),
                        ha="center", va="bottom", fontsize=plt.rcParams["xtick.labelsize"],
                    )
        ax.set_xticks(x_centers)
        ax.set_xticklabels(group_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        if annotate:
            _headroom_for_annotations(ax, log_y)
        if grid:
            ax.grid(True, axis="y", alpha=0.3)
        if rotate_xticks:
            ax.tick_params(axis="x", rotation=rotate_xticks)
        if legend_above:
            # a narrow panel has no empty corner left for a legend box
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=n_per_group, frameon=False, borderpad=0.0,
                      handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
        else:
            ax.legend()
        _save(fig, save, save_to, bbox_inches="tight")
        return fig, ax
