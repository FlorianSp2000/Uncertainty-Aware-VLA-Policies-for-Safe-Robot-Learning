"""Render a single paper plot from a YAML config.

Takes one YAML path. Thesis figures are rebuilt through src/scripts/render_thesis_figures.py.

YAML schemas
------------
Line plot (time-series over training steps):
    plot_type: line
    output: src/results/<name>.pdf
    title: "..."
    xlabel: "Training step"        # optional, defaults to "Training step"
    ylabel: "..."
    style: thesis                  # thesis figure: say_no.utils.figstyle THESIS profile,
    width: 0.49                    #   width = include width (fraction of \\textwidth),
    height: 2.0                    #   height in inches; exact size, provenance in the PDF metadata
    # without `style` (legacy YAMLs, bytes pinned), instead:
    bundle: icml2024               # optional
    bundle_kwargs: {usetex: false} # optional
    font_scale: 1.0                # optional, multiplies every font size in the bundle
    figsize: [w, h]                # optional
    show_diff: false               # optional, requires exactly 2 series
    sci_x: true                    # optional, scientific notation on the x-axis
    x_thousands: false             # optional, steps as "50k" (needs sci_x: false)
    ylim: [0.4, 0.6]               # optional, fixed y-range
    hline: {y: 0.5, color: chance} # optional, dotted reference line
    log_y: false                   # optional, logarithmic y-axis
    legend: true                   # optional, false hides the legend (single-series plots)
    y_decimals: 2                  # optional, fixed decimals on the y tick labels
    series:
      - run: entity/project/run_id
        metric: train/critic_loss
        label: "..."
        smooth: 10                 # optional, default 1
        x: _step                   # optional, default _step
        color: sayno_no_relabeling # optional; see "Colours" below
        ls: "--"                   # optional line style, default solid
      - run:                       # a list stitches sequential (resumed) runs
          - entity/project/run_a    #   into ONE line, x offset continuing
          - entity/project/run_b
        metric: train/critic_loss
        label: "..."

Bar plot (one bar per run, height from run.summary[metric]):
    plot_type: bar
    output: src/results/<name>.pdf
    title: "..."
    xlabel: "..."
    ylabel: "..."
    log_y: false                   # optional
    legend_above: false            # optional (grouped_bar), legend in a row above the axes
    rotate_xticks: 0               # optional, e.g. 30 for tilted labels
    annotate: true                 # optional, label bars with their value
    annotate_fmt: "{:.2f}"         # optional
    entity: <wandb-entity>                         # default for all points
    project: q-ensemble-ablations                  # default for all points
    metric: val/q_value_difference_lang            # default for all points
    points:
      - run_id: <run_name>
        label: "$-1$"
        color: sayno_relabeling_rho5   # optional, all bars or none
        # optional overrides: project, metric

Colours (line `color`, bar `color`, grouped_bar `bar_colors`) go through
say_no.utils.palette.resolve: a palette key (the thesis entity), '#rrggbb', or
{shade_of: <key>, index: i, n: n} for a sweep inside one entity. Omitted = colour cycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from say_no.utils.wandb_plot import (
    WandbBarItem,
    WandbSeries,
    bar_plot,
    fetch_summary,
    grouped_bar_plot,
    line_plot,
)
from say_no.utils.figstyle import THESIS, save_figure, style
from say_no.utils.palette import resolve


def _color(spec: dict) -> str | None:
    return resolve(spec["color"]) if "color" in spec else None


LEGACY_STYLE_KEYS = {"bundle", "bundle_kwargs", "font_scale", "figsize"}


def _style_kwargs(cfg: dict, yaml_path: Path) -> dict:
    """`style: thesis` -> the thesis profile at `width` (fraction of \\textwidth) x `height` (in),
    stamped with the YAML's name; without `style`, the legacy bundle keys (pinned outputs)."""
    if "style" not in cfg:
        return {
            "save_to": cfg["output"],
            "bundle": cfg.get("bundle", "icml2024"),
            "bundle_kwargs": cfg.get("bundle_kwargs", {"usetex": False}),
            "font_scale": cfg.get("font_scale", 1.0),
            "figsize": tuple(cfg["figsize"]) if cfg.get("figsize") else None,
        }
    if cfg["style"] != "thesis":
        raise ValueError(f"{yaml_path}: unknown style {cfg['style']!r}; only 'thesis'")
    clash = LEGACY_STYLE_KEYS & set(cfg)
    if clash:
        raise ValueError(f"{yaml_path}: style: thesis fixes fonts and size; drop {sorted(clash)}")
    provenance = f"{yaml_path.name} · render_plot.py"
    return {"rc": style(THESIS, width=cfg["width"], height=cfg["height"]),
            "save": lambda fig: save_figure(fig, cfg["output"], provenance)}


def _common_kwargs(cfg: dict, *, default_xlabel: str, yaml_path: Path) -> dict:
    return {
        "title": cfg.get("title"),
        "xlabel": cfg.get("xlabel", default_xlabel),
        "ylabel": cfg.get("ylabel", "Value"),
        **_style_kwargs(cfg, yaml_path),
    }


def _build_line(cfg: dict, yaml_path: Path):
    series = []
    for s in cfg["series"]:
        def spec(run, label=s.get("label")):
            return WandbSeries(run=run, metric=s["metric"], label=label,
                               smooth=s.get("smooth", 1), x=s.get("x", "_step"),
                               color=_color(s), ls=s.get("ls", "-"))
        # a list of runs stitches them into one line; line_plot takes the nesting
        if isinstance(s["run"], list):
            series.append([spec(r, label=s.get("label") if i == 0 else None)
                           for i, r in enumerate(s["run"])])
        else:
            series.append(spec(s["run"]))
    kwargs = _common_kwargs(cfg, default_xlabel="Training step", yaml_path=yaml_path)
    kwargs["show_diff"] = cfg.get("show_diff", False)
    kwargs["sci_x"] = cfg.get("sci_x", True)
    kwargs["x_thousands"] = cfg.get("x_thousands", False)
    kwargs["log_y"] = cfg.get("log_y", False)
    kwargs["legend"] = cfg.get("legend", True)
    kwargs["y_decimals"] = cfg.get("y_decimals")
    if "ylim" in cfg:
        kwargs["ylim"] = tuple(cfg["ylim"])
    if "hline" in cfg:
        kwargs["hline"] = (cfg["hline"]["y"], _color(cfg["hline"]))
    return line_plot(series, **kwargs)


def _build_bar(cfg: dict, yaml_path: Path):
    entity = cfg["entity"]
    project = cfg["project"]
    metric = cfg["metric"]
    items = []
    for p in cfg["points"]:
        proj = p.get("project", project)
        items.append(
            WandbBarItem(
                run=f"{entity}/{proj}/{p['run_id']}",
                metric=p.get("metric", metric),
                label=p.get("label"),
                color=_color(p),
            )
        )
    kwargs = _common_kwargs(cfg, default_xlabel="", yaml_path=yaml_path)
    kwargs["log_y"] = cfg.get("log_y", False)
    kwargs["rotate_xticks"] = cfg.get("rotate_xticks", 0.0)
    kwargs["annotate"] = cfg.get("annotate", True)
    kwargs["annotate_fmt"] = cfg.get("annotate_fmt", "{:.2f}")
    return bar_plot(items, **kwargs)


def _build_grouped_bar(cfg: dict, yaml_path: Path):
    groups: list[list[WandbBarItem]] = []
    for g in cfg["groups"]:
        items = [WandbBarItem(run=b["run"], metric=b["metric"]) for b in g["bars"]]
        groups.append(items)
    kwargs = _common_kwargs(cfg, default_xlabel="", yaml_path=yaml_path)
    kwargs["log_y"] = cfg.get("log_y", False)
    kwargs["rotate_xticks"] = cfg.get("rotate_xticks", 0.0)
    kwargs["annotate"] = cfg.get("annotate", True)
    kwargs["annotate_fmt"] = cfg.get("annotate_fmt", "{:.2f}")
    kwargs["group_labels"] = [g["label"] for g in cfg["groups"]]
    kwargs["bar_labels"] = cfg["bar_labels"]
    kwargs["legend_above"] = cfg.get("legend_above", False)
    if "bar_colors" in cfg:
        kwargs["bar_colors"] = [resolve(c) for c in cfg["bar_colors"]]
    return grouped_bar_plot(groups, **kwargs)


def _build_table(cfg: dict, yaml_path: Path):
    """Fetch wandb summary scalars per (row, column) and emit LaTeX tabular rows.

    The output `.tex` file holds ONLY the body rows (one per record), with `&`
    column separators and trailing `\\\\`. Copy them into a `\\begin{tabular}`
    environment in the thesis (per the convention of building tables in-place).
    The rendered rows are also printed to stdout for quick inspection.
    """
    entity = cfg["entity"]
    project = cfg["project"]
    columns = cfg["columns"]  # list of {label, metric, fmt?}
    rows = cfg["rows"]        # list of {run_id, label, project?}
    output = Path(cfg["output"])

    rendered_rows: list[str] = []
    for row in rows:
        proj = row.get("project", project)
        run_path = f"{entity}/{proj}/{row['run_id']}"
        cells: list[str] = [row["label"]]
        for col in columns:
            spec = WandbBarItem(run=run_path, metric=col["metric"])
            val = fetch_summary(spec)
            cells.append(col.get("fmt", "{:.3f}").format(val))
        rendered_rows.append(" & ".join(cells) + r" \\")

    output.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(rendered_rows) + "\n"
    output.write_text(body, encoding="utf-8")
    print("--- LaTeX rows (paste into the thesis tabular) ---")
    print(body, end="")
    print("--- end ---")


DISPATCH = {
    "line": _build_line,
    "bar": _build_bar,
    "grouped_bar": _build_grouped_bar,
    "table": _build_table,
}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: render_plot.py <yaml_file>", file=sys.stderr)
        sys.exit(2)
    yaml_path = Path(sys.argv[1])
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    plot_type = cfg.get("plot_type")
    if plot_type not in DISPATCH:
        raise ValueError(
            f"Unknown plot_type {plot_type!r}; supported: {list(DISPATCH)}"
        )
    DISPATCH[plot_type](cfg, yaml_path)
    print(f"Saved: {cfg['output']}")


if __name__ == "__main__":
    main()
