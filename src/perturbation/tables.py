"""Booktabs tables (reader-facing names of the perturbation analysis: say_no.perturbation.names)."""
from __future__ import annotations

from pathlib import Path


def tex_escape(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("$", "\\$"), ("{", "\\{"), ("}", "\\}")):
        s = s.replace(a, b)
    return s


def write_booktabs(path: Path, header: list[str], rows: list[list[str]], **kw) -> None:
    path.write_text(booktabs(header, rows, **kw), encoding="utf-8")


def write_table(path: Path, tabulars: list[str], caption: str, label: str, resize: bool = False,
                size: str = "\\small", comment: str | None = None, note: str | None = None) -> None:
    """A `table` float holding one or more booktabs tabulars (stacked), caption and label.
    `comment`: leading `%` lines (provenance, protocol); `note`: a line set under the tabular."""
    body = [f"\\resizebox{{\\linewidth}}{{!}}{{%\n{t}}}\n" if resize else t for t in tabulars]
    head = "".join(f"% {line}".rstrip() + "\n" for line in comment.splitlines()) if comment else ""
    tail = f"\\par\\smallskip{{\\footnotesize {note}\\par}}\n" if note else ""
    text = (head + "\\begin{table}[t]\n\\centering\n" + (f"{size}\n" if size else "") +
            "\\par\\medskip\n".join(body) + tail +
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")
    path.write_text(text, encoding="utf-8")


def booktabs(header: list[str], rows: list[list[str]], *, align: str | None = None,
             group_header: list[tuple[str, int]] | None = None, midrule_after: list[int] | None = None) -> str:
    """A complete `tabular` with booktabs rules. `group_header` = [(text, span), ...] above the
    header, spans summing to the column count; `midrule_after` = row indices followed by a rule.
    A row given as a string is emitted verbatim (e.g. a `\\multicolumn` group label)."""
    ncol = len(header)
    if any(not isinstance(r, str) and len(r) != ncol for r in rows):
        raise ValueError("row length differs from header")
    align = align or ("l" + "c" * (ncol - 1))
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    if group_header:
        if sum(s for _, s in group_header) != ncol:
            raise ValueError("group header spans do not sum to the column count")
        cells, rules, col = [], [], 1
        for text, span in group_header:
            cells.append(text if span == 1 else f"\\multicolumn{{{span}}}{{c}}{{{text}}}")
            if text and span > 1:
                rules.append(f"\\cmidrule(lr){{{col}-{col + span - 1}}}")
            col += span
        lines.append(" & ".join(cells) + " \\\\")
        lines.extend(rules)
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for i, r in enumerate(rows):
        lines.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
        if midrule_after and i in midrule_after:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def stack(*lines: str, align: str = "c") -> str:
    """Multi-line cell (header or row label)."""
    return r"\begin{tabular}[b]{@{}" + align + r"@{}}" + r"\\".join(lines) + r"\end{tabular}"


def ranks(values: list[float | None], higher_better: bool = True, eligible: list[bool] | None = None,
          *, digits: int) -> list[str | None]:
    """'bold' / 'underline' / None per value: the marking rule of every thesis table.

    Within a comparison group, bold = best and underline = second best by dense ranking of the
    DISPLAYED values (rounded to `digits`; ties share a rank, so a tie for best is all bold and
    the next distinct value is underlined). Underline only when the group has at least three
    ranked entries. None values and entries with eligible[i] False are not ranked."""
    eligible = [True] * len(values) if eligible is None else eligible
    if len(eligible) != len(values):
        raise ValueError(f"{len(eligible)} eligibility flags for {len(values)} values")
    ranked = [v is not None and e for v, e in zip(values, eligible)]
    pool = [round(v, digits) for v, r in zip(values, ranked) if r]
    distinct = set(pool)
    out = []
    for v, r in zip(values, ranked):
        if not r:
            out.append(None)
            continue
        v = round(v, digits)
        rank = 1 + sum((w > v) if higher_better else (w < v) for w in distinct)
        out.append("bold" if rank == 1 else
                   "underline" if rank == 2 and len(pool) >= 3 else None)
    return out
