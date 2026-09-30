"""Qualitative contact sheet for the embedding baselines (sanity check, not a paper figure)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from say_no.utils.figstyle import save_pinned, stamp


def _wrap(s: str, width: int) -> str:
    return textwrap.fill(s, width, break_long_words=False)


def contact_sheet(examples: list[dict], out: Path, title: str, provenance: str, per_row: int = 3) -> None:
    """examples: [{"images": [(img, subtitle), ...], "caption": str}], all with the same image count.
    Laid out `per_row` examples per row, caption under each example."""
    n_img = {len(e["images"]) for e in examples}
    if len(n_img) != 1:
        raise ValueError(f"examples carry different image counts {n_img}")
    n_img = n_img.pop()
    rows = -(-len(examples) // per_row)
    fig, axes = plt.subplots(rows, per_row * n_img, figsize=(1.25 * per_row * n_img, 1.95 * rows + 0.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, e in enumerate(examples):
        r, c0 = divmod(i, per_row)
        for j, (img, sub) in enumerate(e["images"]):
            ax = axes[r][c0 * n_img + j]
            ax.imshow(img)
            ax.set_title(sub, fontsize=5.5, pad=2)
        axes[r][c0 * n_img].text(0.0, -0.04, _wrap(e["caption"], 24 * n_img), transform=axes[r][c0 * n_img].transAxes,
                                 fontsize=5.0, va="top", ha="left")
    fig.suptitle(_wrap(title, int(14 * per_row * n_img)), fontsize=7)
    fig.subplots_adjust(hspace=0.9, wspace=0.08, top=0.9, bottom=0.16, left=0.01, right=0.99)
    stamp(fig, provenance)
    save_pinned(fig, out, dpi=200)
    plt.close(fig)
