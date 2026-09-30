"""Thesis overview figure of SayNo: offline training of the critic ensemble, conformal calibration of
the alarm threshold on true-premise episodes, deployment next to a VLA policy, and the input
conditions of one frame the evaluation uses (feasible, wrong instruction by relabeling, missing object
by inpainting). Photos are real frames; instructions of relabeled inputs are the relabeling strings
the evaluation dump scored. The calibration panel is real data: the time-normalised score of one
calibration episode of one IVA task and the sorted episode maxima of all tasks' calibration halves,
with the pooled threshold, from the band that the IVA evaluation fits (analysis seed 0, fold 0).
Registered in src/scripts/render_thesis_figures.py.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from say_no.cp import _compat
from say_no.cp.episode_eval import MIN_CALIB_LEN, _frame_z, fold_bands, prepare
from say_no.perturbation.load import load_models, reviewed_rows
from say_no.utils.figstyle import FULL, THESIS, save_figure, style
from say_no.utils.palette import COLORS

C_SAYNO = COLORS["sayno_no_relabeling"]
C_OK = COLORS["expected_execute"]   # feasible input: border and tag
C_BAD = COLORS["perturbed"]         # every infeasible input, whatever was changed
C_DARK = COLORS["neutral_dark"]
C_LINE = COLORS["neutral_marker"]
C_EDGE = COLORS["separator"]
C_PANEL = "white"

HEIGHT_IN = 4.6
# two stroke tiers (boxes and local arrows / main data flow) and two corner radii (panels / boxes)
LW_BOX, LW_FLOW = 0.7, 1.2
R_PANEL, R_BOX = 0.08, 0.04
DASH_BAD = (0, (3, 1.5))  # infeasible inputs: dashed border, readable in black-and-white print


def tint(key_or_hex: str, w: float) -> str:
    """Palette colour mixed with a share `w` of white: pastel fills behind text."""
    h = COLORS[key_or_hex] if key_or_hex in COLORS else key_or_hex
    rgb = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * w):02X}" for c in rgb)


class Canvas:
    """One axes spanning the figure, data units = inches from the bottom-left corner."""

    def __init__(self, fig, w: float, h: float):
        self.ax = fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w), self.ax.set_ylim(0, h), self.ax.axis("off")
        self.big = plt.rcParams["axes.labelsize"]
        self.small = plt.rcParams["xtick.labelsize"]

    def box(self, x, y, w, h, fc, ec=None, lw=LW_BOX, r=R_BOX, z=1, ls="-"):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fc, ec=ec or fc, lw=lw, zorder=z, ls=ls))

    def text(self, x, y, s, big=False, bold=False, color=C_DARK, ha="center", va="center", z=5, **kw):
        return self.ax.text(x, y, s, fontsize=self.big if big else self.small,
                            fontweight="bold" if bold else "normal", color=color, ha=ha, va=va,
                            zorder=z, **kw)

    def arrow(self, p, q, color=C_LINE, lw=LW_BOX, rad=0.0, z=3, style="-|>", ms=6):
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                          connectionstyle=f"arc3,rad={rad}", zorder=z,
                                          shrinkA=0, shrinkB=0))

    def image(self, img, x, y, s, ec=C_EDGE, z=4, ls="-", lw=LW_BOX):
        img = np.asarray(img)
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) uint8 frame, got {img.dtype} {img.shape}")
        self.ax.imshow(img, extent=(x, x + s, y, y + s), zorder=z, interpolation="lanczos")
        self.ax.add_patch(plt.Rectangle((x, y), s, s, fill=False, ec=ec, lw=lw, ls=ls, zorder=z + 0.1))

    def step_badge(self, x, y, n, color=C_DARK):
        self.ax.add_patch(Circle((x, y), 0.085, fc=color, ec="none", zorder=6))
        self.text(x, y - 0.004, str(n), bold=True, color="white", z=7)

    def panel(self, x, y, w, h, n, title, fc=C_PANEL):
        self.box(x, y, w, h, fc=fc, ec=C_EDGE, r=R_PANEL, z=0.5)
        self.step_badge(x + 0.14, y + h - 0.14, n)
        self.text(x + 0.27, y + h - 0.14, title, big=True, bold=True, ha="left")

    def tag(self, x, y, s, fc, fg="white", ha="center", ec="none"):
        t = self.text(x, y, s, bold=True, color=fg, z=8, ha=ha)
        t.set_bbox(dict(boxstyle="round,pad=0.2,rounding_size=0.25", fc=fc, ec=ec, lw=LW_BOX))
        return t

    def feasible_tag(self, x, y, s, ha="center"):
        """Outlined tag for 'feasible / execute', set apart from the filled say-no signs."""
        return self.tag(x, y, s, fc="white", fg=C_DARK, ha=ha, ec=C_OK)

    def snowflake(self, x, y, r=0.045, color=C_SAYNO):
        """Frozen-weights mark: six spokes with short barbs."""
        for a in np.arange(6) * np.pi / 3:
            dx, dy = np.cos(a), np.sin(a)
            self.ax.plot([x, x + r * dx], [y, y + r * dy], color=color, lw=0.7, zorder=6,
                         solid_capstyle="round")
            for s in (-1, 1):  # barbs at 60 % of the spoke
                bx, by = x + 0.6 * r * dx, y + 0.6 * r * dy
                b = a + s * np.pi / 4
                self.ax.plot([bx, bx + 0.35 * r * np.cos(b)], [by, by + 0.35 * r * np.sin(b)],
                             color=color, lw=0.6, zorder=6, solid_capstyle="round")

    def photo(self, img, x, y, s, feasible: bool):
        """Input photo framed by its expected decision: solid green feasible, dashed magenta infeasible."""
        self.image(img, x, y, s, ec=C_OK if feasible else C_BAD, lw=LW_FLOW, ls="-" if feasible else DASH_BAD)

    def decision(self, x, y, execute: bool, label: str, ha="center"):
        """Decision as a bold word in the colour of the inputs it belongs to (green feasible, magenta
        infeasible); the word itself carries the meaning in black-and-white print."""
        self.text(x, y, label, bold=True, color=C_OK if execute else C_BAD, ha=ha)


def wrap(s: str, n: int) -> str:
    return "\n".join(textwrap.wrap(s.strip().rstrip("."), n))


def load_scene(pkl: Path, dump_path: Path, key: str) -> dict:
    """Original / object-removed initial frame, instruction and its random relabeling of one reviewed trajectory."""
    ds, tid = key.rsplit(":", 1)
    trajs, keep = reviewed_rows(pkl)
    kept = [t for t, k in zip(trajs, keep) if k]
    hits = [t for t in kept if (t["dataset"], t["trajectory_id"]) == (ds, int(tid))]
    if len(hits) != 1:
        raise ValueError(f"{key}: {len(hits)} reviewed rows in {pkl}")
    dump = load_models([("ref", dump_path)])["ref"]
    rows = np.flatnonzero((dump.dataset == ds) & (dump.trajectory_ids == int(tid)))
    if len(rows) != 1:
        raise ValueError(f"{key}: {len(rows)} rows in {dump_path}")
    t = hits[0]
    return dict(orig=t["first_image"], removed=t["first_image_inpainted"], language=t["language"],
                relabel=str(dump.swap_language["lang_swap_train"][rows[0]]), dataset=ds)


def draw_training(c: Canvas, x0: float, y0: float, w: float, y1: float, scene: dict) -> None:
    c.panel(x0, y0, w, y1 - y0, 1, "Offline training")
    c.text(x0 + 0.1, y1 - 0.3, "demonstrations only", ha="left", va="top", color=C_LINE)
    s = 0.7
    tx = x0 + 0.1 + s + 0.05
    yA = y1 - 0.5 - s
    yB = yA - (y1 - y0 - 0.5 - 2 * s) / 2 - s   # photo + reward line per demonstration
    rewards = {"feasible": "$r = -1$; goal: $r = 0$", "relabeled": "$r = -1$, no terminal"}
    for y, instr, lab in ((yA, scene["language"], "feasible"), (yB, scene["relabel"], "relabeled")):
        c.photo(scene["orig"], x0 + 0.1, y, s, lab == "feasible")
        t = c.text(tx, y + s - 0.2, "“" + wrap(instr, 13) + "”", ha="left", va="top", linespacing=1.05)
        if lab == "feasible":
            c.feasible_tag(tx + 0.03, y + s - 0.07, lab, ha="left")
        else:
            c.tag(tx + 0.03, y + s - 0.07, lab, fc=C_BAD, ha="left")
        # a long instruction can wrap below the photo; the reward line then goes below the instruction
        text_bottom = c.ax.transData.inverted().transform(
            t.get_window_extent(c.ax.figure.canvas.get_renderer()))[0, 1]
        c.text(x0 + 0.1, min(y, text_bottom) - 0.06, rewards[lab], ha="left", va="top")


def draw_critic(c: Canvas, x0: float, y0: float, w: float, y1: float) -> None:
    """Front card of width w - 0.1 with two cards behind it (the K members)."""
    fw = w - 0.1
    for k in (2, 1):
        c.box(x0 + 0.05 * k, y0 - 0.05 * k, fw, y1 - y0, fc=tint(C_SAYNO, 0.9),
              ec=C_SAYNO, r=R_PANEL, z=0.6 + 0.1 * (2 - k))
    c.box(x0, y0, fw, y1 - y0, fc=tint(C_SAYNO, 0.9), ec=C_SAYNO, r=R_PANEL, z=1)
    mid = x0 + fw / 2
    c.text(mid, y1 - 0.14, "SayNo critic ensemble", big=True, bold=True, color=C_SAYNO)
    # the stack of cards is the ensemble; K labels it from outside, next to the back card
    c.text(x0 + 0.03, y0 - 0.12, r"$\times K$, SARSA", color=C_SAYNO, ha="left", va="top")
    # one member as a block diagram: rows image / instruction / action
    rA, rB, rC = y1 - 0.42, y1 - 0.76, y1 - 1.06
    bh = 0.17
    x_in = x0 + 0.07                           # input labels
    x_enc, w_enc = x0 + 0.5, 0.64              # encoders
    x_head = x_enc + w_enc + 0.14              # image encoding and action feed the head (concatenated)
    w_head = fw - 0.07 - (x_head - x0)
    for y, lab in ((rA, "image $s$"), (rB, "instr. $l$"), (rC, "action $a$")):
        c.text(x_in, y, lab, ha="left")
    # ResNet-34 as an encoder trapezoid, narrowing towards its output
    xe0, xe1 = x_enc, x_enc + w_enc
    c.ax.add_patch(plt.Polygon([(xe0, rA - bh / 2 - 0.02), (xe1, rA - bh / 2 + 0.02),
                                (xe1, rA + bh / 2 - 0.02), (xe0, rA + bh / 2 + 0.02)],
                               closed=True, fc="white", ec=C_SAYNO, lw=LW_BOX, zorder=2))
    c.text(x_enc + w_enc / 2, rA, "ResNet-34", z=3)
    mh = 0.22
    c.box(x_enc, rB - mh / 2, w_enc, mh, fc="white", ec=C_SAYNO, z=2)
    c.text(x_enc + w_enc / 2 - 0.05, rB, "MUSE", z=3)
    c.snowflake(x_enc + w_enc - 0.09, rB)  # frozen: not trained
    for y in (rA, rB):
        c.arrow((x_in + 0.38, y), (x_enc - 0.01, y), color=C_SAYNO, ms=5)
    # FiLM: the instruction embedding modulates the image encoder
    c.arrow((x_enc + 0.12, rB + mh / 2), (x_enc + 0.12, rA - bh / 2 - 0.015), color=C_SAYNO, ms=5)
    c.text(x_enc + 0.16, (rA - bh / 2 + rB + mh / 2) / 2, "FiLM", color=C_SAYNO, ha="left")
    # image encoding and action enter the MLP head side by side (concatenated)
    c.arrow((xe1, rA), (x_head - 0.005, rA), color=C_SAYNO, ms=5)
    c.arrow((x_in + 0.42, rC), (x_head - 0.005, rC), color=C_SAYNO, ms=5)
    hy0, hy1 = rC - bh / 2, rA + bh / 2
    c.box(x_head, hy0, w_head, hy1 - hy0, fc="white", ec=C_SAYNO, z=2)
    c.text(x_head + w_head / 2, (hy0 + hy1) / 2, "MLP\nhead", linespacing=1.0, z=3)
    c.box(x0 + 0.07, y0 + 0.06, fw - 0.14, 0.18, fc=C_SAYNO, z=2)
    c.arrow((x_head + w_head / 2, hy0), (x_head + w_head / 2, y0 + 0.245), color=C_SAYNO, ms=5)
    c.text(mid, y0 + 0.15, r"$Q_{\theta_k}(s, a, l)$", color="white", bold=True, z=3)


def load_calibration(dump_path: Path, task: str, alpha: float, seed: int = 0, fold: int = 0) -> dict:
    """Real calibration data of the IVA episode-level detector (pooled quantile, max statistic):
    the z_t trajectories of `task`'s calibration half B, all tasks' pooled episode maxima, tau."""
    recs = {}
    for r in _compat.load(dump_path)["trajectories"]:
        recs.setdefault(r["task"], []).append(r)
    per_task = prepare(recs, "both")
    fitted = fold_bands(per_task, seed=seed, fold=fold, n_folds=5, alpha=alpha, statistic="max", k=0,
                        tau_alpha=None, align="truncate", min_calib_len=MIN_CALIB_LEN, pool_tasks=True)
    if fitted is None:
        raise ValueError(f"alpha={alpha} unattainable on seed {seed} fold {fold}")
    bands, _, keeps = fitted
    band = bands[task]
    idx_b = band["channels"][0]["idx_B"]
    curves = [_frame_z(per_task[task]["scores"][keeps[task][i]], band, 0) for i in idx_b]
    M = np.concatenate([b["calib_M"] for b in bands.values()])
    return dict(curves=curves, M=M, tau=float(band["q_M"]), task=task, alpha=alpha, seed=seed, fold=fold)


def draw_calibration(c: Canvas, x0: float, y0: float, w: float, y1: float, cal: dict) -> None:
    c.panel(x0, y0, w, y1 - y0, 2, "Calibration")
    M, tau = np.sort(cal["M"]), cal["tau"]
    # plots at most 0.7 in tall (taller only stretches empty z range), centred below the panel title
    ph = min(0.7, y1 - y0 - 0.5)
    py = y0 + 0.13 + (y1 - y0 - 0.5 - ph) / 2
    # both plots share one symlog z axis: the episode maxima are heavy-tailed (a few far above tau),
    # and a log axis shows all of them without clipping
    ylim = (min(-1.0, float(M.min())), float(M.max()) * 1.6)

    def z_axis(ax):
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(*ylim)
        ax.axhline(tau, color=C_DARK, lw=LW_BOX, ls="--", zorder=1)
        ax.set_xticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.patch.set_alpha(0)

    # one calibration episode's z_t (the one of the drawn task with the largest maximum), maximum marked
    pa = c.ax.inset_axes([x0 + 0.27, py, 0.5, ph], transform=c.ax.transData)
    z = max(cal["curves"], key=lambda a: float(a.max()))
    pa.plot(np.arange(z.size), z, color=C_SAYNO, lw=0.9, zorder=3)
    pa.plot(int(np.argmax(z)), z.max(), "o", mfc="white", mec=C_DARK, ms=3.2, mew=0.8, zorder=4)
    z_axis(pa)
    pa.set_yticks([0, tau])
    pa.set_yticklabels(["0", r"$\tau$"], fontsize=c.small)
    pa.tick_params(axis="y", length=2, pad=1)
    pa.set_xlabel("$t$", labelpad=0, fontsize=c.small)
    pa.set_ylabel("$z_t$", fontsize=c.small, rotation=0, ha="center", va="bottom")
    pa.yaxis.set_label_coords(0.0, 1.02)  # above the axis, apart from the tick labels
    c.arrow((x0 + 0.8, py + ph * 0.45), (x0 + 1.05, py + ph * 0.45))
    c.text(x0 + 0.925, py + ph * 0.45 + 0.06, r"$\max_t$", va="bottom")
    # all calibration episodes' maxima, sorted; the ones above tau (the alpha tail) filled, and the
    # episode drawn on the left ringed where its maximum lands
    hx = c.ax.inset_axes([x0 + 1.1, py, w - 1.2, ph], transform=c.ax.transData)
    i = np.arange(M.size)
    hx.plot(i[M <= tau], M[M <= tau], "o", mfc="white", mec=C_SAYNO, ms=1.8, mew=0.5, zorder=3)
    hx.plot(i[M > tau], M[M > tau], "o", mfc=C_SAYNO, mec=C_SAYNO, ms=2.2, zorder=3)
    j = int(np.argmin(np.abs(M - z.max())))
    if not np.isclose(M[j], z.max()):
        raise ValueError(f"drawn episode's maximum {z.max():.4f} is not among the calibration maxima")
    hx.plot(j, M[j], "o", mfc="white", mec=C_DARK, ms=3.2, mew=0.8, zorder=4)
    z_axis(hx)
    # the alpha tail: region above tau, where at most a share alpha of feasible episodes may lie
    hx.axhspan(tau, ylim[1], color=tint(C_SAYNO, 0.85), lw=0, zorder=0)
    hx.set_yticks([])
    hx.spines["left"].set_visible(False)
    hx.set_xlabel("episodes, sorted", labelpad=0, fontsize=c.small)
    hx.text(M.size * 0.35, np.sqrt(tau * ylim[1]), r"$\alpha$ tail", ha="center", va="center",
            fontsize=c.small, color=C_SAYNO)


def draw_deployment(c: Canvas, x0: float, y0: float, w: float, y1: float, scene: dict,
                    critic_x1: float, calib_x1: float, sy: float, qy: float) -> None:
    """`sy`: bottom of the score box, level with the critic card; `qy`: the decision, level with calibration."""
    c.panel(x0, y0, w, y1 - y0, 3, "Deployment")
    mid = x0 + w / 2
    s = 0.56
    iy = y1 - 0.31 - s
    c.image(scene["orig"], x0 + 0.1, iy, s)
    c.text(x0 + 0.1, iy - 0.04, "“" + wrap(scene["language"], 16) + "”", ha="left", va="top")
    vx, vw, vh = x0 + s + 0.3, w - s - 0.4, 0.3
    vy = iy + s / 2 - vh / 2
    c.box(vx, vy, vw, vh, fc="white", ec=C_DARK, z=2)
    c.text(vx + vw / 2, vy + vh / 2, "VLA policy", bold=True)
    c.arrow((x0 + 0.12 + s, vy + vh / 2), (vx - 0.01, vy + vh / 2))
    # SayNo scores image, instruction and the proposed action
    sx, sw, sh = x0 + 0.1, w - 0.2, 0.3
    c.arrow((vx + vw / 2, vy), (vx + vw / 2, sy + sh + 0.005))
    c.text(vx + vw / 2 + 0.05, (vy + sy + sh) / 2, "$a_t$", ha="left")
    c.box(sx, sy, sw, sh, fc=tint(C_SAYNO, 0.9), ec=C_SAYNO, z=2)
    c.text(mid, sy + sh / 2, r"time-normalized score $z_t$", bold=True, color=C_SAYNO)
    c.arrow((critic_x1, sy + sh / 2), (sx - 0.005, sy + sh / 2), color=C_SAYNO, lw=LW_FLOW, ms=8)
    # decision
    c.arrow((mid, sy), (mid, qy + 0.1))
    q = c.text(mid, qy, r"$z_t > \tau$ ?", bold=True)
    q.set_bbox(dict(boxstyle="round,pad=0.3,rounding_size=0.3", fc="white", ec=C_DARK, lw=LW_BOX))
    c.arrow((calib_x1, qy), (mid - 0.3, qy), color=C_DARK, lw=LW_FLOW, ms=8)
    c.text(x0 + 0.2, qy + 0.04, r"$\tau$", big=True, va="bottom")
    ty = qy - 0.75                      # decision words close under the gate
    xl, xr = x0 + 0.45, x0 + w - 0.45   # centres of the two decision words
    c.arrow((mid - 0.14, qy - 0.1), (xl, ty + 0.09))
    c.arrow((mid + 0.14, qy - 0.1), (xr, ty + 0.09))
    c.text((mid + xl) / 2 - 0.1, (qy + ty) / 2 + 0.03, "no", ha="right")
    c.text((mid + xr) / 2 + 0.1, (qy + ty) / 2 + 0.03, "yes", ha="left")
    c.decision(xl, ty, True, "execute $a_t$")
    c.decision(xr, ty, False, "say no")


def draw_evaluation(c: Canvas, y0: float, y1: float, W: float, scene: dict, cols: tuple) -> None:
    """Example strip, not a pipeline step: lighter frame, cards on the top row's columns `cols`;
    the border (solid green / dashed magenta) carries the expected decision."""
    c.box(0.04, y0, W - 0.08, y1 - y0, fc="white", ec=C_EDGE, lw=0.5, r=R_PANEL, z=0.5)
    c.text(0.14, y1 - 0.15, "Evaluation inputs", big=True, bold=True, ha="left")
    # one frame, three inputs: as recorded, with a wrong instruction, with the object removed
    cards = [  # (header, image, instruction, feasible?)
        ("Feasible", scene["orig"], scene["language"], True),
        ("Wrong instruction", scene["orig"], scene["relabel"], False),
        ("Missing object", scene["removed"], scene["language"], False),
    ]
    s = y1 - y0 - 0.62
    for x, (header, img, instr, feasible) in zip(cols, cards):
        c.text(x, y1 - 0.38, header, ha="left", bold=True)
        iy = y0 + 0.1
        c.photo(img, x, iy, s, feasible)
        c.text(x + s + 0.07, iy + s, "“" + wrap(instr, 17) + "”", ha="left", va="top", linespacing=1.1)


def draw(c: Canvas, scene: dict, train_scene: dict, cal: dict, W: float, H: float) -> None:
    eval_y0, eval_y1 = 0.04, 1.2
    top_y0, top_y1 = eval_y1 + 0.1, H - 0.04
    ax0, aw = 0.04, 1.6
    cx0, cw = ax0 + aw + 0.18, 1.8
    dx0 = cx0 + cw + 0.18
    critic_y0 = top_y1 - 1.54                  # the critic card keeps its size; calibration takes the rest
    calib_y1 = critic_y0 - 0.28
    front_x1 = cx0 + cw - 0.1                  # right edge of the front card
    draw_training(c, ax0, top_y0, aw, top_y1, train_scene)
    draw_critic(c, cx0, critic_y0, cw, top_y1)
    draw_calibration(c, cx0, top_y0, cw, calib_y1, cal)
    draw_deployment(c, dx0, top_y0, W - 0.04 - dx0, top_y1, scene, front_x1, cx0 + cw,
                    sy=critic_y0 - 0.02, qy=calib_y1 - 0.14)
    ya = critic_y0 + 0.78                      # data -> critics, level with the critic's inputs
    c.arrow((ax0 + aw, ya), (cx0 - 0.005, ya), color=C_DARK, lw=LW_FLOW, ms=8)
    c.arrow((cx0 + (cw - 0.1) / 2, critic_y0 - 0.11), (cx0 + (cw - 0.1) / 2, calib_y1 + 0.005),
            color=C_DARK, lw=LW_FLOW, ms=8)                                                        # critics -> calibration
    draw_evaluation(c, eval_y0, eval_y1, W, scene, cols=(ax0 + 0.1, cx0 + 0.1, dx0 + 0.1))


def check_layout(fig) -> None:
    """Raise if two texts overlap, a text overlaps a photo, or a text leaves the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in [*ax.texts, ax.xaxis.label, ax.yaxis.label] if t.get_text()]
    boxes = [(t.get_text(), (t.get_bbox_patch() or t).get_window_extent(r)) for t in texts]
    images = [im.get_window_extent(r) for ax in fig.axes for im in ax.images]
    canvas = fig.bbox
    bad = []
    for i, (s, b) in enumerate(boxes):
        if b.x0 < canvas.x0 or b.x1 > canvas.x1 or b.y0 < canvas.y0 or b.y1 > canvas.y1:
            bad.append(f"outside canvas: {s!r}")
        bad += [f"text on photo: {s!r}" for im in images if b.overlaps(im)]
        bad += [f"overlap: {s!r} / {s2!r}" for s2, b2 in boxes[i + 1:] if b.overlaps(b2)]
    if bad:
        raise ValueError("layout defects:\n  " + "\n  ".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inpainting-pkl", required=True, type=Path, help="reviewed object-removal pairs")
    ap.add_argument("--dump", required=True, type=Path, help="evaluation dump with the relabeling strings")
    ap.add_argument("--scene", required=True, help="dataset:trajectory_id for deployment and evaluation inputs")
    ap.add_argument("--train-scene", required=True, help="dataset:trajectory_id for the training panel")
    ap.add_argument("--calib-dump", required=True, type=Path,
                    help="IVA three-way-prompt dump the episode-level evaluation fits its bands on")
    ap.add_argument("--calib-task", required=True, help="IVA task whose calibration episodes are drawn")
    ap.add_argument("--alpha", required=True, type=float, help="episode false-alarm level of the threshold")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--preview", type=Path, help="PNG preview path")
    args = ap.parse_args()

    scene = load_scene(args.inpainting_pkl, args.dump, args.scene)
    train_scene = load_scene(args.inpainting_pkl, args.dump, args.train_scene)
    cal = load_calibration(args.calib_dump, args.calib_task, args.alpha)

    rc = style(THESIS, width=FULL, height=HEIGHT_IN)
    with plt.rc_context({**rc, "figure.constrained_layout.use": False, "image.composite_image": False}):
        fig = plt.figure()
        W, H = fig.get_size_inches()
        c = Canvas(fig, W, H)
        draw(c, scene, train_scene, cal, W, H)
        check_layout(fig)
        prov = (f"{args.scene} {args.train_scene} | "
                f"calib {args.calib_dump.stem.split('_')[0]} {cal['task']} a={cal['alpha']} s{cal['seed']}f{cal['fold']} "
                f"tau={cal['tau']:.2f} n={cal['M'].size} · plot_sayno_overview.py")
        save_figure(fig, args.out, prov, dpi=300)
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            fig.canvas.print_figure(args.preview, dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
