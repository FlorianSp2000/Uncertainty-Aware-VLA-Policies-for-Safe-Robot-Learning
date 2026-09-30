"""One episode, end to end: frames, the score, the threshold, and the two label rows.

Host-side renderer consuming eval_iva_cp.py pkl records (no JAX). One episode block,
top to bottom:

    filmstrip        one frame per cell of the timestep axis, taken at the cell centre
    score trace      sigma_Q with the conformal threshold, crossings shaded
    ground truth     which frames actually carry a false premise
    prediction       which frames the detector alarms on

The two label rows are the point of the figure: a reader compares them directly and
sees both what was caught and what was invented. Same layout as the interactive
explorer, so the static figure and the browsable one cannot drift apart. A key under
the whole figure names all four marks, so nothing on it depends on the caption.

`render_episode_pair` stacks two such blocks in one file -- a worked "it works" and
"it does not" side by side is more convincing than either alone.

Width, fonts and legend size come from the caller's say_no.utils.figstyle profile (thesis or
paper); the layout constants below are in inches and scale with neither.
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from say_no.cp.bands import _extend, exceeds_zband
from say_no.cp.images import decode_frames
from say_no.utils.figstyle import Profile, legend_kwargs, save_pinned, style
from say_no.utils.palette import COLORS

C_SCORE = COLORS["ensemble_disagreement"]    # sigma_Q
C_THRESH = COLORS["threshold"]               # conformal threshold
C_TRUE = COLORS["false_premise_injection"]   # ground-truth false premise
C_ALARM = "black"        # alarms are events, not a palette entity: neutral, borrows no hue
FILMSTRIP_FRAMES = 4
AXES_FRAC = 0.87         # seed only: _fit_filmstrip measures the real axes width at draw time
H_TRACE, H_RUG = 1.05, 0.18
TITLE_PT = 8.0           # two-line headline of a titled block; only the report pair draws one
H_LEGEND = 0.42          # reserved inside the canvas for the 2x2 key
PAD_IN = 0.03            # gutter, so the axes does not sit flush against the column edge


def _tex_safe(s: str, usetex: bool) -> str:
    return s.replace("_", r"\_") if usetex else s


def _strip_height(record, fig_w):
    """The filmstrip, and a first guess at the row height in inches that keeps it square.

    The strip is drawn as FILMSTRIP_FRAMES equal cells spanning the timestep axis, so the
    frames are sampled at those cells' CENTRES. A fixed stride from zero instead puts the
    last frame a whole cell short of the end -- on a 111-step episode, stride 28 stops at
    84 and the episode never gets to show how it finished.

    The height is only a guess: height_ratios are dimensionless, so inches handed to them
    are honoured only if the non-axes space happens to equal the slack allowed for it.
    `_fit_filmstrip` measures the drawn box and corrects. AXES_FRAC just seeds it.
    """
    stored = decode_frames(record)
    n = FILMSTRIP_FRAMES
    idx = (((np.arange(n) + 0.5) * stored.shape[0]) // n).astype(int)
    imgs = stored[idx]
    strip = np.concatenate(list(imgs), axis=1)
    h, w = imgs.shape[1], imgs.shape[2]
    return strip, (fig_w * AXES_FRAC) * h / (w * n)


def _fit_filmstrip(fig, ax, ratios, strip, tol=0.01, iters=6):
    """Grow the figure until the filmstrip row is exactly the strip's own aspect.

    With imshow(aspect="auto") the image fills whatever box it is given, so a row that is
    half an inch too short silently stretches every frame. Constrained layout only resolves
    the box at draw time, hence measure, correct the row ratio and the canvas together,
    repeat.
    """
    want_aspect = strip.shape[0] / strip.shape[1]
    gs = ax.get_subplotspec().get_gridspec()
    for _ in range(iters):
        fig.canvas.draw()
        box = ax.get_window_extent(fig.canvas.get_renderer())
        delta = (box.width * want_aspect - box.height) / fig.dpi
        if abs(delta) < tol:
            return
        ratios[0] += delta
        gs.set_height_ratios(ratios)
        w, h = fig.get_size_inches()
        fig.set_size_inches(w, h + delta)
    raise RuntimeError(
        f"filmstrip aspect did not converge within {iters} iterations (off by {delta:.3f} in)")


def _runs(mask):
    """Contiguous [start, stop) index runs where mask is True.

    The injections are scattered single frames; merging neighbours keeps the shaded bands
    from being drawn as a stack of coincident rectangles.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    stops = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return list(zip(starts, stops))


def _legend_handles(zband, truth_label="injected false premise"):
    """Every mark on the figure, named once. The two rug rows carry only colour.

    Labels are kept short on purpose: the key is laid out 2x2 under a \\columnwidth figure,
    so its width is two labels plus their handles, and text that overflows the figure is
    what makes the panel look cropped at the column edge.

    The maths renders the paper's own symbols under either text engine -- mathtext when
    usetex is off, real TeX when it is on.
    """
    handles = [
        Line2D([], [], color=C_SCORE, lw=1.2),
        Line2D([], [], color=C_THRESH, lw=1.0, ls="--"),
        Patch(facecolor=C_TRUE),
        # the alarm mark appears twice -- as a dot on the trace and as a bar in the rug --
        # so one entry shows both, or a reader is left guessing what the dots are
        (Patch(facecolor=C_ALARM),
         Line2D([], [], color=C_ALARM, marker="o", ls="none", ms=2.8)),
    ]
    labels = [r"$\sigma_Q$: disagreement", rf"threshold, $\alpha={zband['alpha']:g}$",
              "injected false premise", "detector alarm"]
    return handles, labels


def _add_legend(fig, zband, usetex, profile: Profile, truth_label="injected false premise"):
    """One key under the whole figure, so nothing on it is left unnamed.

    ``truth_label`` names the ground-truth rug. It is a parameter because what makes the
    premise false differs by dataset -- an injected instruction swap on RLBench, an object
    absent from the scene on the real robot -- and a key that says "injected" over an
    untouched recording would be wrong.
    """
    handles, labels = _legend_handles(zband, truth_label)
    # "outside" makes constrained layout reserve the key's height inside the canvas, so the
    # saved size is the size asked for. Anchored outside the figure rectangle it would not
    # be, and the export would depend on bbox_inches growing to catch it.
    fig.legend(handles=handles, labels=labels, loc="outside lower center",
               ncol=2, frameon=False, handlelength=1.4,
               columnspacing=1.4, handletextpad=0.5,
               handler_map={tuple: HandlerTuple(ndivide=None)}, **legend_kwargs(profile))


def _draw_episode(axs, record, zband, strip, usetex, title=None, band=None, show_title=True):
    """Draw one episode into four stacked axes; returns the headline counts.

    show_title=False leaves the filmstrip untitled: paper figures carry their message in
    the caption, not in the artifact.
    """
    s = np.asarray(record["s_t"], dtype=np.float64)
    feasible = np.asarray(record["task_feasible"]).astype(bool)
    T = s.size
    assert feasible.size == T

    alarms = exceeds_zband(s, zband)
    ceiling = _extend(zband["mu"] + zband["q_z"] * zband["varsigma"], T)
    t = np.arange(T)
    prompt = Counter(record["language"]).most_common(1)[0][0]

    n_inj, n_hit = int((~feasible).sum()), int(alarms[~feasible].sum())
    n_false, n_feas = int(alarms[feasible].sum()), int(feasible.sum())

    axs[0].imshow(strip, extent=(-0.5, T - 0.5, 0, 1), aspect="auto")
    axs[0].set_yticks([])
    # the frames butt up against each other and several share a background, so without a
    # rule between them the strip reads as one wide photograph
    for x in np.linspace(-0.5, T - 0.5, FILMSTRIP_FRAMES + 1)[1:-1]:
        axs[0].axvline(x, color="black", lw=0.8)
    if show_title:
        head = title or f"{_tex_safe(record['task'], usetex)}, episode {record['episode_id']}"
        axs[0].set_title(
            f"{head} --- ``{_tex_safe(prompt, usetex)}''\n"
            f"{n_hit} of {n_inj} false-premise frames alarmed on; "
            f"{n_false} of {n_feas} feasible frames alarmed on in error", fontsize=TITLE_PT)

    ax = axs[1]
    top = max(s.max(), ceiling.max())
    ax.plot(t, ceiling, color=C_THRESH, lw=1.0, ls="--")
    if band is not None:
        upper = _extend(band["upper"], T)
        ax.plot(t, upper, color=C_ALARM, lw=0.9, ls=":")
        top = max(top, upper.max())
    ax.fill_between(t, ceiling, s, where=s > ceiling, color=C_ALARM, alpha=.35, lw=0)
    ax.plot(t, s, color=C_SCORE, lw=1.2)
    over = s > ceiling
    if over.any():
        ax.plot(t[over], s[over], "o", color=C_ALARM, ms=2.8, zorder=4)
    ax.set_ylim(0.0, 1.08 * top)
    ax.set_ylabel(r"$\sigma_Q$")

    # the rows carry no y label: each is one colour, and the key already names both. At
    # column width their labels were the widest thing in the left margin, so dropping them
    # goes straight into the plot area.
    for axis, mask, colour in ((axs[2], ~feasible, C_TRUE), (axs[3], alarms, C_ALARM)):
        axis.set_yticks([])
        axis.set_ylim(0, 1)
        axis.set_facecolor(COLORS["rug_background"])
        for a, b in _runs(mask):
            axis.axvspan(a - 0.5, b - 0.5, color=colour, lw=0)
    axs[3].set_xlabel("timestep")

    # frame t is drawn as a sample at x=t, so its bar has to straddle t rather than start
    # there, or the rug sits half a frame to the right of the peak it explains
    for axis in axs:
        axis.set_xlim(-0.5, T - 0.5)
    for axis in axs[:3]:
        axis.tick_params(labelbottom=False)
    return n_hit, n_inj, n_false, n_feas


def _check_frames(record):
    assert record["images"] is not None, (
        f"{record['task']}#{record['episode_id']} has no stored frames "
        "-- rerun inference with --keep_images")


def render_episode_figure(record, zband, save, *, profile: Profile, width: float, title=None,
                          usetex=None, band=None, show_title=True,
                          truth_label="injected false premise"):
    """One episode, handed to `save(fig)` (figstyle.save_figure or save_pinned) inside its style.

    zband: zscore_cp_threshold(...) fitted on this task's feasible episodes with THIS
           episode held out, so nothing drawn is in-sample.
    profile, width: the figstyle target and the include width (fraction of its \\columnwidth).
    Returns the headline counts (hits, injected, false alarms, feasible) for the caption.

    The drawn width is exactly the include width: the key is laid out inside the canvas and
    `save` must not use a tight bbox, so the page gets the point sizes drawn here.
    """
    _check_frames(record)
    fig_w = width * profile.column_in
    strip, h_strip = _strip_height(record, fig_w)
    ratios = [h_strip, H_TRACE, H_RUG, H_RUG]
    fig_h = sum(ratios) + (1.0 if show_title else 0.4) + H_LEGEND
    usetex = profile.usetex if usetex is None else usetex
    with plt.rc_context(style(profile, width=width, height=fig_h, usetex=usetex)):
        fig, axs = plt.subplots(4, 1, height_ratios=ratios)
        fig.get_layout_engine().set(w_pad=PAD_IN, h_pad=PAD_IN, hspace=0.03)
        counts = _draw_episode(axs, record, zband, strip, usetex, title=title, band=band,
                               show_title=show_title)
        _add_legend(fig, zband, usetex, profile, truth_label)
        _fit_filmstrip(fig, axs[0], ratios, strip)
        save(fig)
        plt.close(fig)
    return counts


def render_episode_pair(items, out_path, *, profile: Profile, width: float, usetex=None):
    """Two episodes stacked in one file: items = [(record, zband, title), ...].

    A browsing artifact, not a paper panel: it is not placed at a fixed width, so it keeps
    the tight bbox and does not run the filmstrip aspect fit (which would have to resolve
    two nested gridspecs at once). Its frames can be slightly stretched.
    """
    assert len(items) == 2, "render_episode_pair takes exactly two episodes"
    for record, _, _ in items:
        _check_frames(record)
    fig_w = width * profile.column_in
    strips = [_strip_height(r, fig_w) for r, _, _ in items]

    ratios, fig_h = [], 1.4
    for _, h_strip in strips:
        ratios += [h_strip, H_TRACE, H_RUG, H_RUG]
        fig_h += h_strip + H_TRACE + 2 * H_RUG + 0.75

    usetex = profile.usetex if usetex is None else usetex
    with plt.rc_context(style(profile, width=width, height=fig_h, usetex=usetex)):
        fig = plt.figure()
        # one gridspec per block so the gap between blocks is wider than the gap
        # between a block's own rows
        outer = fig.add_gridspec(2, 1, hspace=0.28)
        for block, ((record, zband, title), (strip, h_strip)) in enumerate(zip(items, strips)):
            inner = outer[block].subgridspec(4, 1, hspace=0.10,
                                             height_ratios=[h_strip, H_TRACE, H_RUG, H_RUG])
            axs = [fig.add_subplot(inner[i]) for i in range(4)]
            _draw_episode(axs, record, zband, strip, usetex, title=title)
        _add_legend(fig, items[0][1], usetex, profile)
        save_pinned(fig, out_path, bbox_inches="tight", pad_inches=PAD_IN)
        plt.close(fig)


def pick_examples(records, flags_by_ep):
    """One episode where detection works, one where it does not.

    Both are chosen among episodes that have stored frames and enough injected
    frames for the rates to mean anything, by per-episode (recall - false-alarm
    rate). Deterministic: no sampling, ties broken by episode id.
    """
    usable = [r for r in records
              if r["images"] is not None and int((~np.asarray(r["task_feasible"])).sum()) >= 5]
    assert usable, "no episode with stored frames and >=5 injected frames"

    def separation(r):
        feas = np.asarray(r["task_feasible"]).astype(bool)
        f = flags_by_ep[(r["task"], int(r["episode_id"]))]
        return float(f[~feas].mean()) - float(f[feas].mean()), -int(r["episode_id"])

    return max(usable, key=separation), min(usable, key=separation)
