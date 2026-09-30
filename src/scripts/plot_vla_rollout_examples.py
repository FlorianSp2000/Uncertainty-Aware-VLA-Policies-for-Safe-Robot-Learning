"""Qualitative rollouts: frames plus the per-step score against the conformal band.

For one task of one dump, picks test episodes by a fixed rule (no hand-picking) and draws, per
episode, four frames and the raw score s_t with the calibration mean mu_t and the band
mu_t + h * sigma_t of SAFE's functional CP (vla_rollouts.safe_band, calibration split 0) -- the band
the thesis tables alarm on. The alarm is the first step with s_t >= mu_t + h * sigma_t.

Selection rule, per task (``pick``):
* typical success     -- test success with the median episode score (max_t z_t)
* false-alarm success -- test success with the highest score, if it alarms
* typical detection   -- detected failure with the median first-alarm step
* missed failure      -- undetected failure with the median score, if any

Two steps, because the frames live in the cache on the cluster:
``--list_keys`` prints the chosen episode keys; fetch ``<cache>/test/<key>.npz`` into
``--frames_dir``; then run again without it to draw.
"""
from __future__ import annotations

import argparse
import io
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from say_no import vla_rollouts as vr  # noqa: E402
from say_no.utils.figstyle import THESIS, save_figure, style  # noqa: E402
from say_no.utils.palette import COLORS  # noqa: E402

# per-step scores: name + time-indexed symbol, as in the thesis, Method chapter
SYMBOL = {"value_score": r"Value score $-\bar Q_t$", "ensemble_disagreement": r"Ensemble disagreement $\sigma_{Q,t}$"}


def upper(band: dict, T: int) -> np.ndarray:
    if len(band["mu"]) < T:
        raise ValueError(f"{T} steps, band only fitted on {len(band['mu'])}")
    return band["mu"][:T] + band["h"] * band["sigma"][:T]


def first_alarm(s: np.ndarray, band: dict) -> int | None:
    hit = np.flatnonzero(s >= upper(band, len(s)))
    return int(hit[0]) if hit.size else None


def episode_score(s: np.ndarray, band: dict) -> float:
    """max_t (s_t - mu_t) / sigma_t: alarms iff it reaches h."""
    T = len(s)
    return float(np.max((s - band["mu"][:T]) / band["sigma"][:T]))


def pick(test_succ: list, test_fail: list, task: str, channel: str, band: dict) -> list:
    score = lambda r: episode_score(vr.CHANNELS[channel](r), band)  # noqa: E731
    first = lambda r: first_alarm(vr.CHANNELS[channel](r), band)  # noqa: E731
    succ = sorted([r for r in test_succ if r["condition"] == task], key=score)
    fail = [r for r in test_fail if r["condition"] == task]
    if not succ or not fail:
        raise ValueError(f"task {task}: {len(succ)} test successes, {len(fail)} failures")
    out = [("typical success", succ[len(succ) // 2])]
    if first(succ[-1]) is not None:
        out.append(("false alarm", succ[-1]))
    det = sorted([r for r in fail if first(r) is not None], key=first)
    if det:
        out.append(("detected failure", det[len(det) // 2]))
    miss = sorted([r for r in fail if first(r) is None], key=score)
    if miss:
        out.append(("missed failure", miss[len(miss) // 2]))
    return out


def single_rollout(args, dump, band, r) -> None:
    """One rollout in detail: chosen frames above, the score against the band below, with an
    annotated event step (``--event_step``, e.g. the frame an object falls) and the first alarm."""
    s = vr.CHANNELS[args.channel](r)
    T = len(s)
    mu, up = band["mu"][:T], upper(band, T)
    first = first_alarm(s, band)
    frames = np.load(Path(args.frames_dir) / f"{r['key']}.npz", allow_pickle=True)["images"]
    steps = [int(x) for x in args.steps.split(",")]
    if max(steps) >= T:
        raise ValueError(f"steps {steps} beyond episode length {T}")
    n = len(steps)
    with plt.rc_context(style(THESIS, width=1.0, height=3.8)):
        fig = plt.figure(layout="constrained")
        gs = fig.add_gridspec(2, n, height_ratios=[1, 1.35])
        for j, t in enumerate(steps):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(Image.open(io.BytesIO(bytes(frames[t]))))
            ax.set_xticks([]); ax.set_yticks([])
            # wrapped: a frame title must not run into its neighbours
            tag = ("alarm" if t == first else textwrap.fill(args.event_label, 16) if t == args.event_step else "")
            ax.set_title(f"$t = {t}$\n{tag}", color=COLORS["perturbed"] if t == first else "k")
        ax = fig.add_subplot(gs[1, :])
        t = np.arange(T)
        ax.plot(t, mu, color=COLORS["neutral_dark"], ls="--", lw=0.9, label=r"Calibration mean $\mu_t$")
        ax.fill_between(t, np.full_like(up, min(s.min(), mu.min()) - (up.max() - min(s.min(), mu.min()))), up, color=COLORS["original"], alpha=0.5, lw=0,
                        label=rf"Band $\mu_t + \tau\varsigma_t$ ($\alpha = {args.alpha:g}$)")
        ax.plot(t, s, color=COLORS[args.channel], lw=1.3, marker="o", ms=2,
                label=SYMBOL[args.channel])
        if args.event_step is not None:
            ax.axvline(args.event_step, color=COLORS["neutral_marker"], ls=":", lw=1.0,
                       label=args.event_label)
        if first is not None:
            ax.axvline(first, color=COLORS["perturbed"], lw=1.0, label="First alarm")
        lo = min(s.min(), mu.min()); hi = max(s.max(), up.max())
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_xlabel("Step $t$")
        ax.set_ylabel(SYMBOL[args.channel])
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
        save_figure(fig, args.out, f"{Path(args.dump).name}, step {dump['step']}, {r['key']}, "
                    f"channel {args.channel} | plot_vla_rollout_examples.py", preview=True)
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--channel", default="value_score", choices=list(SYMBOL))
    ap.add_argument("--alpha", type=float, required=True, help="the thesis tables' alpha (SAFE protocol)")
    ap.add_argument("--frames_dir", required=True, help="holds <key>.npz fetched from the cache")
    ap.add_argument("--list_keys", action="store_true")
    ap.add_argument("--key", default=None, help="single-rollout mode: this test rollout only")
    ap.add_argument("--steps", default=None, help="single-rollout mode: comma-separated frames to show")
    ap.add_argument("--event_step", type=int, default=None, help="single-rollout mode: step to mark")
    ap.add_argument("--event_label", default="event", help="label of the marked step")
    ap.add_argument("--out", help="PDF path")
    ap.add_argument("--view", default=None,
                    help="calib_pool:test_pool for SAFE-protocol dumps, e.g. eval_seen:unseen")
    args = ap.parse_args()

    dump = vr.load_dump(args.dump, tuple(args.view.split(":")) if args.view else None)
    calib, test_succ, test_fail = vr.populations(dump["records"])
    band = vr.safe_band(calib, vr.CHANNELS[args.channel], args.alpha, seed=0)
    if args.key:
        match = [r for r in test_succ + test_fail if r["key"] == args.key]
        if len(match) != 1 or not args.steps:
            raise ValueError(f"--key {args.key!r}: {len(match)} test rollouts; --steps required")
        single_rollout(args, dump, band, match[0])
        return
    chosen = pick(test_succ, test_fail, args.task, args.channel, band)
    if args.list_keys:
        for _, r in chosen:
            print(r["key"])
        return

    n = len(chosen)
    with plt.rc_context(style(THESIS, width=1.0, height=1.35 * n + 0.5)):
        fig, axes = plt.subplots(n, 5, squeeze=False, layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 2.6]))
        for i, (label, r) in enumerate(chosen):
            s = vr.CHANNELS[args.channel](r)
            T = len(s)
            mu, up = band["mu"][:T], upper(band, T)
            first = first_alarm(s, band)
            frames = np.load(Path(args.frames_dir) / f"{r['key']}.npz", allow_pickle=True)["images"]
            # Start, the alarm (or two evenly spaced frames), end -- four distinct steps.
            show = [0, T // 3, 2 * T // 3, T - 1]
            if first is not None and first not in show:
                # replace the fixed frame nearest to the alarm, keeping start and end
                j = 1 if abs(first - T // 3) <= abs(first - 2 * T // 3) else 2
                show[j] = first
            show = sorted(show)
            for j, t in enumerate(show):
                ax = axes[i, j]
                ax.imshow(Image.open(io.BytesIO(bytes(frames[t]))))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"$t = {t}$" + (" (alarm)" if first is not None and t == first else ""),
                             color=COLORS["perturbed"] if first is not None and t == first else "k")
            axes[i, 0].set_ylabel(label[0].upper() + label[1:])
            ax = axes[i, 4]
            t = np.arange(T)
            ax.plot(t, mu, color=COLORS["neutral_dark"], ls="--", lw=0.9,
                    label="Calibration mean $\\mu_t$")
            ax.fill_between(t, np.full_like(up, min(s.min(), mu.min()) - (up.max() - min(s.min(), mu.min()))), up, color=COLORS["original"], alpha=0.5, lw=0,
                            label=rf"Band $\mu_t + \tau\varsigma_t$ ($\alpha = {args.alpha:g}$)")
            ax.plot(t, s, color=COLORS[args.channel], lw=1.3, label=SYMBOL[args.channel])
            if first is not None:
                ax.axvline(first, color=COLORS["perturbed"], lw=1.0, label="Alarm")
            lo = min(s.min(), mu.min()); hi = max(s.max(), up.max())
            ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            ax.set_ylabel(SYMBOL[args.channel])
            ax.set_title(f"{'Success' if r['success'] else 'Failure'}, "
                         f"{'alarm at step %d' % first if first is not None else 'no alarm'}",
                         loc="left")
        axes[-1, 4].set_xlabel("Step $t$")
        hh, ll = [], []
        for ax in axes[:, 4]:
            for a, b in zip(*ax.get_legend_handles_labels()):
                if b not in ll:
                    hh.append(a); ll.append(b)
        fig.legend(hh, ll, loc="outside lower center", ncol=4, frameon=False)
        save_figure(fig, args.out, f"{Path(args.dump).name}, step {dump['step']}, task {args.task}, "
                    f"channel {args.channel} | src/scripts/plot_vla_rollout_examples.py", preview=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
