"""The score-violin figures (say_no.perturbation.violins) show the distributions behind the thesis AUROCs.

Inputs are the registry invocations of bf_detector_score_violins.pdf / bf_seen_unseen_task_violins.pdf
(src/scripts/render_thesis_figures.py), parsed by the driver's own parser. Pins:
- detectors: every drawn cell's mean percentile among the originals equals the AUROC of the same cell in
  perturbation_thesis.json (tab:bf-detector-comparison), and every table cell is drawn;
- seen_unseen: per group, the percentile among the group's OWN originals averages to the AUROC of
  instruction_overlap.json (tab:bf-seen-unseen-task); the figure's shared reference is the pooled test originals.
Skips when h5py, a dump or a JSON is absent.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from say_no.cp.metrics import auroc_neg_pos  # noqa: E402
from say_no.perturbation import violins as V  # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _module("render_thesis_figures", "src/scripts/render_thesis_figures.py")
D = _module("plot_perturbation_score_violins", "src/scripts/plot_perturbation_score_violins.py")


def _args(name):
    argv = R.REGISTRY[name].argv
    for a in argv:
        p = a.rsplit("=", 1)[-1]
        if p.endswith((".hdf5", ".json")) and not Path(p).exists():
            pytest.skip(f"{p} absent")
    return D.build_parser().parse_args(list(argv))


def _json(path):
    p = Path(path)
    if not p.exists():
        pytest.skip(f"{p} absent")
    return json.loads(p.read_text(encoding="utf-8"))


def test_percentile_mean_is_the_auroc():
    rng = np.random.default_rng(0)
    o = rng.normal(size=50)
    p = np.r_[rng.normal(1, size=40), o[:5]]  # ties with originals count half
    assert V.percentile_among(o, p).mean() == pytest.approx(auroc_neg_pos(o, p), abs=1e-12)


@pytest.mark.slow
def test_detector_violins_are_the_table_cells():
    args = _args("bf_detector_score_violins.pdf")
    res = _json(f"{R.AT}/perturbation_thesis.json")["sets"]
    sets, rows = D.detector_data(args)
    u = V.detector_percentiles(sets, rows)
    assert len(rows) == 10 and len(u) == 28
    for (lab, rd, c), x in u.items():
        s = dict(V.DETECTOR_CONDITIONS)[c]
        assert x.size == {"val200": 200, "inpainting258": 258}[s]
        assert x.mean() == pytest.approx(res[s][lab]["auroc"][c][V.FRAME][rd]["auroc"], abs=1e-9), (lab, rd, c)


def test_seen_unseen_groups_are_the_table_cells():
    args = _args("bf_seen_unseen_task_violins.pdf")
    dom = _json(f"{R.IO}/instruction_overlap.json")["domains"]
    for lab, groups in D.seen_unseen_groups(args).items():
        assert {g: len(m.trajectory_ids) for g, m in groups.items()} == {"train": 75, "eval_seen": 65, "eval_unseen": 50}
        for g, m in groups.items():
            for c in V.SEEN_UNSEEN_CONDITIONS:
                for rd in V.SCORE_SYMBOLS:
                    o, p = V.pair(m, c, rd)
                    want = dom[lab]["bridge"][g]["cells"][c][rd]["auroc"]  # stored to 4 decimals
                    assert V.percentile_among(o, p).mean() == pytest.approx(want, abs=5e-5), (lab, g, c, rd)
        u = V.seen_unseen_percentiles(groups, "spread")
        ref = np.concatenate([u[(g, "orig", "inpaint")] for g in V.REFERENCE_GROUPS])
        assert ref.mean() == pytest.approx(0.5, abs=1e-12)  # the reference is uniform over itself
