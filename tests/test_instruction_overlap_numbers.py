"""Seen vs unseen instructions on Bridge/Fractal, recomputed from the instruction walk and the dumps.

Pins the vocabulary overlap and the stratified AUROCs of instruction-overlap-bridge-fractal/, and checks
the JSON of `src/scripts/instruction_overlap.py` agrees. Skips when h5py or an input is absent.
"""
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from say_no.cp.metrics import auroc_neg_pos  # noqa: E402
from say_no.perturbation.instructions import load_walk, overlap_counts, seen_mask  # noqa: E402
from say_no.perturbation.load import load_models  # noqa: E402
from say_no.perturbation.scores import plain_readouts  # noqa: E402

CAMPAIGN = "instruction-overlap-bridge-fractal"
CONTROLS = "perturbation-sensitivity-bridge-fractal/controls"
A = "SayNo"


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"{p} absent")
    return p


@pytest.fixture(scope="module")
def walk(results_dir):
    return load_walk(_need(results_dir / CAMPAIGN / "instructions_bridge_fractal_full.hdf5"))


@pytest.fixture(scope="module")
def report(results_dir):
    return json.loads(_need(results_dir / CAMPAIGN / "instruction_overlap.json").read_text())


def test_vocabulary_overlap(walk, report):
    c = overlap_counts(walk, "exact")
    assert c == report["overlap"]["exact"]
    b, f = c["bridge"], c["fractal"]
    assert (b["train_trajectories"], b["val_trajectories"], f["train_trajectories"], f["val_trajectories"]) == (38660, 5147, 82843, 4361)
    assert (b["train_unique"], b["val_unique"], b["unique_in_both"], b["unique_val_only"]) == (19973, 2919, 694, 2225)
    assert b["val_trajectories_with_seen_instruction"] == 2912
    assert (f["train_unique"], f["val_unique"], f["unique_val_only"]) == (598, 477, 0)


@pytest.mark.parametrize("set_name,file,cond,seen_rows,unseen_rows,auroc_seen,auroc_unseen", [
    ("inpainting258", "nonegdemo_n300_inpainted", "inpaint", 65, 50, 0.715, 0.642),
    ("val200", "nonegdemo_val_n200", "lang_swap_train", 60, 40, 0.881, 0.589),
])
def test_bridge_spread_auroc_by_instruction(walk, report, results_dir, set_name, file, cond, seen_rows, unseen_rows,
                                            auroc_seen, auroc_unseen):
    m = load_models([(A, _need(results_dir / CONTROLS / f"{file}.hdf5"))])[A]
    seen = seen_mask(m.language, m.dataset, walk, "exact")
    bridge = m.dataset == "bridge"
    assert ((bridge & seen).sum(), (bridge & ~seen).sum(), (~bridge & ~seen).sum()) == (seen_rows, unseen_rows, 0)
    orig, pert = plain_readouts(m.q["orig"], "both")["spread"], plain_readouts(m.q[cond], "both")["spread"]
    for mask, want, group in ((bridge & seen, auroc_seen, "bridge_seen"), (bridge & ~seen, auroc_unseen, "bridge_unseen")):
        got = auroc_neg_pos(orig[mask], pert[mask])
        assert got == pytest.approx(want, abs=5e-4)
        assert report["sets"][set_name]["models"][A]["exact"][group]["auroc"][cond]["both"]["spread"]["auroc"] == pytest.approx(got)


def test_pkls_come_from_their_split(report):
    chk = report["pkl_split_check"]
    assert chk["bridge_fractal_n300_with_inpainted_cleaned.pkl"]["split"] == "val"
    assert chk["bridge_fractal_train_n200.pkl"]["split"] == "train"
    assert np.all([v[d]["trajectories"] in (100, 150) for v in chk.values() for d in ("bridge", "fractal")])


def test_domain_rows_recomputed(walk, report, results_dir):
    """Thesis table tab:bf-seen-unseen-task: every cell recomputed from the dumps (Bridge scope)."""
    from say_no.perturbation.instructions import seen_mask as seen_of
    dom = report["domains"]
    for lab, prefix in (("SayNo", "nonegdemo"), (r"SayNo-CR", "negdemo0.10")):
        tr = load_models([(lab, _need(results_dir / CAMPAIGN / "dumps" / f"{prefix}_train_n200_inpainted.hdf5"))])[lab]
        ho = load_models([(lab, _need(results_dir / CONTROLS / f"{prefix}_n300_inpainted.hdf5"))])[lab]
        seen = seen_of(ho.language, ho.dataset, walk, "exact")
        groups = {"train": (tr, tr.dataset == "bridge"), "eval_seen": (ho, (ho.dataset == "bridge") & seen),
                  "eval_unseen": (ho, (ho.dataset == "bridge") & ~seen)}
        for g, (m, mask) in groups.items():
            assert dom[lab]["bridge"][g]["n"] == mask.sum()
            for cond in ("lang_swap_train", "inpaint"):
                for rd in ("spread", "neg_mean_q"):
                    got = auroc_neg_pos(plain_readouts(m.q["orig"], "both")[rd][mask], plain_readouts(m.q[cond], "both")[rd][mask])
                    assert dom[lab]["bridge"][g]["cells"][cond][rd]["auroc"] == pytest.approx(got)
