"""Bridge/Fractal thesis numbers, recomputed from the perturbation-sensitivity dumps.

Each case pins a number that enters the thesis to three
decimals, recomputed here through say_no.perturbation from the local HDF5 dumps, and checks the
analysis JSON of `analyse_perturbation_thesis.py` agrees. Skips when h5py or a dump is absent.
"""
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from say_no.cp.metrics import auroc_neg_pos  # noqa: E402
from say_no.perturbation.evaluate import operating_point  # noqa: E402
from say_no.perturbation.load import legacy_originals, load_models, reviewed_frames, reviewed_rows  # noqa: E402
from say_no.perturbation.scores import crossfit_folds, member_q, plain_readouts, spread, z_readouts  # noqa: E402

CAMPAIGN = "perturbation-sensitivity-bridge-fractal"
PKL = Path("src/data/bridge_fractal_n300_with_inpainted_cleaned.pkl")
SEED, FOLDS = 0, 2  # as in docs/REPRODUCE.md, step 2b


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"{p} absent")
    return p


# --------------------------------------------------------------------------- thesis tables

ENC_RHO_INPAINT = 0.279  # Spearman(shift of own-encoder Mahalanobis, shift of spread), inpaint, no neg. demos
NO_RELABEL = "SayNo"
PER_SAMPLE = "ResNet--MUSE 3 (per sample)"
JSON_PINS = [(NO_RELABEL, "lang_swap_near", "spread", 0.619),
             (PER_SAMPLE, "lang_swap_near", "p_infeasible", 0.731),
             ("CLIP ViT-B/32 | clip_alignment", "lang_swap_near", "clip_alignment", 0.500),
             ("Penalty $-100$", "lang_swap_train", "spread", 0.752)]
CTL = Path("src/results") / CAMPAIGN / "controls"
CLIP = Path("src/results/embedding-baselines-bridge-fractal")
THESIS_JSON = f"{CAMPAIGN}/analysis_thesis/perturbation_thesis.json"


def ctl(name: str):
    return load_models([(name, _need(CTL / f"{name}.hdf5"))])[name]


def both(m, cond, rd="spread"):
    if m.kind == "detector":
        return auroc_neg_pos(m.score["orig"].mean(1), m.score[cond].mean(1))
    return auroc_neg_pos(plain_readouts(m.q["orig"], "both")[rd], plain_readouts(m.q[cond], "both")[rd])


@pytest.mark.parametrize("name,cond,rd,expected", [
    ("nonegdemo_n300_inpainted", "inpaint", "spread", 0.702),
    ("nonegdemo_n300_inpainted", "composite_local_r16", "spread", 0.702),
    ("nonegdemo_n300_inpainted", "composite_global_r16", "spread", 0.500),
    ("nonegdemo_n300_inpainted", "jpeg_orig", "spread", 0.502),
    ("nonegdemo_n300_inpainted", "inpaint", "neg_mean_q", 0.536),
    ("negdemo0.10_n300_inpainted", "inpaint", "spread", 0.683),
    ("cls_resnet_muse_3_n300_inpainted", "inpaint", None, 0.618),
    ("cls_resnet_muse_3_n300_inpainted", "composite_local_r16", None, 0.619),
    ("cls_octo_n300_inpainted", "inpaint", None, 0.504),
    ("nonegdemo_val_n200", "lang_swap_train", "spread", 0.831),
    ("nonegdemo_val_n200", "lang_swap_near", "spread", 0.619),
    ("negdemo0.10_val_n200", "lang_swap_train", "neg_mean_q", 0.932),
    ("negdemo0.10_val_n200", "lang_swap_near", "spread", 0.647),
    ("cls_resnet_muse_3_val_n200", "lang_swap_near", None, 0.731),
    ("cls_octo_val_n200", "lang_swap_train", None, 0.774),
    ("abl_penalty100_val_n200", "lang_swap_train", "spread", 0.752),
])
def test_controls_recomputed(name, cond, rd, expected):
    assert both(ctl(name), cond, rd) == pytest.approx(expected, abs=5e-4)


def test_ssot_images_match_cluster_dump():
    """The host rebuild of the image conditions (cluster code, image_perturbations.py) reproduces the
    per-frame median |delta| and mask areas the cluster logged (first 12 rows; the driver checks all)."""
    pytest.importorskip("scipy")
    from say_no.perturbation.images import rebuild_like_dump
    m = ctl("nonegdemo_n300_inpainted")
    trajs = reviewed_frames(_need(PKL), m.trajectory_ids, m.dataset)[:12]
    images, info = rebuild_like_dump(trajs, CTL / "nonegdemo_n300_inpainted.hdf5", n_rows=12)
    assert info["jpeg_quality"] == 60 and info["mask_threshold"] == 30
    assert set(images) == {"inpaint", "jpeg_orig"} | {f"composite_{k}_r{r}" for k in ("local", "global") for r in (8, 16, 32)}


def test_clip_scores_rows_and_controls():
    from say_no.perturbation.load import load_detector_scores
    d = load_detector_scores(_need(CLIP / "clip_scores_n300_inpainted.hdf5"),
                             {"clip_mahalanobis": "maha", "clip_alignment": "align"})
    m = ctl("nonegdemo_n300_inpainted")
    assert np.array_equal(d["maha"].trajectory_ids, m.trajectory_ids)
    for cond, expected in (("inpaint", 0.975), ("composite_local_r16", 0.551), ("composite_global_r16", 0.934),
                           ("jpeg_orig", 0.990)):
        assert both(d["maha"], cond) == pytest.approx(expected, abs=5e-4)
    v = load_detector_scores(_need(CLIP / "clip_scores_val_n200.hdf5"), {"clip_alignment": "align"})["align"]
    assert both(v, "lang_swap_near") == pytest.approx(0.500, abs=5e-4)


def test_encoder_novelty_recomputed():
    from say_no.perturbation.evaluate import shift_spearman
    from say_no.perturbation.novelty import encoder_novelty
    m = ctl("nonegdemo_n300_inpainted")
    nov = encoder_novelty(m, _need(CTL / "nonegdemo_train_n200_feat.hdf5"), k=5)
    maha, knn = nov[f"{m.label} | enc_mahalanobis"], nov[f"{m.label} | enc_knn"]
    assert both(maha, "inpaint") == pytest.approx(0.577, abs=5e-4)
    assert both(knn, "inpaint") == pytest.approx(0.603, abs=5e-4)
    assert shift_spearman(maha, m, "inpaint", 2000, 0)["rho"] == pytest.approx(ENC_RHO_INPAINT, abs=5e-4)


def test_training_split_calibration_from_canonical_dump():
    """train_n200 canonical dump gives the same operating point as the legacy calibration dump."""
    from say_no.perturbation.report import calibration_scores
    cal = calibration_scores(ctl("nonegdemo_train_n200_feat"))["spread"]
    m = ctl("nonegdemo_n300_inpainted")
    r = operating_point(cal, plain_readouts(m.q["orig"], "both")["spread"], plain_readouts(m.q["inpaint"], "both")["spread"], 0.10)
    assert (r["tpr"], r["fpr"]) == pytest.approx((0.558, 0.252), abs=5e-4)


def test_thesis_json_agrees_with_recomputation(load_results):
    res = load_results(THESIS_JSON)
    inp, val = res["sets"]["inpainting258"], res["sets"]["val200"]
    ens = inp[NO_RELABEL]["auroc"]
    assert ens["inpaint"]["both"]["spread"]["auroc"] == pytest.approx(0.702, abs=5e-4)
    assert ens["composite_global_r16"]["both"]["spread"]["auroc"] == pytest.approx(0.500, abs=5e-4)
    assert ens["jpeg_orig"]["both"]["spread"]["auroc"] == pytest.approx(0.502, abs=5e-4)
    assert inp["CLIP ViT-B/32 | clip_mahalanobis"]["auroc"]["jpeg_orig"]["both"]["clip_mahalanobis"]["auroc"] == pytest.approx(0.990, abs=5e-4)
    assert inp[PER_SAMPLE]["auroc"]["inpaint"]["both"]["p_infeasible"]["auroc"] == pytest.approx(0.618, abs=5e-4)
    assert inp[f"{NO_RELABEL} | enc_mahalanobis"]["auroc"]["inpaint"]["both"]["enc_mahalanobis"]["auroc"] == pytest.approx(0.577, abs=5e-4)
    op = inp[NO_RELABEL]["operating_point"]["readouts"]["spread"]["split_half_test_originals"]["0.1"]
    assert (op["tpr_mean"], op["fpr_mean"]) == pytest.approx((0.253, 0.101), abs=5e-4)
    assert res["shift_spearman"]["inpainting258"][NO_RELABEL]["inpaint"]["enc_mahalanobis"]["rho"] == pytest.approx(ENC_RHO_INPAINT, abs=5e-4)
    for lab, cond, rd, expected in JSON_PINS:
        assert val[lab]["auroc"][cond]["both"][rd]["auroc"] == pytest.approx(expected, abs=5e-4), (lab, cond, rd)
    lat = res["latency"]["critic"]
    assert [round(c["median_ms"], 1) for c in lat if c["members"] == 8] == [7.7, 12.1, 38.2, 120.8]


def test_tables_reproduce_from_json(load_results, tmp_path, results_dir):
    """Every number in the thesis tables: the committed .tex files equal a regeneration from the
    analysis JSON (whose numbers the tests above recompute from the dumps)."""
    from say_no.perturbation import thesis_tables as T
    res = load_results(THESIS_JSON)
    out_dir = results_dir / CAMPAIGN / "analysis_thesis"
    T.write_all(res["sets"], res["shift_spearman"], res["controls_caption"], res["roles"],
                Path(res["args"]["latency"]), tmp_path)
    written = sorted(p.name for p in tmp_path.glob("*.tex"))
    assert written == sorted(T.FILES.values())
    for name in written:
        assert (tmp_path / name).read_text(encoding="utf-8") == _need(out_dir / name).read_text(encoding="utf-8"), name


def test_table_cells_and_marks():
    """+- is the half-width of the bootstrap interval; marks rank the displayed values (dense ranking: ties share a rank)."""
    from say_no.perturbation.tables import ranks
    from say_no.perturbation.thesis_tables import pm
    assert pm({"auroc": 0.8304, "ci95": [0.79, 0.87]}, "bold") == r"$\mathbf{0.83}{\scriptstyle\,\pm\,0.04}$"
    assert ranks([0.701, 0.699, 0.5, None], digits=2) == ["bold", "bold", "underline", None]  # tie for best: both bold, next distinct second
    assert ranks([0.9, 0.7, 0.5], digits=2) == ["bold", "underline", None]
    assert ranks([0.9, 0.7], digits=2) == ["bold", None]
