"""Pins the delivery rules of the IVA thesis tables (say_no.cp.thesis_tables): which cells are
marked best / second best, and that interval distances reproduce the rounded Wilson bounds.
The numbers themselves are pinned in test_episode_statistics_numbers.py."""

import re

import pytest

from say_no.cp.thesis_tables import TABLES, cell, write_all
from say_no.perturbation.tables import ranks


def test_ranks_ties_eligibility_and_group_size():
    assert ranks([0.9, 0.8, 0.7], True, digits=3) == ["bold", "underline", None]
    assert ranks([0.9, 0.9, 0.7], True, digits=3) == ["bold", "bold", "underline"]  # tie for best, next distinct is second
    assert ranks([0.9, 0.8, 0.8, 0.7], True, digits=3) == ["bold", "underline", "underline", None]  # tie for second
    assert ranks([0.9, 0.8], True, digits=3) == ["bold", None]  # two entries: no underline
    assert ranks([3.0, 1.0, 2.0], False, digits=3) == [None, "bold", "underline"]
    assert ranks([0.9, 0.8, 0.7, 0.6], True, [False, True, True, True], digits=3) == [None, "bold", "underline", None]


def test_interval_distances_reproduce_the_rounded_bounds():
    assert cell(0.0461, ci=(0.0249, 0.0821)).startswith("$0.046^{+.036}_{-.021}$")
    with pytest.raises(AssertionError):
        cell(0.5, ci=(0.6, 0.7))


@pytest.fixture(scope="module")
def tables(load_results, tmp_path_factory):
    out = tmp_path_factory.mktemp("iva_tables")
    write_all(load_results("episode-operating-point-iva/thesis/episode_statistics.json"), out)
    return {p.stem: p.read_text(encoding="utf-8") for p in out.glob("*.tex")}


def _row(tex, label, nth=0):
    """Cells of the nth row starting with `label` (Maximum appears once per score group)."""
    hits = [l for l in tex.splitlines() if l.startswith(label + " &")]
    assert len(hits) > nth, label
    return [c.strip() for c in hits[nth].rstrip(" \\").split("&")]


def test_every_table_is_written_once_with_its_label(tables):
    assert len(tables) == len(TABLES) == 5
    for name, tex in tables.items():
        assert r"\label{tab:" + name.replace("_", "-") + "}" in tex
        assert r"\resizebox" not in tex and r"\footnotesize" not in tex.split(r"\par\smallskip")[0]


def test_episode_statistic_marks(tables):
    tex = tables["iva_episode_statistic_comparison"]
    mx, frac = _row(tex, "Maximum"), _row(tex, "Fraction")  # first Maximum = both scores
    assert r"\mathbf{0.750}" in mx[2] and r"\mathbf{0.901}" in mx[4]
    assert r"\mathbf" not in "".join(_row(tex, "Maximum", 1))  # value-score group: unranked
    # fraction exceeds alpha in FAR, so its high cross-task detection rate is not ranked ...
    assert r"\mathbf" not in frac[3]
    # ... while its threshold-free AUROC is
    assert r"\mathbf{0.958}" in frac[5]
    assert r"\underline{0.728}" in _row(tex, "Top-5 mean")[2]


def test_episode_detection_marks_compare_the_two_scores_at_the_headline_alpha(tables):
    tex = tables["iva_episode_detection"]
    rows = [l for l in tex.splitlines() if l.startswith("Episode-level & $0.05$ &")]
    both, value = rows
    assert r"\mathbf{0.767}" in value and r"\mathbf" not in both
    frame = _row(tex, "Frame-level")
    assert frame[2] == "$0.202$" and frame[3] == "--" and r"\mathbf" not in "".join(frame)
    # single-calibration FAR range = min/max over the 15 cross-fitted calibrations
    assert _row(tex, "Episode-level", 1)[3] == "$0.000$--$0.156$"


def test_relabeling_named_as_in_bridge_fractal_and_no_flagged_run(tables):
    for tex in tables.values():
        assert "Cross-Task" not in tex and "cross-task" not in tex
    rd = tables["iva_frame_auroc_reward_design"]
    assert r"\dagger" not in rd and len([l for l in rd.splitlines() if l.startswith("$")]) == 3


def test_no_result_number_in_any_caption(tables):
    """Captions carry setup (alpha, task and episode counts), never a rate or an AUROC."""
    for name, tex in tables.items():
        caption = re.search(r"\\caption\{(.*)\}\n\\label", tex, re.S).group(1)
        assert not re.search(r"0\.\d{3}", caption), name
