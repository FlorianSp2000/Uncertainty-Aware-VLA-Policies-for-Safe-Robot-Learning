import numpy as np
import pytest

from say_no.baselines.embedding import (FeatureNovelty, alignment_score, auroc_rows, frame_novelty,
                                        ledoit_wolf, paired_auroc_diff_ci, spearman)
from say_no.cp.metrics import auroc_neg_pos

RNG = np.random.default_rng(0)
D = 32
# embedding-like data: a common mean direction (cosine kNN is blind to shifts along the origin)
MU = np.r_[np.full(D // 2, 1.0), np.zeros(D - D // 2)]
SHIFT = np.r_[np.zeros(D // 2), np.full(D - D // 2, 1.0)]


def _gauss(n, shift=0.0, members=()):
    return MU + shift * SHIFT + 0.5 * RNG.normal(size=(n, *members, D))


def _ref(n=200):
    return _gauss(n), np.repeat(["bridge", "fractal"], n // 2)


@pytest.mark.parametrize("method", FeatureNovelty.METHODS)
def test_shifted_gaussian_is_detected(method):
    ref, g = _ref()
    model = FeatureNovelty(method, k=5).fit(ref, g)
    inlier = _gauss(300)
    shifted = _gauss(300, shift=1.0)
    assert auroc_neg_pos(model.score(inlier), model.score(shifted)) > 0.9


@pytest.mark.parametrize("method", FeatureNovelty.METHODS)
def test_identical_inputs_give_chance(method):
    ref, g = _ref()
    x = _gauss(100)
    s = FeatureNovelty(method).fit(ref, g).score(x)
    assert auroc_neg_pos(s, s.copy()) == 0.5


def test_min_over_groups_uses_nearest_group():
    ref = np.r_[RNG.normal(size=(100, D)), RNG.normal(size=(100, D)) + 10.0]
    g = np.repeat(["a", "b"], 100)
    model = FeatureNovelty("mahalanobis").fit(ref, g)
    far_from_both = RNG.normal(size=(50, D)) + 5.0
    near_b = RNG.normal(size=(50, D)) + 10.0
    assert auroc_neg_pos(model.score(near_b), model.score(far_from_both)) > 0.9


def test_ledoit_wolf_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    x = RNG.normal(size=(60, 40))
    cov, shrink = ledoit_wolf(x)
    ref = sk.LedoitWolf().fit(x)
    np.testing.assert_allclose(cov, ref.covariance_, atol=1e-10)
    assert shrink == pytest.approx(ref.shrinkage_)


def test_alignment_matched_beats_shuffled():
    txt = RNG.normal(size=(200, D))
    img = txt + 0.5 * RNG.normal(size=(200, D))
    matched = alignment_score(img, txt)
    shuffled = alignment_score(img, np.roll(txt, 1, axis=0))
    assert auroc_neg_pos(matched, shuffled) > 0.9
    with pytest.raises(ValueError):
        alignment_score(img, txt[:10])


def test_frame_novelty_member_axis_and_rows():
    ref = _gauss(100, members=(3, 2))
    g = np.repeat(["bridge", "fractal"], 50)
    feats = {"orig": _gauss(80, members=(3, 2)), "shift": _gauss(80, shift=1.0, members=(3, 2))}
    s = frame_novelty(lambda: FeatureNovelty("knn"), ref, g, feats)
    assert s["orig"].shape == (80, 2)
    rows = auroc_rows("knn", s["orig"], s["shift"], n_boot=200)
    assert rows["auroc knn"] > 0.9 and rows["auroc knn (first frame)"] > 0.9
    lo, hi = rows["auroc knn ci95"]
    assert lo <= rows["auroc knn"] <= hi
    with pytest.raises(ValueError):
        frame_novelty(lambda: FeatureNovelty("knn"), ref, g, {"bad": RNG.normal(size=(80, 2, 2, D))})


def test_diff_ci_and_spearman():
    neg = RNG.normal(size=300)
    good, weak = neg + 2 + RNG.normal(size=300), neg + 0.2 + RNG.normal(size=300)
    d, lo, hi = paired_auroc_diff_ci(neg, good, neg, weak, n_boot=300)
    assert lo > 0 and lo <= d <= hi
    x = RNG.normal(size=100)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        FeatureNovelty("cosine")
    with pytest.raises(RuntimeError):
        FeatureNovelty("knn").score(np.zeros((2, D)))
