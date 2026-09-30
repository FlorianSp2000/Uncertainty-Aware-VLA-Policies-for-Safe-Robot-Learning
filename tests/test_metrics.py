import numpy as np
import pytest

from say_no.cp.metrics import auroc_neg_pos, paired_auroc_ci


def test_auroc_neg_pos_perfect_chance_and_ties():
    assert auroc_neg_pos([0, 1, 2], [3, 4, 5]) == 1.0
    assert auroc_neg_pos([3, 4, 5], [0, 1, 2]) == 0.0
    assert auroc_neg_pos([1, 1], [1, 1]) == 0.5


def test_paired_ci_brackets_point_estimate_and_narrows_with_n():
    rng = np.random.default_rng(0)
    neg = rng.normal(size=400)
    pos = neg + 0.5 + rng.normal(size=400)
    point = auroc_neg_pos(neg, pos)
    lo, hi = paired_auroc_ci(neg, pos, n_boot=500)
    assert lo < point < hi
    lo_s, hi_s = paired_auroc_ci(neg[:50], pos[:50], n_boot=500)
    assert hi_s - lo_s > hi - lo


def test_paired_ci_rejects_unpaired_input():
    with pytest.raises(ValueError):
        paired_auroc_ci(np.zeros(3), np.zeros(4))
