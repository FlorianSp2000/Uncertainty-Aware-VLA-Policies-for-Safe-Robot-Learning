"""The `step_reward` knob must leave the reference reward design bit-identical.

The step-reward ablation turns one number in the IVA reward from -1.0 to 0.0. Everything
downstream (an ensemble that costs GPU-hours, a per-task floor comparison) is only
interpretable if the control arm is the design that produced the existing IVA
results. These cases pin both ends of that: the default reproduces the reference,
and 0.0 removes the step-reward horizon without touching the false-premise branch.

Imported by source text rather than `from octo.data.iva import ...` -- the module
pulls in tensorflow/dlimp, which the host analysis venv does not carry.
"""

from pathlib import Path

import numpy as np
import pytest

IVA_PY = Path(__file__).resolve().parents[1] / "external" / "V-GPS" / "octo" / "octo" / "data" / "iva.py"


@pytest.fixture(scope="module")
def iva_funcs():
    src = IVA_PY.read_text(encoding="utf-8")

    def grab(name):
        i = src.index(f"def {name}(")
        return src[i:src.index("\ndef ", i + 1)]

    ns = {"np": np, "List": list, "Optional": object}
    exec(grab("create_rewards_with_penalties"), ns)
    exec(grab("_compute_mc_return"), ns)
    return ns["create_rewards_with_penalties"], ns["_compute_mc_return"]


def test_default_reproduces_the_reference_design(iva_funcs):
    """-1 on every step, 0 on the last three: the step-reward horizon the existing IVA runs used."""
    make, _ = iva_funcs
    L = 10
    assert np.array_equal(make(L, []), np.r_[np.full(L - 3, -1.0), np.zeros(3)])


def test_step_reward_zero_removes_the_horizon_countdown(iva_funcs):
    """A feasible episode carries no reward at all, so its return is flat 0."""
    make, mc = iva_funcs
    L = 10
    r = make(L, [], step_reward=0.0)
    assert np.array_equal(r, np.zeros(L))
    td_mask = np.r_[np.ones(L - 3), np.zeros(3)]
    assert np.array_equal(mc(r, td_mask, 0.98), np.zeros(L))


def test_false_premise_penalty_is_independent_of_step_reward(iva_funcs):
    """With the step-reward horizon off, the -1 on an injected step is the only thing left in Q."""
    make, _ = iva_funcs
    r = make(10, [4], penalty=-1.0, step_reward=0.0)
    assert r[4] == -1.0
    assert np.array_equal(np.delete(r, 4), np.zeros(9))


def test_horizon_countdown_matches_the_closed_form(iva_funcs):
    """The Q horizon of the step-reward design follows exactly this recursion."""
    _, mc = iva_funcs
    gamma, L = 0.98, 40
    r = np.r_[np.full(L - 3, -1.0), np.zeros(3)].astype(np.float32)
    td_mask = np.r_[np.ones(L - 3), np.zeros(3)]
    K = np.maximum(0, L - 3 - np.arange(L))
    closed_form = -(1 - gamma ** K) / (1 - gamma)
    np.testing.assert_allclose(mc(r, td_mask, gamma), closed_form, atol=1e-5)
