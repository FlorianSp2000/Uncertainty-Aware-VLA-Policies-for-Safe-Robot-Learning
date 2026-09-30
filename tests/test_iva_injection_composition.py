"""What IVA's false premises are actually made of, pinned against the raw JSONs.

`test_iva_instruction_inventory.py` asserts that no injected string equals a true-premise
one. That is a statement about whole strings, and it is easy to over-read into "the injected
words are novel". They are not: 98.4% of injected frames swap in another task's own target
object, and only 1.6% name something the corpus has never contained.

The distinction is load-bearing for the three-way prompt evaluation, whose whole point is to
separate "detects a contradiction" from "detects an unfamiliar string". If IVA's injections
were alien text the control would be measuring a different axis than the paper says, so the
composition is pinned here rather than restated in prose that can drift.

Skips when `iva_dataset/` is absent.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
IVA_DATA = REPO / "datasets" / "iva_dataset"
INSTRUCTION_RE = re.compile(r'The task is "(.*?)"')

IVA_TASKS = [
    "close_jar", "meat_off_grill", "open_drawer", "push_buttons", "put_money_in_safe",
    "reach_and_drag", "slide_block_to_color_target", "sweep_to_dustpan_of_size", "turn_tap",
]

#: The borrowed nouns, and the one task whose target object each of them is. The injector
#: draws from these seven and nothing else -- `button`, `dirt`, `steak`, `money` and `stick`
#: are target objects too and are never borrowed.
BORROWED_NOUN_OWNER = {
    "jar": "close_jar",
    "chicken": "meat_off_grill",
    "drawer": "open_drawer",
    "safe": "put_money_in_safe",
    "cube": "reach_and_drag",
    "block": "slide_block_to_color_target",
    "tap": "turn_tap",
}
NOVEL_NOUNS = {"tree", "bicycle", "elephant", "car", "book", "couch"}

EVAL_INJECTED_FRAMES = 2926
EVAL_INJECTED_STRINGS = 86
EVAL_BORROWED_FRAMES = 2880
EVAL_NOVEL_FRAMES = 46


def _instruction(entry):
    return INSTRUCTION_RE.search(entry["conversations"][0]["value"]).group(1).strip().lower()


@pytest.fixture(scope="module")
def eval_fp():
    """Per task: (true-premise strings, injected entries) straight out of the _fp files.

    An _fp file holds both: entries without `is_false_premise` carry the task's real
    instruction, the flagged ones carry the swap.
    """
    if not IVA_DATA.exists():
        pytest.skip(f"{IVA_DATA} absent - this test needs the IVA JSONs")
    out = {}
    for task in IVA_TASKS:
        entries = json.loads(
            (IVA_DATA / "Eval" / f"iva_eval_{task}_fp.json").read_text(encoding="utf-8"))
        true = {_instruction(e) for e in entries if not e.get("is_false_premise", False)}
        injected = [e for e in entries if e.get("is_false_premise", False)]
        out[task] = (true, injected)
    return out


@pytest.fixture(scope="module")
def swapped_nouns(eval_fp):
    """Frame counts per swapped-in noun, and the distinct strings each accounts for.

    The swapped token is the one word of the injection that its own task's true instructions
    never use. `test_every_injection_is_one_noun_substitution` is what licenses reading a
    single token out of that difference.
    """
    frames, strings = Counter(), {}
    for task, (true, injected) in eval_fp.items():
        vocab = {w for s in true for w in s.split()}
        for e in injected:
            new = [w for w in _instruction(e).split() if w not in vocab]
            assert len(new) == 1, (task, _instruction(e), new)
            frames[new[0]] += 1
            strings.setdefault(new[0], set()).add(_instruction(e))
    return frames, strings


def test_injection_totals(eval_fp):
    injected = [e for _, inj in eval_fp.values() for e in inj]
    distinct = {_instruction(e) for e in injected}
    assert len(injected) == EVAL_INJECTED_FRAMES
    assert len(distinct) == EVAL_INJECTED_STRINGS


def test_every_injection_is_one_noun_substitution(eval_fp):
    """Verb and modifiers survive; exactly one word changes. Without this the dataset would
    also be testing attribute errors ("teal"->"blue"), which it never does."""
    for task, (true, injected) in eval_fp.items():
        vocab = {w for s in true for w in s.split()}
        for e in injected:
            words = _instruction(e).split()
            new = [w for w in words if w not in vocab]
            assert len(new) == 1, (task, _instruction(e), new)
            parents = [s for s in true if len(s.split()) == len(words)]
            assert parents, (task, _instruction(e))


def test_almost_every_injection_borrows_another_tasks_object(swapped_nouns):
    """The headline: 2880 of 2926 frames use a word the model has seen thousands of times,
    in a combination it has not. Not alien text."""
    frames, _ = swapped_nouns
    borrowed = sum(c for n, c in frames.items() if n in BORROWED_NOUN_OWNER)
    novel = sum(c for n, c in frames.items() if n in NOVEL_NOUNS)
    assert borrowed == EVAL_BORROWED_FRAMES
    assert novel == EVAL_NOVEL_FRAMES
    assert borrowed + novel == EVAL_INJECTED_FRAMES


def test_the_borrowed_and_novel_noun_sets_are_exactly_these(swapped_nouns):
    frames, _ = swapped_nouns
    assert set(frames) == set(BORROWED_NOUN_OWNER) | NOVEL_NOUNS


def test_each_borrowed_noun_is_another_tasks_target_object(eval_fp, swapped_nouns):
    """Borrowing is 1:1 with the tasks, and a task never borrows its own object."""
    _, strings = swapped_nouns
    for noun, owner in BORROWED_NOUN_OWNER.items():
        users = {t for t, (true, _) in eval_fp.items() if any(noun in s.split() for s in true)}
        assert users == {owner}, (noun, users)
    for task, (_, injected) in eval_fp.items():
        for e in injected:
            swapped = next(n for n in strings if _instruction(e) in strings[n])
            assert BORROWED_NOUN_OWNER.get(swapped) != task, (task, _instruction(e))


def test_novel_nouns_appear_nowhere_in_the_corpus(eval_fp, swapped_nouns):
    frames, _ = swapped_nouns
    all_true = {w for true, _ in eval_fp.values() for s in true for w in s.split()}
    for noun in NOVEL_NOUNS:
        assert noun not in all_true, noun
        assert frames[noun] > 0, noun


def test_the_two_tiers_are_the_datasets_own_easy_flag(eval_fp, swapped_nouns):
    """`is_easy_false_premise` is not an independent axis: easy == borrowed noun, and it also
    decides the conversation shape (4 turns with a recoverable action, vs a 2-turn terminal
    refusal). Anything reading the last human turn gets the CORRECTED instruction on the
    2880 easy frames."""
    _, strings = swapped_nouns
    for task, (_, injected) in eval_fp.items():
        for e in injected:
            swapped = next(n for n in strings if _instruction(e) in strings[n])
            easy = e["is_easy_false_premise"]
            assert easy == (swapped in BORROWED_NOUN_OWNER), (task, _instruction(e))
            assert len(e["conversations"]) == (4 if easy else 2), (task, _instruction(e))
