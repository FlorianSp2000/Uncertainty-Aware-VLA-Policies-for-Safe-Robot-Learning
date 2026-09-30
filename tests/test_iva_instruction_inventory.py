"""The three-way prompt control stands on these strings being the right ones.

`instruction_inventory` supplies both halves of the three-way prompt evaluation: each task's
representative instruction (the "swapped task prompt" the control scores) and
each episode's true instruction (the positive condition, and the thing the representative
must never accidentally be). A silent error in either turns the control into nonsense that
still produces a plausible AUROC, so the invariants are pinned here.

Imported by source text rather than `from octo.data.iva import ...` -- the module pulls in
tensorflow/dlimp, which the host analysis venv does not carry.
"""

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

REPO = Path(__file__).resolve().parents[1]
IVA_PY = REPO / "external" / "V-GPS" / "octo" / "octo" / "data" / "iva.py"
IVA_DATA = REPO / "datasets" / "iva_dataset"

IVA_TASKS = [
    "close_jar", "meat_off_grill", "open_drawer", "push_buttons", "put_money_in_safe",
    "reach_and_drag", "slide_block_to_color_target", "sweep_to_dustpan_of_size", "turn_tap",
]


@pytest.fixture(scope="module")
def inventory_fn():
    """`instruction_inventory` and its helpers, lifted out of the tf-importing module."""
    src = IVA_PY.read_text(encoding="utf-8")
    ns = {
        "json": json, "re": re, "Path": Path, "Counter": Counter, "defaultdict": defaultdict,
        "Dict": Dict, "List": List, "Optional": Optional, "Tuple": Tuple,
        "IVA_TASKS": IVA_TASKS,
        "_TASK_INSTRUCTION_RE": re.compile(r'The task is "(.*?)"'),
    }
    wanted = {"_split_fp_json_paths", "instruction_inventory", "_instruction_of"}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module([node], []), str(IVA_PY), "exec"), ns)
    missing = wanted - set(ns)
    assert not missing, f"functions no longer present in iva.py: {missing}"
    return ns["instruction_inventory"]


@pytest.fixture(scope="module")
def eval_inventory(inventory_fn):
    if not IVA_DATA.exists():
        pytest.skip(f"{IVA_DATA} absent - this test needs the IVA JSONs")
    return inventory_fn(str(IVA_DATA), "eval")


@pytest.fixture(scope="module")
def train_inventory(inventory_fn):
    if not IVA_DATA.exists():
        pytest.skip(f"{IVA_DATA} absent - this test needs the IVA JSONs")
    return inventory_fn(str(IVA_DATA), "train")


def test_every_eval_episode_the_loader_yields_has_a_true_instruction(eval_inventory):
    """224, not 225: open_drawer/episode17 is a 1-frame terminal refusal that the loader
    drops (iva.py logs "dropping open_drawer/episode17") and that carries no true premise
    anywhere, so there is nothing to recover."""
    _, episode_true = eval_inventory
    assert len(episode_true) == 224
    assert ("open_drawer", 17) not in episode_true


def test_an_episode_never_has_two_true_instructions(eval_inventory):
    """instruction_inventory raises on ambiguity; reaching here means it found none."""
    _, episode_true = eval_inventory
    assert all(isinstance(v, str) and v for v in episode_true.values())


def test_representatives_are_distinct_and_familiar(train_inventory, eval_inventory):
    """The control's premise: a real string the model saw thousands of times."""
    train_counts, _ = train_inventory
    reps = {t: max(train_counts[t].items(), key=lambda kv: (kv[1], kv[0])) for t in IVA_TASKS}
    assert len(set(s for s, _ in reps.values())) == len(IVA_TASKS)
    assert all(n >= 2000 for _, n in reps.values()), \
        {t: n for t, (_, n) in reps.items()}


def test_no_representative_is_correct_for_another_tasks_scene(train_inventory, eval_inventory):
    """If a task's representative were also some other task's true instruction, the control
    would score a CORRECT prompt as a hard negative and the AUROC would be meaningless."""
    train_counts, _ = train_inventory
    _, episode_true = eval_inventory
    reps = {t: max(train_counts[t].items(), key=lambda kv: (kv[1], kv[0]))[0] for t in IVA_TASKS}
    clashes = [
        (task, episode, src) for (task, episode), true in episode_true.items()
        for src in IVA_TASKS if src != task and reps[src] == true
    ]
    assert clashes == []


def test_injections_are_never_a_swapped_task_prompt(eval_inventory):
    """The confound the three-way evaluation exists to measure: IVA's false premises are novel
    STRINGS, not misapplied real ones. If this ever fails, the benchmark changed.

    Novel as strings only. The words are almost all familiar -- 98.4% of injected frames
    swap in another task's own target object, see test_iva_injection_composition.py -- so
    this must not be read as the injections being unfamiliar text."""
    eval_counts, _ = eval_inventory
    valid = {s for c in eval_counts.values() for s in c}
    injected = set()
    pattern = re.compile(r'The task is "(.*?)"')
    for task in IVA_TASKS:
        entries = json.loads((IVA_DATA / "Eval" / f"iva_eval_{task}_fp.json").read_text(encoding="utf-8"))
        for e in entries:
            if e.get("is_false_premise", False):
                injected.add(pattern.search(e["conversations"][0]["value"]).group(1))
    assert injected & valid == set()
