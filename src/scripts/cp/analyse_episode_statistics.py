"""Evaluation behind the IVA thesis tables: test-split composition, which episode statistic,
frame vs episode level, detection time, operating points over alpha (plus training-seed
replicates), per task, and the frame-level reward-design comparison -> episode_statistics.json.
The .tex tables are rendered from that JSON by src/scripts/cp/write_iva_thesis_tables.py.

One checkpoint, one alpha, cross-task pooled quantile, K-fold x analysis seeds, all on the
three-way-prompt dump (see say_no.cp.episode_eval for the conditions and the fold protocol).
tau_alpha, the exceedance level of the counting statistics, is fixed a priori to alpha -- it is
not tuned on these episodes; a sensitivity sweep over it is saved in the JSON only.

Host-side only, the exact command is stored in the
JSON's "command" field. Numbers pinned in tests/test_episode_statistics_numbers.py.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from say_no.cp import _compat
from say_no.cp.episode_eval import frame_level, operating_point

STATISTIC_ROWS = [  # (label kept in the JSON, episode statistic, k, frame channels)
    (r"max over frames", "max", 0, "both"),
    (r"mean of top-5 frames", "topk_mean", 5, "both"),
    (r"count of frames $z_t>\tau$", "count_above", 0, "both"),
    (r"fraction of frames $z_t>\tau$", "frac_above", 0, "both"),
    (r"CUSUM, drift $\tau$", "cusum", 0, "both"),
    (r"max over frames, $-\bar Q$ only", "max", 0, "neg_mean_q"),
]
TAU_SWEEP = (0.01, 0.02, 0.10, 0.20)


def by_task_of(dump):
    by_task = {}
    for r in dump["trajectories"]:
        by_task.setdefault(r["task"], []).append(r)
    return by_task


def eval_split(by_task):
    """Per task: episodes, episodes with an injection, frames, injected frames."""
    out = {}
    for t in sorted(by_task):
        inj = [~np.asarray(r["task_feasible"]).astype(bool) for r in by_task[t]]
        out[t] = {"n_episodes": len(inj), "n_iva_episodes": sum(bool(m.any()) for m in inj),
                  "n_frames": int(sum(m.size for m in inj)),
                  "n_injected_frames": int(sum(m.sum() for m in inj))}
    return out


def reward_design_rows(three_way):
    """Frame AUROCs per checkpoint from the analyse_three_way_prompt.py JSON; step reward and
    negative-demonstration ratio parsed from its labels, which are the producer's record."""
    rows = []
    for label, m in three_way.items():
        sr = re.search(r"sr (-?\d+)", label)
        assert sr, label
        nd = re.search(r"negdemo ([\d.]+)", label)
        rows.append({"label": label, "step_reward": int(sr.group(1)),
                     "negdemo_ratio": float(nd.group(1)) if nd else 0.0,
                     "n_members": m["n_members"], "step": m["step"],
                     "auroc_iva_injection": m["summary"]["mean_auroc_inj"],
                     "auroc_foreign_instruction": m["summary"]["mean_auroc_other_inj"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="three-way-prompt pkl of one checkpoint")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--tau-alpha", type=float, required=True)
    ap.add_argument("--sweep-alphas", required=True, help="nominal levels of the operating-point table")
    ap.add_argument("--replicate", action="append", default=[],
                    help="three-way-prompt pkl of a training-seed replicate, scored at --alpha")
    ap.add_argument("--three-way-json", required=True,
                    help="analyse_three_way_prompt.py output with the checkpoints to compare")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-folds", type=int, default=5)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    sweep_alphas = [float(x) for x in a.sweep_alphas.split(",")]
    assert a.alpha in sweep_alphas, "the headline alpha must be a row of the sweep"
    dump = _compat.load(a.dump)
    by_task = by_task_of(dump)
    common = dict(seeds=seeds, n_folds=a.n_folds, pool_tasks=True)

    rows = [dict(label=lab, **operating_point(by_task, statistic=st, k=k, frame_statistic=fs,
                                              tau_alpha=a.tau_alpha, alpha=a.alpha, **common))
            for lab, st, k, fs in STATISTIC_ROWS]
    sweep = [operating_point(by_task, statistic=st, k=0, frame_statistic="both", tau_alpha=ta,
                             alpha=a.alpha, **common)
             for ta in TAU_SWEEP for st in ("count_above", "frac_above", "cusum")]
    frame = frame_level(by_task, alpha=a.alpha, seeds=seeds, n_folds=a.n_folds)
    alpha_sweep = [operating_point(by_task, statistic="max", k=0, frame_statistic=fs,
                                   alpha=al, **common)
                   for fs in ("both", "neg_mean_q") for al in sweep_alphas]
    replicates = []
    for pkl in a.replicate:
        rd = _compat.load(pkl)
        seed = re.search(r"_b\d+s(\d+)_", rd["model_dir"])
        assert seed, rd["model_dir"]
        replicates.append({"pkl": pkl, "model_dir": rd["model_dir"], "step": rd["step"],
                           "n_members": len(rd["member_ids"]), "training_seed": int(seed.group(1)),
                           **operating_point(by_task_of(rd), statistic="max", k=0,
                                             frame_statistic="both", alpha=a.alpha, **common)})
    split = eval_split(by_task)
    reward = reward_design_rows(json.loads(Path(a.three_way_json).read_text(encoding="utf-8")))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"command": " ".join(sys.argv), "pkl": a.dump, "model_dir": dump["model_dir"],
           "step": dump["step"], "n_members": len(dump["member_ids"]), "alpha": a.alpha,
           "tau_alpha": a.tau_alpha, "seeds": seeds, "n_folds": a.n_folds,
           "statistics": rows, "tau_alpha_sweep": sweep, "frame_level": frame,
           "alpha_sweep": alpha_sweep, "replicates": replicates, "eval_split": split,
           "reward_design": reward}
    (a.out_dir / "episode_statistics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out_dir / 'episode_statistics.json'}")


if __name__ == "__main__":
    main()
