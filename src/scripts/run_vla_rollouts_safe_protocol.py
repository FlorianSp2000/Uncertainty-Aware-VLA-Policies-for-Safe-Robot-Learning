"""VLA-rollout failure detection under the SAFE protocol: every (dataset, fold, view) analysis
plus the cross-fold summaries, from the dumps of ``slurm/eval-vla-rollouts.sbatch``.

Expects ``<root>/<dataset>/fold<k>/dumps/eval_<label>_<pools>.pkl`` with label
``zeroshot`` or ``step<N>``; writes ``<root>/<dataset>/fold<k>/<view>/results.{json,md}`` via
``analyse_vla_rollout_failure_detection.py`` and ``<root>/summary_<dataset>_<view>_<label>.md``
via ``summarise_vla_rollout_results.py``. Invocation: docs/REPRODUCE.md, section 3b.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent
# view name -> (calibration pool:test pool, horizon); A is SAFE's own. Every LIBERO failure is
# a timeout: "calib_task_or_global" cuts each episode at its task's shortest calibration success
# (unseen tasks: the overall shortest), so length carries no label; Foresight scores full episodes.
SAFE_VIEWS = {"A_evalseen-cal_unseen-test": "eval_seen:unseen",
              "B_holdout-cal_evalseen-test": "holdout:eval_seen",
              "C_holdout-cal_unseen-test": "holdout:unseen"}
DATASETS = {
    "safe_widowx": dict(num_terminal=3, pooltag="holdout-evalseen-unseen",
                        views={v: (p, "none") for v, p in SAFE_VIEWS.items()}),
    "libero_pi0fast": dict(num_terminal=1, pooltag="holdout-evalseen-unseen",
                           views={v: (p, "calib_task_or_global") for v, p in SAFE_VIEWS.items()}),
    # Foresight's 3-fold CV (zhang2026, App. 8): full episodes as Foresight scores them, and cut
    "libero_pi0fast_cv": dict(num_terminal=1, pooltag="holdout-cvcalib-cvtest",
                              views={"D_cvcalib-cal_cvtest-test_full": ("cv_calib:cv_test", "none"),
                                     "E_cvcalib-cal_cvtest-test_cut": ("cv_calib:cv_test",
                                                                       "calib_task_or_global")}),
}
ACTIONS_KEY = {"safe_widowx": "safe_widowx", "libero_pi0fast": "libero_pi0fast",
               "libero_pi0fast_cv": "libero_pi0fast"}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--actions_npz", nargs="+", required=True, help="dataset=path")
    ap.add_argument("--policy_csv_dir", default=None, help="SAFE's openvla_widowx dir (token entropy)")
    ap.add_argument("--alpha", default="0.15")
    ap.add_argument("--n_boot", default="2000")
    args = ap.parse_args()
    actions = dict(a.split("=", 1) for a in args.actions_npz)
    root = Path(args.root)

    for ds in args.datasets:
        cfg = DATASETS[ds]
        dump_re = re.compile(rf"^eval_(?P<label>zeroshot|step\d+)_{cfg['pooltag']}\.pkl$")
        folds = sorted(p for p in (root / ds).glob("fold*") if (p / "dumps").is_dir())
        if not folds:
            raise FileNotFoundError(f"no {root / ds}/fold*/dumps")
        labels_per_fold = {}
        for fold in folds:
            dumps = {dump_re.match(p.name)["label"]: p for p in (fold / "dumps").glob("*.pkl")
                     if dump_re.match(p.name)}
            labels_per_fold[fold.name] = set(dumps)
            for view, (pools, horizon) in cfg["views"].items():
                cmd = [sys.executable, str(SCRIPTS / "analyse_vla_rollout_failure_detection.py"),
                       "--dataset", f"{ds} {fold.name} {view}", "--view", pools,
                       "--dump", *[f"{k}={v}" for k, v in sorted(dumps.items())],
                       "--discount", "0.98", "--num_terminal", str(cfg["num_terminal"]),
                       "--horizon", horizon, "--actions_npz", actions[ACTIONS_KEY[ds]],
                       "--alpha", args.alpha, "--n_boot", args.n_boot,
                       "--out_dir", str(fold / view)]
                if ds == "safe_widowx" and args.policy_csv_dir:
                    cmd += ["--policy_csv_dir", args.policy_csv_dir]
                run(cmd)
        common = set.intersection(*labels_per_fold.values())
        for view in cfg["views"]:
            for label in sorted(common):
                entries = [f"{f.name}={f / view / 'results.json'}:{label}" for f in folds]
                run([sys.executable, str(SCRIPTS / "summarise_vla_rollout_results.py"),
                     "--entry", *entries, "--alpha", args.alpha,
                     "--group", f"{len(folds)} folds=" + ";".join(f.name for f in folds),
                     "--out", str(root / f"summary_{ds}_{view}_{label}.md")])


if __name__ == "__main__":
    main()
