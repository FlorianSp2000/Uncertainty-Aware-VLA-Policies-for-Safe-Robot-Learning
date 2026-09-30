"""Rebuild every figure of thesis/plots that code draws; `--copy <thesis>/plots` puts them in place.

REGISTRY: thesis/plots file name -> producer command, the file it writes, the width the thesis
includes it at (fraction of \\textwidth; tests/test_thesis_figures.py checks it against the .tex
and against the drawn figure). EXCLUDED: includes no code here draws. All figures use the
THESIS profile of say_no.utils.figstyle.
"""
from __future__ import annotations

import sys

from say_no.utils.figstyle import FULL, HALF
from say_no.utils.figure_registry import Entry, run
from say_no.vla_rollouts import OPERATING_ALPHA

RP = "src/scripts/render_plot.py"
PLOTS = "src/scripts/plots"
THESIS = "src/results/thesis"
AT = "src/results/perturbation-sensitivity-bridge-fractal/analysis_thesis"
C = "src/results/perturbation-sensitivity-bridge-fractal/controls"
EMB = "src/results/embedding-baselines-bridge-fractal"
IO = "src/results/instruction-overlap-bridge-fractal"
IVA = "src/results/episode-operating-point-iva/thesis"
IVA_CP = "src/data/iva_cp"
VLA_SP = "src/results/vla-rollout-failure-detection/safe_protocol"
VLA_ALPHA = str(OPERATING_ALPHA)


def _yaml(stem: str, name: str, width: float) -> Entry:
    return Entry(RP, (f"{PLOTS}/{stem}.yaml",), f"{THESIS}/{name}", width, wandb=True)


_PERTURBATION = ("src/scripts/plot_perturbation_thesis_figures.py", ("--json", f"{AT}/perturbation_thesis.json", "--out_dir", AT))

REGISTRY: dict[str, Entry] = {
    "baseline_critic_loss.pdf": _yaml("thesis_baseline_critic_loss", "baseline_critic_loss.pdf", HALF),
    "baseline_mean_q.pdf": _yaml("thesis_baseline_mean_q", "baseline_mean_q.pdf", HALF),
    "contrastive_hardest_distances.pdf": _yaml("thesis_contrastive_hardest_distances", "contrastive_hardest_distances.pdf", HALF),
    "contrastive_separation.pdf": _yaml("thesis_contrastive_separation", "contrastive_separation.pdf", HALF),
    "reward_penalty_critic_loss.pdf": _yaml("thesis_reward_penalty_critic_loss", "reward_penalty_critic_loss.pdf", 0.65),
    "roboreward_inpainting_auroc.pdf": _yaml("thesis_roboreward_inpainting_auroc", "roboreward_inpainting_auroc.pdf", 0.65),
    # drawn from the analysis JSON; `analyse_perturbation_thesis.py` (docs/REPRODUCE.md) produces it
    "auroc_random_vs_nearest_relabeling.pdf": Entry(*_PERTURBATION, f"{AT}/auroc_random_vs_nearest_relabeling.pdf", FULL),
    "auroc_object_removal_vs_controls.pdf": Entry(*_PERTURBATION, f"{AT}/auroc_object_removal_vs_controls.pdf", FULL),
    "object_removal_controls_examples.pdf": Entry(
        "src/scripts/plot_image_condition_examples.py",
        ("--pkl", "src/data/bridge_fractal_n300_with_inpainted_cleaned.pkl", "--hdf5", f"{C}/nonegdemo_n300_inpainted.hdf5",
         "--radius", "16", "--frame", "first", "--examples", "bridge:30", "fractal:148",
         "--out", f"{AT}/object_removal_controls_examples.pdf"),
        f"{AT}/object_removal_controls_examples.pdf", FULL, slow=True),
    "disagreement_original_vs_object_removal.pdf": Entry(
        "src/scripts/plot_inpainting_score_distributions.py",
        ("--dump", f"{C}/nonegdemo_n300_inpainted.hdf5", "--profile", "thesis", "--width", "0.62",
         "--conformal_alpha", "0.1", "--out", f"{AT}/disagreement_original_vs_object_removal.pdf"),
        f"{AT}/disagreement_original_vs_object_removal.pdf", 0.62),
    # score violins (say_no.perturbation.violins), inputs as analysis_thesis/ and instruction-overlap (docs/REPRODUCE.md)
    "bf_detector_score_violins.pdf": Entry(
        "src/scripts/plot_perturbation_score_violins.py",
        ("detectors", *(a for lab, f in (("SayNo", "nonegdemo"), ("SayNo-CR", "negdemo0.10"),
                                         ("ResNet--MUSE 3 (per sample)", "cls_20260111"), ("Octo (per sample)", "octo_full_s100k"))
                        for a in ("--val", f"{lab}={C}/{f}_val_n200.hdf5", "--inpainting", f"{lab}={C}/{f}_n300_inpainted.hdf5")),
         "--train_ref", f"SayNo={C}/nonegdemo_train_n200_feat.hdf5", "--train_ref", f"SayNo-CR={C}/negdemo0.10_train_n200_feat.hdf5",
         "--clip_scores", f"val200={EMB}/clip_scores_val_n200.hdf5",
         "--clip_scores", f"inpainting258={EMB}/clip_scores_n300_inpainted.hdf5", "--clip_label", "CLIP ViT-B/32",
         "--color", "SayNo=sayno_no_relabeling", "--color", "SayNo-CR=sayno_relabeling_rho10",
         "--color", "ResNet--MUSE 3 (per sample)=resnet_muse_classifier", "--color", "Octo (per sample)=octo_classifier",
         "--k", "5", "--main_ensembles", "SayNo", "SayNo-CR", "--out", f"{AT}/bf_detector_score_violins.pdf"),
        f"{AT}/bf_detector_score_violins.pdf", FULL, slow=True),
    "bf_seen_unseen_task_violins.pdf": Entry(
        "src/scripts/plot_perturbation_score_violins.py",
        ("seen_unseen", "--walk", f"{IO}/instructions_bridge_fractal_full.hdf5",
         "--train", f"SayNo={IO}/dumps/nonegdemo_train_n200_inpainted.hdf5",
         "--train", f"SayNo-CR={IO}/dumps/negdemo0.10_train_n200_inpainted.hdf5",
         "--inpainting", f"SayNo={C}/nonegdemo_n300_inpainted.hdf5", "--inpainting", f"SayNo-CR={C}/negdemo0.10_n300_inpainted.hdf5",
         "--main_ensembles", "SayNo", "SayNo-CR", "--out", f"{AT}/bf_seen_unseen_task_violins.pdf"),
        f"{AT}/bf_seen_unseen_task_violins.pdf", FULL),
    "iva_example_episode_three_instructions.pdf": Entry(
        "src/scripts/cp/plot_episode_detector_examples.py",
        ("--dump", f"{IVA_CP}/three_way_prompt/A1noclock200k_prompt3_step200000_eval-fp_all.pkl", "--alpha", "0.05",
         "--out", f"{IVA}/iva_example_episode_three_instructions.pdf"),
        f"{IVA}/iva_example_episode_three_instructions.pdf", FULL, slow=True),
    "sayno_overview.pdf": Entry(
        "src/scripts/plot_sayno_overview.py",
        ("--inpainting-pkl", "src/data/bridge_fractal_n300_with_inpainted_cleaned.pkl",
         "--dump", f"{C}/nonegdemo_n300_inpainted.hdf5", "--scene", "fractal:148", "--train-scene", "bridge:125",
         "--calib-dump", f"{IVA_CP}/three_way_prompt/A1noclock200k_prompt3_step200000_eval-fp_all.pkl",
         "--calib-task", "close_jar", "--alpha", "0.05",
         "--out", f"{THESIS}/sayno_overview.pdf"),
        f"{THESIS}/sayno_overview.pdf", FULL, slow=True),
    # VLA rollouts, SAFE protocol (3 folds of seen / unseen tasks):
    # results + dumps under {VLA_SP}/<dataset>/fold<k>, frames under src/data/safe_widowx/frames
    "vla_detection_over_finetuning.pdf": Entry(
        "src/scripts/plot_vla_rollout_safe_protocol_curves.py",
        ("--root", VLA_SP, "--datasets", "safe_widowx", "libero_pi0fast",
         "--out", f"{VLA_SP}/thesis/vla_detection_over_finetuning.pdf"),
        f"{VLA_SP}/thesis/vla_detection_over_finetuning.pdf", FULL),
    # single unseen-task rollouts, eval-seen calibration, calibration draw 0
    **{f"vla_example_{name}.pdf": Entry(
        "src/scripts/plot_vla_rollout_examples.py",
        ("--dump", f"{VLA_SP}/safe_widowx/fold{fold}/dumps/eval_step50000_holdout-evalseen-unseen.pkl",
         "--view", "eval_seen:unseen", "--task", key.split("__")[0], "--alpha", VLA_ALPHA,
         "--frames_dir", "src/data/safe_widowx/frames", "--key", key, "--steps", steps,
         "--event_step", event, "--event_label", label, "--out", f"{VLA_SP}/thesis/vla_example_{name}.pdf"),
        f"{VLA_SP}/thesis/vla_example_{name}.pdf", FULL, slow=True)
       for name, fold, key, steps, event, label in (
           ("cup_knocked_over", 0, "put_blue_cup_on_plate__ep001__task_put_blue_cup_on_plate_2__fail",
            "0,12,16,28,40,49", "16", "Cup tipped over"),
           ("cup_missed_grasp", 0, "put_blue_cup_on_plate__ep030__task_put_blue_cup_on_plate_1__fail",
            "0,20,30,36,41,49", "36", "Gripper reaches plate empty"),
           ("block_behind_schedule", 1, "put_the_red_block_into_the_pot__ep009__task_put_the_red_block_into_the_pot_1__fail",
            "0,15,21,30,40,49", "40", "Arm leaves without block"),
           ("eggplant_hover_missed", 0, "lift_eggplant__ep038__task_lift_eggplant_1__fail",
            "0,14,20,30,40,49", "20", "Gripper reaches eggplant"))},
}

EXCLUDED: dict[str, str] = {
}

if __name__ == "__main__":
    run(REGISTRY, sys.argv[1:], description=__doc__.split("\n")[0], copy_allowed=True)
