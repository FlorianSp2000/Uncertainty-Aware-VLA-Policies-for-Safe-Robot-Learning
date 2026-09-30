"""Ensemble inference over IVA trajectories -> per-frame disagreement pickle.

Restores a trained SARSA ensemble from per-member checkpoints and computes
per-frame Q for every ordered IVA eval trajectory. Primary output is the
per-frame ensemble disagreement s_t = std over members of the per-timestep
min-inner-critic Q. One pickle per split:

    tp: feasible-only trajectories                    -> CP calibration set
    fp: trajectories with false-premise injections    -> detection set

Members are evaluated sequentially on a single GPU.

Knows nothing about conformal prediction — banding happens host-side in say_no.cp.
"""

import os
import pickle
from itertools import islice

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from flax.training import checkpoints
from flax.traverse_util import flatten_dict

from jaxrl_m.agents import agents
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors

from octo.data.iva import IVA_META_KEYS, IVA_TASKS, make_iva_dataset
from octo.data.dataset import apply_trajectory_transforms, apply_frame_transforms

# Reuses the training script's helpers and its --config/--name/--project flag
# definitions (train_iva_ensemble has a __main__ guard, import is side-effect
# safe apart from initializing the jax compilation cache).
from experiments.train_iva_ensemble import (
    create_ensemble_agents,
    detect_checkpoint_action_dim,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Ensemble run dir (contains ensemble_member_{0..7}/). Weights are not modified.")
flags.DEFINE_integer("step", None, "Checkpoint step to restore (e.g. 50000).")
flags.DEFINE_enum("split", None, ["tp", "fp"], "tp = calibration (feasible only), fp = detection.")
flags.DEFINE_list("tasks", IVA_TASKS, "IVA task names.")
flags.DEFINE_string("iva_data_dir", None, "Path to iva_dataset dir (contains Train/ and Eval/).")
flags.DEFINE_string("out", None, "Output pickle path. Must NOT be inside --model_dir.")
flags.DEFINE_integer("max_trajs", 0, "Per-task trajectory cap. 0 = all.")
flags.DEFINE_bool(
    "keep_images", True,
    "Store every RGB frame (PNG-encoded) so the dump is self-contained for plotting.",
)
flags.DEFINE_integer(
    "image_size", 128,
    "Edge length for stored frames. The model always sees 256; this only affects what "
    "--keep_images writes. IVA on disk is 128 and the 256 is a bilinear upsample from it, "
    "so 128 costs no real information and is ~3x smaller. 0 = store at model resolution.",
)
flags.DEFINE_enum(
    "data_split", "eval", ["eval", "train"],
    "Which IVA split to score. 'train' (100 episodes/task) gives a calibration set "
    "disjoint from the eval episodes the detector is tested on -- the leakage-free "
    "calibration source. 'eval' is the 25-episode-per-task test set.",
)
flags.DEFINE_float("alpha", 0.15, "CP significance level, recorded in the dump for the host-side pipeline.")
flags.DEFINE_bool(
    "log_param_tree", False,
    "Additionally dump every parameter array with its path, shape and dtype (~150 lines). "
    "Off by default: the always-on [wiring] block already reports which classes the config "
    "resolved to and the batch structure, which is what catches a config/checkpoint mismatch. "
    "Turn this on when a mismatch is suspected inside the param tree itself.",
)

flags.mark_flags_as_required(["model_dir", "step", "split", "iva_data_dir", "out"])


def param_l2_norm(agent):
    """Scalar fingerprint of a whole parameter tree.

    Sole purpose is the restore guard: `checkpoints.restore_checkpoint` returns
    `target` unchanged (only a log warning) when it finds nothing to load, so
    without comparing against the random-init norm we would happily score an
    untrained network. Pairwise comparison across members additionally catches
    every member having been restored from the same file.
    """
    leaves = jax.tree_util.tree_leaves(agent.state.params)
    return float(jnp.sqrt(sum(jnp.vdot(x, x).real for x in leaves)))


def param_count(agent):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(agent.state.params)))


def describe_tree(tree, prefix=""):
    """Yield 'path: shape dtype' lines for a nested dict of arrays / string lists."""
    for key in sorted(tree):
        val = tree[key]
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            yield from describe_tree(val, prefix=f"{path}/")
        elif isinstance(val, (list, tuple)):
            elem = type(val[0]).__name__ if len(val) else "empty"
            yield f"{path}: list[{len(val)}] of {elem}"
        else:
            arr = np.asarray(val)
            yield f"{path}: {tuple(arr.shape)} {arr.dtype}"


def log_param_tree(agent, tag):
    """Every param array with its path, shape and dtype -- verifies encoder/critic wiring."""
    flat = flatten_dict(jax.device_get(agent.state.params))
    logging.info(f"[{tag}] {len(flat)} param arrays, {param_count(agent):,} scalars")
    for path, arr in sorted(flat.items()):
        logging.info(f"[{tag}]   {'/'.join(path)}: {tuple(arr.shape)} {arr.dtype}")


def encode_frames(images):
    """Frames as stored in the dump, downscaled to --image_size if set.

    Storage only -- the critic has already seen the full-resolution frames by the
    time this runs. `area` is the correct kernel for downscaling and inverts the
    bilinear 128->256 upsample that apply_frame_transforms applied.
    """
    if not FLAGS.image_size or FLAGS.image_size == images.shape[1]:
        return images
    assert FLAGS.image_size <= images.shape[1], \
        f"--image_size {FLAGS.image_size} would upsample from {images.shape[1]}"
    resized = tf.image.resize(images, [FLAGS.image_size] * 2, method="area")
    return tf.cast(tf.round(resized), tf.uint8)


def build_task_iterator(task):
    """Finite full-trajectory iterator for one task.

    No .unbatch()/.repeat() here: each element is one whole trajectory and the
    iterator ends after the last one. train=False disables subsample_length and
    task augmentation, so trajectories keep their full length and true per-frame
    prompts.

    Order is reproducible for a given config.seed: _build_trajectories returns a
    canonical (task, episode) order, which the seeded .shuffle() then permutes
    deterministically. Join records across splits on `episode_id`, not position.
    """
    data = make_iva_dataset(
        data_dir=FLAGS.iva_data_dir,
        split=FLAGS.data_split,
        tasks=[task],
        seed=FLAGS.config.seed,
        discount=FLAGS.config.agent_kwargs.discount,
        use_tp_only=(FLAGS.split == "tp"),
        fp_strategy=FLAGS.config.iva_fp_strategy,
    )
    data = apply_trajectory_transforms(data, train=False, **FLAGS.config.traj_transform_kwargs.to_dict())
    data = apply_frame_transforms(data, train=False, **FLAGS.config.frame_transform_kwargs.to_dict())
    return data.iterator(prefetch=0)


def process_traj(traj, text_processor):
    """process_iva_batch equivalent that also carries task_feasible + language strings.

    Input: raw post-transform trajectory dict (numpy, whole trajectory).
    Output: batch consumable by SARSAEnsembleAgent.get_eval_values.
    """
    images = traj["observation"]["image_primary"].squeeze()
    actions = traj["action"].squeeze()
    task_feasible = np.asarray(traj["task_feasible"]).astype(bool)

    T = task_feasible.shape[0]
    assert images.shape == (T, 256, 256, 3), f"unexpected image shape {images.shape}"
    assert images.dtype == np.uint8, f"unexpected image dtype {images.dtype} (expected uint8)"
    assert actions.shape == (T, 8), f"unexpected action shape {actions.shape} (expected 8D IVA actions)"

    language_str = [s.decode("utf-8") for s in traj["task"]["language_instruction"]]
    assert len(language_str) == T

    language = text_processor.encode(language_str)
    # MUSE returns (T, 512); a per-frame embedding is required because IVA swaps
    # the prompt on injected frames only.
    assert language.shape[0] == T, f"language embedding {language.shape} has no leading T={T} axis"

    # Per-frame identity from IVA_META_KEYS. Constant within a trajectory for
    # task_id/episode_id -- if not, the (task, episode) grouping in
    # _build_trajectories is broken and every downstream join is wrong.
    meta = {k: np.asarray(traj[k]) for k in IVA_META_KEYS}
    for k in ("task_id", "episode_id"):
        assert meta[k].shape == (T,), f"{k} {meta[k].shape} != ({T},)"
        assert len(np.unique(meta[k])) == 1, f"{k} varies within trajectory: {np.unique(meta[k])}"
    assert meta["step_id"].shape == (T,), f"step_id {meta['step_id'].shape} != ({T},)"

    batch = {
        "observations": {"image": images},
        "actions": actions,
        "rewards": traj["reward"],
        "masks": traj["td_mask"],
        "goals": {
            "language": language,
            "language_str": language_str,
        },
        "task_feasible": task_feasible,
        "meta": meta,
    }
    return batch


def main(_):
    # TF must not grab the GPU (dataloading only).
    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed)

    run_dir = FLAGS.model_dir.rstrip("/")
    out_path = os.path.abspath(FLAGS.out)
    # Hard guardrail: never write anywhere near the checkpoint dir.
    assert not out_path.startswith(os.path.abspath(run_dir)), \
        f"--out {out_path} is inside --model_dir {run_dir}; refusing to write there"
    assert "ensemble_member" not in out_path, f"--out {out_path} points into a member dir"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ensemble_size = FLAGS.config.ensemble_size
    assert ensemble_size == 8, f"expected 8-member ensemble, config says {ensemble_size}" # might wanna relax this later on

    # Checkpoint sanity before doing any heavy work. Members whose save was
    # interrupted (orbax tmp dirs, e.g. Run 3 member 7 at 50k) are skipped —
    # disagreement is then computed over the complete members only.
    member_ids = [
        k for k in range(ensemble_size)
        if tf.io.gfile.exists(os.path.join(run_dir, f"ensemble_member_{k}", f"checkpoint_{FLAGS.step}"))
    ]
    missing = sorted(set(range(ensemble_size)) - set(member_ids))
    assert len(member_ids) >= 4, \
        f"only {len(member_ids)} complete member checkpoints at step {FLAGS.step} (need >=4)"
    if missing:
        logging.warning(f"members without checkpoint_{FLAGS.step}, skipped: {missing}")

    ckpt_action_dim = detect_checkpoint_action_dim(
        os.path.join(run_dir, f"ensemble_member_{member_ids[0]}")
    )
    assert ckpt_action_dim == 8, f"checkpoint action dim {ckpt_action_dim} != 8 (IVA)"

    text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)

    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    agent_class = agents[FLAGS.config.agent]

    # Wiring: which classes the config strings actually resolved to. Catches a
    # config/checkpoint mismatch (wrong encoder, wrong text processor) before
    # any inference happens, when the failure is still cheap and legible.
    logging.info(f"[wiring] devices: {jax.local_devices()}")
    logging.info(f"[wiring] seed={FLAGS.config.seed} data_split={FLAGS.data_split} "
                 f"split={FLAGS.split} tasks={list(FLAGS.tasks)}")
    logging.info(f"[wiring] agent      {FLAGS.config.agent!r} -> {agent_class.__module__}.{agent_class.__name__}")
    logging.info(f"[wiring] encoder    {FLAGS.config.encoder!r} -> {type(encoder_def).__name__} "
                 f"kwargs={dict(FLAGS.config.encoder_kwargs)}")
    logging.info(f"[wiring] text_proc  {FLAGS.config.text_processor!r} -> {type(text_processor).__name__} "
                 f"kwargs={dict(FLAGS.config.text_processor_kwargs)}")
    logging.info(f"[wiring] agent_kwargs {dict(FLAGS.config.agent_kwargs)}")
    logging.info(f"[wiring] traj_transform {FLAGS.config.traj_transform_kwargs.to_dict()}")
    logging.info(f"[wiring] frame_transform {FLAGS.config.frame_transform_kwargs.to_dict()}")
    logging.info(f"[wiring] iva_fp_strategy={FLAGS.config.iva_fp_strategy} "
                 f"use_tp_only={FLAGS.split == 'tp'} discount={FLAGS.config.agent_kwargs.discount}")

    # Template agent from the first trajectory of the first task. The iterator is
    # consumed only for its shapes and thrown away; the scoring loop below builds
    # a fresh iterator per task so every task takes the same code path.
    tasks = list(FLAGS.tasks)
    template_batch = process_traj(next(build_task_iterator(tasks[0])), text_processor)
    logging.info(f"[wiring] template batch structure ({tasks[0]}):")
    for line in describe_tree(template_batch):
        logging.info(f"[wiring]   {line}")

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, template_batch, encoder_def, agent_class, FLAGS.config.agent_kwargs
    )
    # create_ensemble_agents builds all `ensemble_size` members stacked along a
    # leading axis (the pmap layout used in training). Only the pytree STRUCTURE
    # is wanted here -- restore_checkpoint restores *into* a target of matching
    # shapes, the random weights themselves are never used. Slice member 0 as
    # that target and drop the rest, otherwise the full stacked tree sits in
    # device memory for the whole run.
    # TODO: jax.eval_shape around create_ensemble_agents would give the same
    # structure without materialising 8 random members; needs a shape-only path
    # in sarsa_ensemble.create_ensemble_vectorized.
    template_member = jax.tree_map(lambda x: x[0], ensemble_agents)
    template_norm = param_l2_norm(template_member)
    del ensemble_agents

    logging.info(
        f"[wiring] template member: {param_count(template_member):,} params, "
        f"randomly-initialised norm {template_norm:.2f}"
    )
    if FLAGS.log_param_tree:
        log_param_tree(template_member, "params")

    # Restore members individually (read-only) and keep them as a list;
    # inference walks them sequentially on one GPU.
    members = []
    norms = {}
    for k in member_ids:
        member_path = os.path.join(run_dir, f"ensemble_member_{k}")
        member = checkpoints.restore_checkpoint(member_path, target=template_member, step=FLAGS.step)
        norm = param_l2_norm(member)
        # restore_checkpoint returns `target` unchanged when it finds nothing to
        # load (no matching step, unreadable dir) instead of raising. An exact
        # match against the random-init norm is the only way to catch that.
        assert norm != template_norm, f"member {k}: restore returned template params (norm {norm})"
        assert param_count(member) == param_count(template_member), \
            f"member {k}: restored param count {param_count(member)} != template {param_count(template_member)}"
        logging.info(f"[restore] member {k} @ step {FLAGS.step}: param norm {norm:.4f}")
        norms[k] = norm
        members.append(member)
    n_members = len(members)

    # Independently trained members cannot coincide to float precision. Equal
    # norms mean every member was restored from the same file -- disagreement
    # would then be identically zero and the whole CP pipeline silently vacuous.
    assert len(set(norms.values())) == n_members, \
        f"restored members have duplicate param norms {norms}; members are not distinct"
    logging.info(f"[restore] {n_members} distinct members restored: {sorted(norms)}")

    # Seeds indexed by original member id so scores are stable under skips.
    # NOTE: with train=False the critic is deterministic, so the seed only feeds
    # the (unused here) OOD-action sampling inside get_eval_values. q_members is
    # seed-independent.
    seeds = jax.random.split(jax.random.PRNGKey(FLAGS.config.seed), ensemble_size)

    records = []
    for task in tasks:
        stream = (process_traj(t, text_processor) for t in build_task_iterator(task))
        if FLAGS.max_trajs:
            stream = islice(stream, FLAGS.max_trajs)

        episode_idx = 0
        n_frames = 0
        n_fp_frames = 0
        task_lengths = []
        for traj_batch in stream:
            # Only the arrays the critic actually reads reach the model. Keeps
            # the string/identity payload (language_str, meta) out of a pytree
            # that gets traced.
            model_batch = {k: traj_batch[k] for k in ("observations", "actions", "rewards", "masks")}
            model_goals = {"language": traj_batch["goals"]["language"]}
            q_members = np.stack(
                [
                    np.asarray(m.get_eval_values(model_batch, seeds[k], model_goals)["q"])
                    for k, m in zip(member_ids, members)
                ],
                axis=0,
            )  # (n_members, T): per-member per-timestep Q, already min over the 2 inner critics
            T = traj_batch["task_feasible"].shape[0]
            assert q_members.shape == (n_members, T), f"q_members {q_members.shape} != ({n_members}, {T})"
            assert np.isfinite(q_members).all(), f"[{task}] episode {episode_idx}: non-finite Q values"

            feasible = traj_batch["task_feasible"]
            meta = traj_batch["meta"]
            record = {
                "task": task,
                # Stable identity: joins straight back to the source JSON entry.
                "task_id": meta["task_id"][0].decode("utf-8"),
                "episode_id": int(meta["episode_id"][0]),
                "step_id": meta["step_id"],
                # Positional counter only -- carries no meaning across runs.
                "episode_idx": episode_idx,
                "length": T,
                "s_t": q_members.std(axis=0).astype(np.float32),
                "q_min_t": q_members.min(axis=0).astype(np.float32),
                "q_mean_t": q_members.mean(axis=0).astype(np.float32),
                "q_members": q_members.astype(np.float32),
                "task_feasible": feasible,
                "inj_idx": np.where(~feasible)[0],
                "language": traj_batch["goals"]["language_str"],
                "images": None,
                "image_encoding": None,
            }
            if FLAGS.keep_images:
                # Every frame, PNG-encoded, so the dump is self-contained for
                # plotting. SIZE: PNG only buys ~2.2x over raw uint8 on these
                # renders (86 KB vs 192 KB per 256x256 frame), so a full 9-task
                # run at 256 is ~6 GB across both splits -- hence --image_size.
                record["images"] = [
                    tf.io.encode_png(f).numpy() for f in encode_frames(traj_batch["observations"]["image"])
                ]
                record["image_encoding"] = "png"
                record["image_size"] = FLAGS.image_size or int(traj_batch["observations"]["image"].shape[1])

            records.append(record)
            if episode_idx == 0:
                logging.info(
                    f"[{task}] first traj: T={T}, q_members{q_members.shape}, "
                    f"s_t range [{record['s_t'].min():.3f}, {record['s_t'].max():.3f}], "
                    f"q_mean range [{record['q_mean_t'].min():.2f}, {record['q_mean_t'].max():.2f}], "
                    f"{int((~feasible).sum())} FP frames, prompt={traj_batch['goals']['language_str'][0]!r}"
                )
                # Per-member trajectory means: if these collapse onto one value
                # the members are effectively identical and s_t carries no signal.
                per_member = ", ".join(
                    f"m{k}={q_members[i].mean():.3f}" for i, k in enumerate(member_ids)
                )
                logging.info(f"[{task}] first traj per-member mean Q: {per_member}")
                logging.info(
                    f"[{task}] first traj id: episode_id={record['episode_id']} "
                    f"step_id[0..-1]={record['step_id'][0]}..{record['step_id'][-1]} "
                    f"stored_image_size={record['image_size'] if FLAGS.keep_images else 'n/a'}"
                )
                logging.info(
                    f"[{task}] first traj rewards[min/max]="
                    f"[{traj_batch['rewards'].min():.2f}, {traj_batch['rewards'].max():.2f}] "
                    f"masks[min/max]=[{traj_batch['masks'].min():.2f}, {traj_batch['masks'].max():.2f}] "
                    f"n_distinct_prompts={len(set(traj_batch['goals']['language_str']))}"
                )
            episode_idx += 1
            n_frames += T
            n_fp_frames += int((~feasible).sum())
            task_lengths.append(T)

        assert episode_idx > 0, f"task {task}: no trajectories yielded — wrong data dir or task name?"
        task_records = [r for r in records if r["task"] == task]
        # The whole point of plumbing episode_id through: it must identify a
        # trajectory uniquely, otherwise the host-side join is ambiguous.
        episode_ids = [r["episode_id"] for r in task_records]
        assert len(set(episode_ids)) == len(episode_ids), \
            f"[{task}] duplicate episode_id in {len(episode_ids)} trajectories: {sorted(episode_ids)}"
        assert all(r["task_id"] == task for r in task_records), \
            f"[{task}] task_id from the loader disagrees with the requested task"
        n_with_fp = sum(1 for r in task_records if len(r["inj_idx"]) > 0)
        logging.info(
            f"[{task}] split={FLAGS.split}: {episode_idx} trajs, {n_frames} frames, "
            f"{n_fp_frames} FP frames | length min/median/max "
            f"{min(task_lengths)}/{int(np.median(task_lengths))}/{max(task_lengths)} | "
            f"{n_with_fp}/{episode_idx} trajs contain >=1 FP frame"
        )

    total_fp = sum(len(r["inj_idx"]) for r in records)
    if FLAGS.split == "fp":
        assert total_fp > 0, "fp split produced zero false-premise frames — label plumbing is broken"
    else:
        assert total_fp == 0, f"tp split contains {total_fp} FP frames — _tp files should be FP-free"

    dump = {
        "model_dir": run_dir,
        "step": FLAGS.step,
        "split": FLAGS.split,
        "data_split": FLAGS.data_split,
        "alpha": FLAGS.alpha,
        "tasks": tasks,
        "member_ids": member_ids,
        "trajectories": records,
    }
    with open(out_path, "wb") as f:
        pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(out_path) / 1e6
    logging.info(
        f"Wrote {len(records)} trajectories from {len(tasks)} task(s), "
        f"{sum(r['length'] for r in records)} frames, {total_fp} FP frames, "
        f"{n_members} members -> {out_path} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    app.run(main)
