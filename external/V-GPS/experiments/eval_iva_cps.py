"""Contrastive Prompt Scoring: score every frame under the given prompt AND K references.

Motivation (measured in a paired-twin analysis of the IVA dumps):
the ensemble's disagreement s_t responds to a false premise by ~+1.0 units against
a 5e-4 noise floor -- a ~2000-sigma effect. Yet the deployable detector only
reaches AUROC ~0.65, because the BETWEEN-frame spread of s_t across scenes and
timesteps is far larger than that shift. The signal is there; the nuisance
variance hides it.

The twin difference s_t(wrong prompt) - s_t(true prompt) cancels that nuisance
exactly, and lifts AUROC to ~0.82 -- but it needs the ground-truth prompt, so it
is an oracle, not a detector.

This script makes the same cancellation deployable: hold the frame fixed and vary
the PROMPT. For each frame we score the given instruction plus K reference
instructions drawn from a fixed pool. A true premise should sit at the low-
disagreement end of that reference distribution; a false premise should look like
just another random prompt. The per-frame normalised rank of the given prompt is
then a self-normalising statistic -- no ground truth, and the scene/timestep
nuisance divides out frame by frame.

Reuses eval_iva_cp's dataloading and restore path; adds only the multi-prompt
scoring. Uses a lean forward_critic call (1 encoder pass) rather than
get_eval_values (4 passes), so K=8 references cost ~2x the baseline run.

Imported by eval_iva_three_way_prompt.py (score_under_prompt).
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

from jaxrl_m.agents import agents
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors

from octo.data.iva import IVA_TASKS, make_iva_dataset

# Importing eval_iva_cp defines the shared flags (--model_dir, --step, --split,
# --data_split, --tasks, --iva_data_dir, --out, --max_trajs, ...) and gives us
# its dataloading/restore helpers. Single source of truth for both scripts.
from experiments.eval_iva_cp import (
    build_task_iterator,
    process_traj,
    param_l2_norm,
)
from experiments.train_iva_ensemble import (
    create_ensemble_agents,
    detect_checkpoint_action_dim,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer("n_ref_prompts", 8, "Reference prompts scored per frame, besides the given one.")
flags.DEFINE_integer("prompt_seed", 0, "Seed for drawing the reference prompt pool.")
flags.DEFINE_bool(
    "cross_task_refs", True,
    "Draw reference prompts from ALL tasks (True) or only the current task's own "
    "instruction set (False). Cross-task references are the realistic pool: at "
    "deployment you do not know which task the scene belongs to.",
)


def collect_prompt_pool(data_dir, data_split, tasks):
    """Every distinct language instruction in the split, as plain strings.

    Read straight from the trajectory builder rather than the JSON so the pool is
    exactly the set of prompts the model could be shown.
    """
    pool = set()
    for task in tasks:
        data = make_iva_dataset(
            data_dir=data_dir, split=data_split, tasks=[task], seed=0,
            discount=0.98, use_tp_only=True, fp_strategy="copy_prev",
            load_images_as_bytes=False,  # strings only; do not read the PNGs
        )
        for traj in data.iterator(prefetch=0):
            for s in traj["task"]["language_instruction"]:
                pool.add(s.decode("utf-8"))
    return sorted(pool)


def score_under_prompt(member, obs_images, actions, lang_emb):
    """min-over-inner-critics Q for one member under one prompt embedding.

    One encoder pass. get_eval_values would run four (online/target x
    actions/ood) and we need none of the extras here.
    """
    goals = {"language": lang_emb}
    q = member.forward_critic(
        ({"image": obs_images}, goals), actions, jax.random.PRNGKey(0), train=False
    )
    return np.asarray(jnp.min(q, axis=0))


def main(_):
    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed)

    run_dir = FLAGS.model_dir.rstrip("/")
    out_path = os.path.abspath(FLAGS.out)
    assert not out_path.startswith(os.path.abspath(run_dir)), "refusing to write into --model_dir"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ensemble_size = FLAGS.config.ensemble_size
    member_ids = [
        k for k in range(ensemble_size)
        if tf.io.gfile.exists(os.path.join(run_dir, f"ensemble_member_{k}", f"checkpoint_{FLAGS.step}"))
    ]
    assert len(member_ids) >= 4, f"only {len(member_ids)} complete members at step {FLAGS.step}"
    assert detect_checkpoint_action_dim(
        os.path.join(run_dir, f"ensemble_member_{member_ids[0]}")) == 8, "checkpoint action dim != 8"

    text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)
    tasks = list(FLAGS.tasks)

    # Reference prompt pool. Cross-task by default: at deployment the scene's task
    # identity is unknown, so the reference set must span tasks.
    pool_tasks = list(IVA_TASKS) if FLAGS.cross_task_refs else tasks
    pool = collect_prompt_pool(FLAGS.iva_data_dir, FLAGS.data_split, pool_tasks)
    logging.info(f"[cps] prompt pool: {len(pool)} distinct instructions from {len(pool_tasks)} task(s)")
    assert len(pool) > FLAGS.n_ref_prompts, (
        f"prompt pool has only {len(pool)} instructions, need > n_ref_prompts="
        f"{FLAGS.n_ref_prompts}; is --cross_task_refs set?")
    for p in pool[:12]:
        logging.info(f"[cps]   {p!r}")

    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    agent_class = agents[FLAGS.config.agent]
    template_batch = process_traj(next(build_task_iterator(tasks[0])), text_processor)
    ensemble_agents = create_ensemble_agents(
        ensemble_size, jax.random.PRNGKey(FLAGS.config.seed), template_batch,
        encoder_def, agent_class, FLAGS.config.agent_kwargs,
    )
    template_member = jax.tree_map(lambda x: x[0], ensemble_agents)
    template_norm = param_l2_norm(template_member)
    del ensemble_agents

    members, norms = [], {}
    for k in member_ids:
        m = checkpoints.restore_checkpoint(
            os.path.join(run_dir, f"ensemble_member_{k}"), target=template_member, step=FLAGS.step)
        n = param_l2_norm(m)
        assert n != template_norm, f"member {k}: restore returned template params"
        norms[k] = n
        members.append(m)
    assert len(set(norms.values())) == len(members), f"duplicate member norms {norms}"
    logging.info(f"[cps] {len(members)} distinct members restored: {sorted(norms)}")

    records = []
    for task_i, task in enumerate(tasks):
        stream = (process_traj(t, text_processor) for t in build_task_iterator(task))
        if FLAGS.max_trajs:
            stream = islice(stream, FLAGS.max_trajs)

        # Fixed reference prompts per task, drawn once so every frame of every
        # episode is compared against the same reference set. Seeded by task
        # INDEX, not hash(task) -- str hashing is salted per process.
        rng = np.random.default_rng(FLAGS.prompt_seed * 1000 + task_i)
        refs = list(rng.choice(pool, size=min(FLAGS.n_ref_prompts, len(pool)), replace=False))
        ref_emb = text_processor.encode(refs)  # (K, 512)

        n_ep = 0
        for b in stream:
            images, actions = b["observations"]["image"], b["actions"]
            T = b["task_feasible"].shape[0]

            per_member = np.stack(
                [score_under_prompt(m, images, actions, b["goals"]["language"]) for m in members], 0)
            assert per_member.shape == (len(members), T)
            s_true = per_member.std(axis=0)
            q_true = per_member.mean(axis=0)

            s_ref = np.empty((len(refs), T), np.float32)
            q_ref = np.empty((len(refs), T), np.float32)
            for j in range(len(refs)):
                tiled = np.repeat(ref_emb[j][None, :], T, axis=0)
                pm = np.stack([score_under_prompt(m, images, actions, tiled) for m in members], 0)
                s_ref[j] = pm.std(axis=0)
                q_ref[j] = pm.mean(axis=0)

            meta = b["meta"]
            records.append({
                "task": task,
                "task_id": meta["task_id"][0].decode("utf-8"),
                "episode_id": int(meta["episode_id"][0]),
                "step_id": meta["step_id"],
                "length": T,
                "task_feasible": b["task_feasible"],
                "inj_idx": np.where(~b["task_feasible"])[0],
                "language": b["goals"]["language_str"],
                "refs": refs,
                "s_true": s_true.astype(np.float32),
                "q_true": q_true.astype(np.float32),
                "s_ref": s_ref,
                "q_ref": q_ref,
            })
            if n_ep == 0:
                r = records[-1]
                rank_s = (r["s_ref"] < r["s_true"][None, :]).mean(0)
                feas = np.asarray(b["task_feasible"]).astype(bool)
                logging.info(
                    f"[{task}] first traj: T={T} ep={r['episode_id']} refs={len(refs)} | "
                    f"s_true[{s_true.min():.2f},{s_true.max():.2f}] "
                    f"s_ref[{s_ref.min():.2f},{s_ref.max():.2f}] | "
                    f"mean rank feasible={rank_s[feas].mean():.3f} "
                    f"injected={rank_s[~feas].mean() if (~feas).any() else float('nan'):.3f}"
                )
            n_ep += 1

        assert n_ep > 0, f"task {task}: no trajectories"
        logging.info(f"[{task}] {n_ep} trajectories scored under 1+{len(refs)} prompts")

    dump = {
        "model_dir": run_dir, "step": FLAGS.step, "split": FLAGS.split,
        "data_split": FLAGS.data_split, "alpha": FLAGS.alpha, "tasks": tasks,
        "member_ids": member_ids, "n_ref_prompts": FLAGS.n_ref_prompts,
        "cross_task_refs": FLAGS.cross_task_refs, "prompt_seed": FLAGS.prompt_seed,
        "trajectories": records,
    }
    with open(out_path, "wb") as f:
        pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"[cps] wrote {len(records)} trajectories -> {out_path} "
                 f"({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    app.run(main)
