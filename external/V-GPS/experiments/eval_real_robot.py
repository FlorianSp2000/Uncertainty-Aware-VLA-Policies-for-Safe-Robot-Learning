"""Score the episodes of a rollout cache (thesis: SAFE WidowX / LIBERO, pools calib + test) with a finetuned SARSA ensemble.

(The loader also supports a real-robot demonstration dataset from preliminary experiments that are not part of the thesis.)

Walks the cache written by ``convert_real_robot.py`` and, for every frame, evaluates each
ensemble member on the tuple (external image, prompt in force at that frame, action the policy
actually emitted there). Writes one pickle of per-episode records; the conformal thresholding
and the operating-point tables are host-side work on that dump.

The three groups the dump has to keep apart, because a rollout that fails and a rollout whose
instruction is impossible are different things:

* feasible + success  -- calibration and false-alarm rate
* feasible + failure  -- false-alarm rate too; the control that separates an infeasibility
  detector from a plain failure detector, since every infeasible episode here also failed
* infeasible          -- recall (conditions ``missing`` and ``half_missing``)

Run via slurm/eval-real-robot.sbatch.
"""
import json
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from flax.training import checkpoints

from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders

# Reuses the training script's helper and its --config/--name/--project flag
# definitions (it has a __main__ guard, so importing is side-effect safe).
from experiments.train_real_robot_ensemble import create_ensemble_agents
from experiments.utils.real_robot_actions import (
    normalize_array, resolve_normalization, resolve_statistics)

FLAGS = flags.FLAGS

flags.DEFINE_string("run_dir", None, "Ensemble run directory holding ensemble_member_*.")
flags.DEFINE_integer("step", None, "Checkpoint step to restore.")
flags.DEFINE_string("data_dir", None, "Cache directory from convert_real_robot.py.")
flags.DEFINE_string("out", None, "Output pickle path.")
flags.DEFINE_list("pools", ["rollout"], "Which pools to score: rollout, scene, teleop.")
flags.DEFINE_integer("image_size", 256, "Resize frames to this before scoring.")
flags.DEFINE_integer("max_episodes", 0, "Score at most this many episodes (0 = all).")
flags.DEFINE_bool("swap_instruction", False,
                  "Positive control: score every episode under another task's instruction "
                  "(the next task in sorted order), so the critic's reading of language can be "
                  "checked on episodes whose own instruction is satisfiable.")


def param_l2_norm(agent):
    leaves = jax.tree_util.tree_leaves(agent.state.params)
    return float(np.sqrt(sum(float(np.sum(np.asarray(x) ** 2)) for x in leaves)))


def decode_frames(png_bytes, size):
    """PNG bytes -> (T, size, size, 3) uint8, matching dlimp's bilinear resize_image."""
    frames = []
    for blob in png_bytes:
        img = tf.io.decode_png(bytes(blob), channels=3)
        img = tf.image.resize(tf.cast(img, tf.float32), [size, size], method="bilinear")
        frames.append(tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy())
    return np.stack(frames)


def main(_):
    tf.config.set_visible_devices([], "GPU")

    data_dir = Path(FLAGS.data_dir)
    index = json.loads((data_dir / "index.json").read_text())
    rows = [r for r in index if r["pool"] in FLAGS.pools]
    rows.sort(key=lambda r: (r["pool"], r["condition"], r["key"]))
    if FLAGS.max_episodes:
        rows = rows[: FLAGS.max_episodes]
    if not rows:
        raise ValueError(f"no episodes for pools={FLAGS.pools} in {data_dir}/index.json")
    logging.info(f"scoring {len(rows)} episodes from pools {FLAGS.pools}")

    # The same call the training loader makes, so it resolves to the same cached JSON: the two
    # agree because they ask the same question, not because two call sites were kept in step.
    normalization_type = resolve_normalization(FLAGS.config.real_robot_normalization)
    stats = None
    if normalization_type is not None:
        stats = resolve_statistics(FLAGS.data_dir, FLAGS.config.real_robot_frame_stride,
                                   FLAGS.config.real_robot_statistics_path,
                                   FLAGS.config.real_robot_train_pool,
                                   FLAGS.config.real_robot_causal_gripper)
    else:
        logging.warning("eval_real_robot: action normalization is OFF")

    text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    agent_class = agents[FLAGS.config.agent]
    ensemble_size = FLAGS.config.ensemble_size

    def prepare_actions(raw_actions: np.ndarray, key: str) -> np.ndarray:
        """Map an episode's actions into the space the critic was trained on."""
        actions = raw_actions.astype(np.float32)
        if stats is None:
            return actions
        actions, clipped = normalize_array(actions, stats, normalization_type,
                                           causal_gripper=FLAGS.config.real_robot_causal_gripper)
        if clipped.max() > 0.05:
            # The bounds come from the demonstrations. A policy acting outside them saturates
            # here instead of reaching the critic, which would silently flatten the very
            # actions most likely to be anomalous.
            logging.warning(f"{key}: clipped by pose dimension {np.round(clipped, 3).tolist()}")
        return actions

    # Template from the first episode: create_ensemble_agents needs a batch of the right shapes.
    first = np.load(data_dir / rows[0]["pool"] / f"{rows[0]['key']}.npz", allow_pickle=True)
    tmpl_images = decode_frames(first["images"][:2], FLAGS.image_size)
    tmpl_batch = {
        "observations": {"image": tmpl_images},
        "actions": prepare_actions(first["actions"][:2], rows[0]["key"]),
        "rewards": np.full(2, -1.0, dtype=np.float32),
        "masks": np.ones(2, dtype=np.float32),
        "goals": {"language": text_processor.encode([str(first["prompts"][0])] * 2)},
    }
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, tmpl_batch, encoder_def, agent_class, FLAGS.config.agent_kwargs
    )
    template_member = jax.tree_map(lambda x: x[0], ensemble_agents)
    template_norm = param_l2_norm(template_member)

    members, norms = [], {}
    for k in range(ensemble_size):
        path = os.path.join(FLAGS.run_dir, f"ensemble_member_{k}")
        member = checkpoints.restore_checkpoint(path, target=template_member, step=FLAGS.step)
        norm = param_l2_norm(member)
        # restore_checkpoint silently returns the target when it finds nothing to load.
        assert norm != template_norm, f"member {k}: restore returned random-init params"
        norms[k] = norm
        members.append(member)
        logging.info(f"[restore] member {k} @ step {FLAGS.step}: param norm {norm:.4f}")
    assert len(set(norms.values())) == ensemble_size, \
        f"duplicate member norms {norms}: members are not distinct, disagreement would be vacuous"

    seeds = jax.random.split(jax.random.PRNGKey(FLAGS.config.seed), ensemble_size)

    @jax.jit
    def member_q(member, observations, actions, language, rng):
        """The ``q`` entry of ``get_eval_values`` (min over the member's inner critic heads),
        compiled once and without the target and OOD passes that method also runs -- those
        made a 50-frame episode cost ~10 s."""
        obs = ((observations, {"language": language}) if member.config["goal_conditioned"]
               else observations)
        return jnp.min(member.forward_critic(obs, actions, rng, train=False), axis=0)

    swap_to = None
    if FLAGS.swap_instruction:
        by_task = {}
        for r in index:
            by_task.setdefault(r["condition"], set()).add(r["instruction"])
        if any(len(v) != 1 for v in by_task.values()):
            raise ValueError(f"swap needs one instruction per task, got {by_task}")
        tasks = sorted(by_task)
        swap_to = {t: next(iter(by_task[tasks[(i + 1) % len(tasks)]])) for i, t in enumerate(tasks)}
        logging.info(f"swapped instructions: {swap_to}")

    records = []
    for n, row in enumerate(rows, 1):
        payload = np.load(data_dir / row["pool"] / f"{row['key']}.npz", allow_pickle=True)
        images = decode_frames(payload["images"], FLAGS.image_size)
        actions = prepare_actions(payload["actions"], row["key"])
        prompts = [str(p) for p in payload["prompts"]]
        if swap_to is not None:
            prompts = [swap_to[row["condition"]]] * len(prompts)
        T = len(actions)
        if not (images.shape[0] == T == len(prompts)):
            raise ValueError(f"{row['key']}: {images.shape[0]} frames, {T} actions, {len(prompts)} prompts")

        # Per-frame embedding: the prompt changes mid-episode in the exchange, half_missing
        # and two of the missing rollouts.
        language = text_processor.encode(prompts)
        if language.shape[0] != T:
            raise ValueError(f"{row['key']}: language {language.shape} has no leading T={T} axis")

        model_batch = {
            "observations": {"image": images},
            "actions": actions,
            "rewards": np.full(T, -1.0, dtype=np.float32),
            "masks": np.ones(T, dtype=np.float32),
        }
        q_members = np.stack(
            [np.asarray(member_q(m, model_batch["observations"], model_batch["actions"],
                                 language, seeds[k]))
             for k, m in enumerate(members)],
            axis=0,
        )
        if q_members.shape != (ensemble_size, T):
            raise ValueError(f"{row['key']}: q_members {q_members.shape} != {(ensemble_size, T)}")
        if not np.isfinite(q_members).all():
            raise ValueError(f"{row['key']}: non-finite Q")

        records.append({
            "key": row["key"], "pool": row["pool"], "condition": row["condition"],
            "source": row["source"], "length": T,
            "success": row["success"], "feasible": row["feasible"],
            "instruction": row["instruction"], "prompts": prompts,
            "prompt_switches": row["prompt_switches"],
            "q_members": q_members.astype(np.float32),
            "s_t": q_members.std(axis=0, ddof=1).astype(np.float32),
            "q_mean_t": q_members.mean(axis=0).astype(np.float32),
        })
        if n == 1 or n % 20 == 0:
            r = records[-1]
            logging.info(
                f"[{n}/{len(rows)}] {r['condition']:13s} success={r['success']} T={T} "
                f"s_t [{r['s_t'].min():.3f}, {r['s_t'].max():.3f}] "
                f"q_mean [{r['q_mean_t'].min():.2f}, {r['q_mean_t'].max():.2f}]"
            )

    out = Path(FLAGS.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"records": records, "run_dir": FLAGS.run_dir, "step": FLAGS.step,
                     "member_norms": norms, "image_size": FLAGS.image_size,
                     "swap_instruction": FLAGS.swap_instruction,
                     "causal_gripper": FLAGS.config.real_robot_causal_gripper,
                     # So a dump can never be mistaken for one produced under the other
                     # action treatment.
                     "normalization": str(normalization_type), "action_statistics": stats}, f)
    logging.info(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    app.run(main)
