"""Three-way prompt evaluation on IVA: true instruction / swapped task prompt / injection.

Every published IVA false-premise number scores a frame under an instruction that is a
NOVEL STRING: IVA builds an injection by keeping the task's own verb frame and replacing
the object noun from a fixed pool ("turn left tap" -> "turn left chicken"). None of the
2926 eval injections is a swapped task prompt. So no existing number separates

    "the ensemble noticed the instruction contradicts the scene"      (grounding)
from
    "the ensemble noticed the string is one it was never trained on"  (novelty).

This script holds the frame fixed and varies only the instruction string:

    given    what IVA shows at that frame -- the injection on injected frames
    true     the episode's own instruction, on every frame
    swapped  the train-modal instruction of each of the 8 OTHER tasks: a real string seen
             thousands of times in training, wrong for this scene

The stored record keys keep the earlier `foreign` spelling (`foreign_tasks`,
`foreign_instructions`, `q_members_foreign`); renaming them would break existing dumps.

The swapped task prompt is the discriminating stimulus. Under grounding it is as detectable
as the injection; under novelty it is invisible.

Scoring reuses eval_iva_cps.score_under_prompt, which is the same quantity eval_iva_cp
dumps (min over the inner critics of forward_critic, train=False), so `s_given` reproduces
that dump and the comparison is against the published pipeline, not a variant of it.

Run via slurm/eval-iva-three-way-prompt.sbatch.
"""

import os
import pickle
from itertools import islice

import jax
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from flax.training import checkpoints

from jaxrl_m.agents import agents
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors

from octo.data.iva import IVA_TASKS, instruction_inventory

# Defines the shared flags (--model_dir, --step, --split, --data_split, --tasks,
# --iva_data_dir, --out, --max_trajs) and the dataloading/restore helpers.
from experiments.eval_iva_cp import build_task_iterator, process_traj, param_l2_norm
from experiments.eval_iva_cps import score_under_prompt
from experiments.train_iva_ensemble import create_ensemble_agents, detect_checkpoint_action_dim

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "instruction_split", "train", ["train", "eval"],
    "Split whose modal instruction represents each task in the swapped-prompt condition. "
    "'train' is the sharp version of the control: the most frequent string the model was "
    "actually trained on, so a null result cannot be blamed on the string being rare.",
)

# The `given` and `true` conditions carry the same string on every feasible frame, so
# their scores must agree. They do not agree exactly: MUSE encodes a batch of sentences,
# and the batch the loader hands it (mixed, because injected frames carry a different
# string) does not reproduce a uniform batch of the same sentence bit-for-bit. The
# residual is ~0.1 % of a task's floor, three orders below the language effect the
# experiment measures. The bound is therefore relative to the episode's own floor and set
# where a genuine mis-wiring -- wrong string, wrong tiling, wrong member set -- would land,
# which is order 1, not order 1e-3. The realised gap is recorded per episode and reported.
_TWIN_TOL_REL = 0.02


def foreign_representatives(counts):
    """task -> its single most frequent true-premise instruction, ties broken by string."""
    reps = {}
    for task in IVA_TASKS:
        assert task in counts, f"no instructions found for task {task!r}"
        reps[task] = max(counts[task].items(), key=lambda kv: (kv[1], kv[0]))[0]
    assert len(set(reps.values())) == len(reps), f"two tasks share a representative: {reps}"
    return reps


def main(_):
    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed)

    run_dir = FLAGS.model_dir.rstrip("/")
    out_path = os.path.abspath(FLAGS.out)
    assert not out_path.startswith(os.path.abspath(run_dir)), "refusing to write into --model_dir"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    assert FLAGS.split == "fp", "the injection condition only exists on the fp split"

    tasks = list(FLAGS.tasks)
    rep_counts, _ = instruction_inventory(FLAGS.iva_data_dir, FLAGS.instruction_split)
    reps = foreign_representatives(rep_counts)
    _, episode_true = instruction_inventory(FLAGS.iva_data_dir, FLAGS.data_split, tasks)
    logging.info(f"[three-way-prompt] representatives from the {FLAGS.instruction_split} split:")
    for task in IVA_TASKS:
        logging.info(f"[three-way-prompt]   {task:30s} {rep_counts[task][reps[task]]:6d}x  {reps[task]!r}")
    logging.info(f"[three-way-prompt] true instruction resolved for {len(episode_true)} episodes")

    ensemble_size = FLAGS.config.ensemble_size
    member_ids = [
        k for k in range(ensemble_size)
        if tf.io.gfile.exists(os.path.join(run_dir, f"ensemble_member_{k}", f"checkpoint_{FLAGS.step}"))
    ]
    assert len(member_ids) >= 4, f"only {len(member_ids)} complete members at step {FLAGS.step}"
    assert detect_checkpoint_action_dim(
        os.path.join(run_dir, f"ensemble_member_{member_ids[0]}")) == 8, "checkpoint action dim != 8"

    text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)
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
    logging.info(f"[three-way-prompt] {len(members)} distinct members restored: {sorted(norms)}")

    # Encode `length` copies rather than encoding once and repeating, so a uniform-prompt
    # condition reaches the critic through the same batched MUSE call the loader uses for
    # the given prompts. Encoding one sentence alone and tiling it differs at 1e-7 and the
    # critic turns that into a twin gap of 3e-3; the first-episode log line measures both.
    embed_cache = {}

    def embed(instruction, length):
        key = (instruction, length)
        if key not in embed_cache:
            embed_cache[key] = np.asarray(text_processor.encode([instruction] * length))
        return embed_cache[key]

    def q_members(images, actions, lang_emb):
        """(n_members, T) per-member Q. Stored raw: the host side picks its own ddof for
        `std_Q`, and per-member traces are what a disagreement claim gets inspected against."""
        return np.stack([score_under_prompt(m, images, actions, lang_emb) for m in members], 0).astype(np.float32)

    records = []
    for task in tasks:
        foreign_tasks = [t for t in IVA_TASKS if t != task]
        foreign_instructions = [reps[t] for t in foreign_tasks]
        stream = (process_traj(t, text_processor) for t in build_task_iterator(task))
        if FLAGS.max_trajs:
            stream = islice(stream, FLAGS.max_trajs)

        n_ep = 0
        for b in stream:
            images, actions = b["observations"]["image"], b["actions"]
            feasible = np.asarray(b["task_feasible"]).astype(bool)
            T = feasible.shape[0]
            episode_id = int(b["meta"]["episode_id"][0])
            true_instruction = episode_true[(task, episode_id)]
            assert true_instruction not in foreign_instructions, (
                f"[{task}] ep{episode_id}: its own instruction {true_instruction!r} is also the "
                f"representative of another task -- the control would score a correct prompt "
                f"as a negative")
            given = b["goals"]["language_str"]
            assert all(given[i] == true_instruction for i in np.where(feasible)[0]), (
                f"[{task}] ep{episode_id}: a feasible frame carries an instruction other than the "
                f"episode's true one; the true-instruction map disagrees with the loader")

            qm_given = q_members(images, actions, b["goals"]["language"])
            qm_true = q_members(images, actions, embed(true_instruction, T))
            s_given, s_true = qm_given.std(axis=0), qm_true.std(axis=0)

            # Same string, same frame -> same score. The one internal check that the
            # uniform-prompt conditions reach the critic the way the given prompts do.
            floor = float(s_true[feasible].mean())
            twin_gap = float(np.abs(s_given[feasible] - s_true[feasible]).max()) if feasible.any() else 0.0
            assert twin_gap < _TWIN_TOL_REL * floor, (
                f"[{task}] ep{episode_id}: given and true disagree by {twin_gap:.2e} on feasible "
                f"frames where they are the same string ({100 * twin_gap / floor:.2f} % of this "
                f"episode's floor {floor:.3f})")

            qm_foreign = np.stack(
                [q_members(images, actions, embed(s, T)) for s in foreign_instructions], 0)
            s_foreign = qm_foreign.std(axis=1)

            records.append({
                "task": task,
                "task_id": b["meta"]["task_id"][0].decode("utf-8"),
                "episode_id": episode_id,
                "step_id": b["meta"]["step_id"],
                "length": T,
                "task_feasible": feasible,
                "inj_idx": np.where(~feasible)[0],
                "language": given,
                "true_instruction": true_instruction,
                "foreign_tasks": foreign_tasks,
                "foreign_instructions": foreign_instructions,
                "q_members_given": qm_given,
                "q_members_true": qm_true,
                "q_members_foreign": qm_foreign,
                "twin_gap": twin_gap,
            })
            if n_ep == 0 and task == tasks[0]:
                # Attribute the twin gap rather than tolerating it blind. All three arrays
                # below encode the SAME sentence; any non-zero difference is MUSE's batched
                # graph, not a wiring error, and the twin gap is what the critic makes of it.
                feas_rows = np.where(feasible)[0]
                mixed = np.asarray(b["goals"]["language"])
                uniform = embed(true_instruction, T)
                single = np.repeat(
                    np.asarray(text_processor.encode([true_instruction]))[0][None, :], T, axis=0)
                logging.info(
                    f"[three-way-prompt] MUSE encode of one sentence, three ways: "
                    f"|mixed batch - uniform batch| = {np.abs(mixed[feas_rows] - uniform[feas_rows]).max():.2e}, "
                    f"|uniform batch - single encode| = {np.abs(uniform - single).max():.2e}")
            if n_ep == 0:
                inj = ~feasible
                logging.info(
                    f"[{task}] first traj: T={T} ep={episode_id} true={true_instruction!r} "
                    f"twin_gap={twin_gap:.2e} | mean s_true(feasible)={s_true[feasible].mean():.3f} "
                    f"s_given(inj)={s_given[inj].mean() if inj.any() else float('nan'):.3f} "
                    f"s_foreign(inj)={s_foreign[:, inj].mean() if inj.any() else float('nan'):.3f} "
                    f"s_foreign(all)={s_foreign.mean():.3f}")
            n_ep += 1

        assert n_ep > 0, f"task {task}: no trajectories"
        logging.info(f"[{task}] {n_ep} trajectories scored under 2+{len(foreign_tasks)} prompts")

    total_inj = sum(len(r["inj_idx"]) for r in records)
    assert total_inj > 0, "no injected frames -- label plumbing is broken"
    logging.info(f"[three-way-prompt] worst twin gap over all episodes: {max(r['twin_gap'] for r in records):.2e}")

    dump = {
        "model_dir": run_dir, "step": FLAGS.step, "split": FLAGS.split,
        "data_split": FLAGS.data_split, "instruction_split": FLAGS.instruction_split,
        "tasks": tasks, "member_ids": member_ids, "representatives": reps,
        "trajectories": records,
    }
    with open(out_path, "wb") as f:
        pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"[three-way-prompt] wrote {len(records)} trajectories, {total_inj} injected frames "
                 f"-> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    app.run(main)
