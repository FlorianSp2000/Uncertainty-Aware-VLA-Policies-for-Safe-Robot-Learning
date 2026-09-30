"""Score a Bridge/Fractal model on paired perturbations of the same trajectory.

For every trajectory of a pkl (first + last frame, action, instruction) the model is run
under a list of conditions that change exactly one input and keep the rest byte-identical:

  orig               the dataset frame, instruction and action
  lang_swap_train    instruction replaced by one drawn from the training prompt pool (the pool
                     the negative demonstrations draw from: a string the critic has seen, wrong
                     for this scene). The trajectory's own instruction is excluded from the draw.
  lang_swap_pkl      instruction replaced by another trajectory's instruction from the same pkl
                     (diagnostic control: a held-out string, not necessarily familiar)
  lang_swap_near     the training-stream instruction with the highest MUSE cosine to the own one
                     among those that substitute a content word and have cosine <= --near_max_cosine
                     (rule: experiments/utils/perturbations.py:near_swaps). Cosine stored per trajectory.
  inpaint            the reviewed inpainted frames (only pkls that carry them)
  jpeg_orig          original frames JPEG re-encoded at the quality whose median per-frame |delta|
                     matches the inpaintings' (chosen on this pkl, stored in metadata)
  composite_local_rR  original outside the edit mask, inpainted inside. Mask: max-channel
                     |inpainted - original| > --mask_threshold, dilated by a disk of R px
  composite_global_rR inpainted outside the mask, original inside
  act_*              action perturbations (ensemble only; see perturb_action)

plus, for the ensemble, an action sweep over --sweep_dim for the first --n_sweep trajectories.

Model families (--model_type): `ensemble` writes q/<cond> (n, members, heads, frames);
`classifier` (ResNet+MUSE) and `octo` (Octo + feasibility head) write p/<cond> in the same
layout with members = heads = 1 holding P(infeasible). --swaps_from takes the swapped strings
from an earlier dump so every model scores identical inputs. --dump_features adds
feat/<cond> (n, members, frames, D): the critic encoder output (image features after FiLM).
Run by slurm/eval-perturbation-sensitivity.sbatch; read by
src/scripts/analyse_perturbation_thesis.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from datetime import datetime

import h5py
import jax
import numpy as np
from absl import app, flags, logging

from experiments.utils import perturbations as P

FLAGS = flags.FLAGS
flags.DEFINE_enum("model_type", None, ["ensemble", "classifier", "octo"], "model family")
flags.DEFINE_string("resume_path", None, "ensemble dir (ensemble_member_k inside) | classifier checkpoint | Octo run dir")
flags.DEFINE_integer("step", None, "checkpoint step; None = latest")
flags.DEFINE_string("classifier_wandb", None, "W&B run path holding the classifier's config")
flags.DEFINE_integer("ensemble_size", 8, "")
flags.DEFINE_integer("batch_size", 1024, "training batch size of the run (config consistency only)")
flags.DEFINE_string("pkl", None, "trajectory pkl (first/last frame, action, language)")
flags.DEFINE_string("out", None, "output HDF5")
flags.DEFINE_string("data_dir", "/V-GPS/datasets/open_x", "")
flags.DEFINE_list("conditions", ["orig", "lang_swap_train", "lang_swap_pkl", "lang_swap_near", "inpaint", "jpeg_orig",
                                 "composite_local_r8", "composite_local_r16", "composite_local_r32",
                                 "composite_global_r8", "composite_global_r16", "composite_global_r32",
                                 "act_other", "act_far", "act_zero", "act_neg", "act_gripper", "act_scaled",
                                 "act_uniform"], "")
flags.DEFINE_string("swaps_from", None, "earlier dump of the same pkl whose swap_language/* strings are reused")
flags.DEFINE_bool("dump_features", False, "write feat/<cond> (ensemble only)")
flags.DEFINE_integer("n_traj", 0, "0 = all trajectories")
flags.DEFINE_integer("n_verbose", 0, "trajectories logged in full (smoke tests)")
flags.DEFINE_integer("n_sweep", 12, "trajectories used for the action sweep")
flags.DEFINE_integer("sweep_dim", 0, "action dimension swept (0 = x translation)")
flags.DEFINE_list("sweep_values", [str(v) for v in np.linspace(-1.0, 1.0, 11)], "")
flags.DEFINE_integer("prompt_pool_size", 30_000, "training-stream frames scanned for the lang_swap_train pool")
flags.DEFINE_integer("near_pool_frames", 500_000, "training-stream frames scanned for the lang_swap_near pool")
flags.DEFINE_float("near_max_cosine", P.NEAR_MAX_COSINE, "near swaps above this MUSE cosine count as paraphrases")
flags.DEFINE_list("jpeg_qualities", ["100", "98", "95", "90", "85", "80", "75", "70", "60", "50", "40", "30", "20", "10"], "")
flags.DEFINE_integer("mask_threshold", 30, "grey levels; max-channel |inpainted - original| above this is 'edited'")
flags.DEFINE_string("qualitative_dir", None, "if set: contact sheet PNG + near-swap pairs txt written here")
flags.DEFINE_integer("n_far_candidates", 4000, "uniform candidates scored for act_far")
flags.DEFINE_integer("seed", 0, "seed for the swap / action draws")
flags.mark_flags_as_required(["model_type", "resume_path", "pkl", "out"])

GRIPPER_DIM = 6
ACTION_DIM = 7


def perturb_action(name: str, a: np.ndarray, other: np.ndarray, far: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """a, other, far: (frames, ACTION_DIM). Returns the perturbed action for `name`."""
    a = a.copy()
    if name == "act_other":
        return other.copy()
    if name == "act_far":
        return far.copy()
    if name == "act_zero":
        return np.zeros_like(a)
    if name == "act_neg":
        a[:, :GRIPPER_DIM] *= -1.0
        return a
    if name == "act_gripper":
        a[:, GRIPPER_DIM] = 1.0 - a[:, GRIPPER_DIM]  # bridge 0/1, fractal 0/1 after rel2abs
        return a
    if name == "act_scaled":
        a[:, :GRIPPER_DIM] *= 3.0
        return a
    if name == "act_uniform":
        a[:, :GRIPPER_DIM] = rng.uniform(-1.0, 1.0, size=a[:, :GRIPPER_DIM].shape)
        a[:, GRIPPER_DIM] = rng.integers(0, 2, size=a.shape[0]).astype(np.float32)
        return a
    raise ValueError(f"unknown action condition {name}")


def far_actions(demo: np.ndarray, n_candidates: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """In-bounds actions far from the demonstration set.

    demo: (N, ACTION_DIM) every action of every trajectory of one source dataset. Candidates
    are uniform in [-1, 1]^6 with a random gripper bit; distance is the Euclidean nearest-
    neighbour distance on the first six dims. Returns the candidates sorted by distance
    (farthest first) and the percentile of the farthest against the demo set's own
    leave-one-out nearest-neighbour distances, so the reader knows how far "far" is.
    """
    if demo.ndim != 2 or demo.shape[1] != ACTION_DIM:
        raise ValueError(f"demo actions must be (N, {ACTION_DIM}), got {demo.shape}")
    d6 = demo[:, :GRIPPER_DIM]
    cand = rng.uniform(-1.0, 1.0, size=(n_candidates, GRIPPER_DIM)).astype(np.float32)
    nn = np.array([np.min(np.linalg.norm(d6 - c, axis=1)) for c in cand])
    order = np.argsort(-nn)
    sub = d6[rng.choice(len(d6), size=min(3000, len(d6)), replace=False)]
    own = np.array([np.sort(np.linalg.norm(sub - x, axis=1))[1] for x in sub])  # leave-one-out
    stats = {"farthest_nn_distance": float(nn[order[0]]),
             "demo_loo_nn_distance_median": float(np.median(own)),
             "demo_loo_nn_distance_p99": float(np.percentile(own, 99)),
             "fraction_of_demo_loo_distances_below_farthest": float((own < nn[order[0]]).mean()),
             "n_demo_actions": int(len(d6))}
    out = np.zeros((n_candidates, ACTION_DIM), np.float32)
    out[:, :GRIPPER_DIM] = cand[order]
    out[:, GRIPPER_DIM] = rng.integers(0, 2, size=n_candidates).astype(np.float32)
    return out, stats


def training_prompt_pools(train_config, data_config, sizes: list[int]) -> list[list[str]]:
    """Unique instructions of the first `size` frames of the training stream, one pool per size
    (one pass). The first-30k pool is the one the negative demonstrations draw from."""
    from octo.data.dataset import make_interleaved_dataset
    from octo.data.oxe import make_oxe_dataset_kwargs_and_weights

    cfg = data_config.to_dict()
    kw = dict(cfg.pop("oxe_kwargs"))
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(**kw)
    ds = make_interleaved_dataset(
        dataset_kwargs_list=dataset_kwargs_list, sample_weights=sample_weights, train=True,
        shuffle_buffer_size=1000, batch_size=None, balance_weights=True,
        traj_transform_kwargs={}, frame_transform_kwargs={},
        exclude_empty_lang_instr=train_config.exclude_empty_lang_instr, seed=train_config.seed,
    )
    # frame-level stream: each instruction is a scalar byte string
    pools, pool, t0 = {}, set(), time.time()
    for k, example in enumerate(ds.take(max(sizes))):
        lang = example["task"]["language_instruction"].numpy()
        if isinstance(lang, np.ndarray):
            lang = lang.reshape(-1)[0]
        lang = lang.decode("utf-8") if isinstance(lang, bytes) else str(lang)
        if lang:
            pool.add(lang)
        if k + 1 in sizes:
            pools[k + 1] = sorted(pool)
            logging.info("prompt pool after %d frames: %d unique (%.0fs)", k + 1, len(pool), time.time() - t0)
    for s in sizes:
        if not pools.get(s):
            raise ValueError(f"training prompt pool of {s} frames is empty or the stream ended early")
    return [pools[s] for s in sizes]


def load_trajectories(path: str, n_traj: int) -> tuple[list[dict], bool]:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    trajs = data["trajectories"] if isinstance(data, dict) else data
    has_inpaint = any("first_image_inpainted" in t for t in trajs)
    if has_inpaint:
        failed = {(f["dataset"], f["trajectory_id"]) for f in data["failed_ids"]} if isinstance(data, dict) and "failed_ids" in data else set()
        trajs = [t for t in trajs if t.get("first_image_inpainted") is not None
                 and t.get("last_image_inpainted") is not None
                 and (t["dataset"], t["trajectory_id"]) not in failed]
    return (trajs[:n_traj] if n_traj else trajs), has_inpaint


def reused_swaps(path: str, trajs: list[dict], conditions: list[str]) -> tuple[dict, dict]:
    """swap_language/<c> strings + near cosines from an earlier dump of the same trajectories."""
    with h5py.File(path) as f:
        if [x for x in f["trajectory_ids"][()]] != [t["trajectory_id"] for t in trajs] or \
                [s.decode() for s in f["language"][()]] != [t["language"] for t in trajs]:
            raise ValueError(f"{path} was written for different trajectories")
        swaps = {c: [s.decode() for s in f[f"swap_language/{c}"][()]] for c in conditions if c.startswith("lang_swap")}
        extra = {"own_instruction_in_training_pool": f["own_instruction_in_training_pool"][()],
                 "near_swap_cosine": f["near_swap_cosine"][()] if "near_swap_cosine" in f else None,
                 "meta": json.loads(f.attrs["metadata"])}
    return swaps, extra


def load_scorer(text_processor_holder: dict):
    """Returns score(images, actions, language) -> (out (members, heads, F), features | None)."""
    if FLAGS.model_type == "ensemble":
        from experiments.utils.model_utils import load_ensemble_agent
        from experiments.utils.ood_utils import load_configs_for_jupyter
        from jaxrl_m.data.text_processing import text_processors
        train_config, data_config = load_configs_for_jupyter(
            algorithm="ensemble_sarsa", data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size,
            ensemble_size=FLAGS.ensemble_size, seed=44, resume_path=FLAGS.resume_path)
        train_config.agent_kwargs.sow_embeddings = FLAGS.dump_features
        text_processor_holder["configs"] = (train_config.copy_and_resolve_references(), data_config.copy_and_resolve_references())
        cfg = train_config
        cfg.config = train_config.to_dict()
        cfg.oxedata_config = data_config
        tp = text_processors[cfg.text_processor](**cfg.text_processor_kwargs)
        text_processor_holder["muse"] = tp
        agents = load_ensemble_agent(resume_path=FLAGS.resume_path, config=cfg, data_dir=FLAGS.data_dir, step=FLAGS.step)[0]
        return P.ensemble_scorer(agents, FLAGS.ensemble_size, tp, FLAGS.dump_features)
    if FLAGS.model_type == "classifier":
        from experiments.utils.model_utils import load_classifier_agent
        if not FLAGS.classifier_wandb:
            raise ValueError("--classifier_wandb is required for the classifier")
        agent, tp = load_classifier_agent(FLAGS.resume_path, wandb_run_name=FLAGS.classifier_wandb)
        logging.info("classifier restored at step %s", int(agent.state.step))
        text_processor_holder["muse"] = tp
        return P.classifier_scorer(agent, tp)
    from experiments.utils.model_utils import load_octo_model
    model, _ = load_octo_model(FLAGS.resume_path, step=FLAGS.step)
    return P.octo_scorer(model)


def main(_):
    rng = np.random.default_rng(FLAGS.seed)
    trajs, has_inpaint = load_trajectories(FLAGS.pkl, FLAGS.n_traj)
    conditions = [c for c in FLAGS.conditions if has_inpaint or not (c == "inpaint" or P.is_image_condition(c))]
    is_ensemble = FLAGS.model_type == "ensemble"
    if not is_ensemble and any(c.startswith("act_") for c in conditions):
        raise ValueError("classifiers take no action input; drop the act_* conditions")
    if FLAGS.dump_features and not is_ensemble:
        raise ValueError("--dump_features is implemented for the ensemble critic only")
    lang_conds = [c for c in conditions if c.startswith("lang_swap")]
    if FLAGS.model_type == "octo" and lang_conds and not FLAGS.swaps_from:
        raise ValueError("Octo has no MUSE encoder; pass --swaps_from to reuse the ensemble's swaps")
    n = len(trajs)
    logging.info("%d trajectories, conditions %s", n, conditions)

    holder = {}
    score = load_scorer(holder)

    swap_lang = {c: [""] * n for c in lang_conds}
    near_cos = np.full(n, np.nan, np.float32)
    pool, near_pool, swap_meta = [], [], None
    if FLAGS.swaps_from:
        reused, extra = reused_swaps(FLAGS.swaps_from, trajs, lang_conds)
        swap_lang.update(reused)
        own_in_pool = extra["own_instruction_in_training_pool"]
        if "lang_swap_near" in lang_conds:
            near_cos = extra["near_swap_cosine"]
        swap_meta = extra["meta"]
        logging.info("reusing swaps from %s", FLAGS.swaps_from)
    else:
        need_train, need_near = "lang_swap_train" in lang_conds, "lang_swap_near" in lang_conds
        if need_train or need_near:
            if not is_ensemble:
                raise ValueError("training prompt pools are built in the ensemble run; pass --swaps_from")
            pool, near_pool = training_prompt_pools(*holder["configs"], [FLAGS.prompt_pool_size, FLAGS.near_pool_frames])
        own_in_pool = np.array([t["language"] in set(pool) for t in trajs]) if pool else np.zeros(n, bool)
        if need_near:
            near, near_cos = P.near_swaps([t["language"] for t in trajs], near_pool, holder["muse"], FLAGS.near_max_cosine)
            swap_lang["lang_swap_near"] = near
    logging.info("training prompt pool: %d, near-swap pool: %d unique instructions", len(pool), len(near_pool))

    t0 = time.time()
    images, masks, image_stats = P.image_conditions(trajs, conditions, [int(q) for q in FLAGS.jpeg_qualities], FLAGS.mask_threshold)
    logging.info("image conditions built in %.0fs: %s", time.time() - t0,
                 {k: v for k, v in image_stats.items() if not k.endswith("per_frame")})

    by_dataset: dict[str, list[int]] = {}
    for i, t in enumerate(trajs):
        by_dataset.setdefault(t["dataset"], []).append(i)
    far_by_dataset, far_stats = {}, {}
    if "act_far" in conditions:
        for name, idx in by_dataset.items():
            demo = np.concatenate([np.asarray(trajs[i]["action"], np.float32) for i in idx if "action" in trajs[i]])
            far_by_dataset[name], far_stats[name] = far_actions(demo, FLAGS.n_far_candidates, rng)
            logging.info("act_far %s: %s", name, far_stats[name])

    members, heads = (FLAGS.ensemble_size, 2) if is_ensemble else (1, 1)
    out_store = {c: np.zeros((n, members, heads, 2), np.float32) for c in conditions}
    feat_store = {}
    act_store = {c: np.zeros((n, 2, ACTION_DIM), np.float32) for c in conditions if c.startswith("act_")}
    far_rank = np.zeros(n, np.int32)
    delta_store = {c: np.zeros((n, 2), np.float32) for c in conditions if c == "inpaint" or P.is_image_condition(c)}

    t0 = time.time()
    for i, t in enumerate(trajs):
        orig_images = np.stack([t["first_image"], t["last_image"]])
        actions = np.stack([t["first_action"], t["last_action"]]).astype(np.float32)
        lang = [t["language"], t["language"]]
        others = [j for j in by_dataset[t["dataset"]] if j != i]
        j_other = int(rng.choice(others))
        other_actions = np.stack([trajs[j_other]["first_action"], trajs[j_other]["last_action"]]).astype(np.float32)
        far = None
        if "act_far" in conditions:
            # rotate through the farthest candidates so the condition is not one single vector
            far_rank[i] = i % min(50, FLAGS.n_far_candidates)
            far = np.stack([far_by_dataset[t["dataset"]][far_rank[i]]] * 2)
        for c in conditions:
            img, a, l = orig_images, actions, lang
            if c == "lang_swap_train" and not FLAGS.swaps_from:
                s = t["language"]
                while s == t["language"]:
                    s = pool[int(rng.integers(len(pool)))]
                swap_lang[c][i] = s
            elif c == "lang_swap_pkl" and not FLAGS.swaps_from:
                swap_lang[c][i] = trajs[int(rng.choice(others))]["language"]
            if c.startswith("lang_swap"):
                l = [swap_lang[c][i]] * 2
            elif c == "inpaint":
                img = np.stack([t["first_image_inpainted"], t["last_image_inpainted"]])
            elif P.is_image_condition(c):
                img = images[c][i]
            elif c.startswith("act_"):
                a = perturb_action(c, actions, other_actions, far, rng)
                act_store[c][i] = a
            elif c != "orig":
                raise ValueError(f"unknown condition {c}")
            if c in delta_store:
                delta_store[c][i] = [P.median_abs_delta(orig_images[k], img[k]) for k in range(2)]
            o, feat = score(img, a, l)
            out_store[c][i] = o
            if feat is not None and not c.startswith("act_"):
                if c not in feat_store:
                    feat_store[c] = np.zeros((n, members, 2, feat.shape[-1]), np.float32)
                feat_store[c][i] = feat
            if i < FLAGS.n_verbose:
                logging.info("[verbose] traj %d cond %-22s lang=%r img_md5=%s median|d|=%s out(member means)=%s feat=%s",
                             i, c, l[0], hashlib.md5(img.tobytes()).hexdigest()[:8],
                             delta_store[c][i].tolist() if c in delta_store else "-",
                             np.round(o.mean(axis=(1, 2)), 3).tolist(),
                             None if feat is None else (feat.shape, float(np.linalg.norm(feat[0, 0]))))
        if i % 25 == 0:
            logging.info("trajectory %d/%d (%.0fs)", i, n, time.time() - t0)

    sweep = None
    if is_ensemble and FLAGS.n_sweep:
        values = np.array([float(v) for v in FLAGS.sweep_values], np.float32)
        n_sweep = min(FLAGS.n_sweep, n)
        sweep_q = np.zeros((n_sweep, len(values), members, heads, 2), np.float32)
        for i in range(n_sweep):
            t = trajs[i]
            img = np.stack([t["first_image"], t["last_image"]])
            for v_idx, v in enumerate(values):
                a = np.stack([t["first_action"], t["last_action"]]).astype(np.float32)
                a[:, FLAGS.sweep_dim] = v
                sweep_q[i, v_idx] = score(img, a, [t["language"], t["language"]])[0]
        sweep = (sweep_q, values, n_sweep)

    if FLAGS.qualitative_dir:
        os.makedirs(FLAGS.qualitative_dir, exist_ok=True)
        if masks:
            r = 16 if 16 in masks else sorted(masks)[0]
            P.contact_sheet(trajs, images, masks, r, os.path.join(FLAGS.qualitative_dir, f"image_conditions_contact_sheet_r{r}.png"))
        if "lang_swap_near" in swap_lang:
            idx = np.linspace(0, n - 1, min(20, n)).astype(int)
            with open(os.path.join(FLAGS.qualitative_dir, "near_swap_pairs.txt"), "w") as fh:
                fh.write(f"# {os.path.basename(FLAGS.pkl)}: original instruction -> lang_swap_near (MUSE cosine); "
                         f"rule: highest cosine among content-word substitutions with cosine <= {FLAGS.near_max_cosine}\n")
                for i in idx:
                    fh.write(f"{near_cos[i]:.3f}\t{trajs[i]['language']!r} -> {swap_lang['lang_swap_near'][i]!r}\n")

    pkl_langs = {t["language"] for t in trajs}
    meta = {
        "model_type": FLAGS.model_type, "model_path": FLAGS.resume_path, "step": FLAGS.step,
        "classifier_wandb": FLAGS.classifier_wandb, "ensemble_size": members,
        "pkl": FLAGS.pkl, "n_trajectories": n, "conditions": conditions, "seed": FLAGS.seed,
        "swaps_from": FLAGS.swaps_from,
        "prompt_pool_size": swap_meta["prompt_pool_size"] if swap_meta else len(pool),
        "near_pool_size": swap_meta.get("near_pool_size") if swap_meta else len(near_pool),
        "near_max_cosine": FLAGS.near_max_cosine,
        "fraction_of_pkl_instructions_in_training_pool": float(own_in_pool.mean()),
        "fraction_of_pkl_unique_instructions_in_training_pool": float(np.mean([s in set(pool) for s in pkl_langs])) if pool else None,
        "image_stats": {k: v for k, v in image_stats.items() if not k.endswith("per_frame")},
        "act_far_stats": far_stats,
        "sweep_dim": FLAGS.sweep_dim, "n_sweep": sweep[2] if sweep else 0,
        "output": ("q/<cond>: (n, member, head, frame), frame = (first, last), head = the two inner Q-heads" if is_ensemble
                   else "p/<cond>: (n, 1, 1, frame) = P(infeasible), frame = (first, last)"),
        "features": "feat/<cond>: (n, member, frame, D) critic encoder output (ResNet+FiLM, before the Q MLP)" if feat_store else None,
        "rendered": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(FLAGS.out) or ".", exist_ok=True)
    with h5py.File(FLAGS.out, "w") as f:
        f.attrs["metadata"] = json.dumps(meta)
        f.create_dataset("trajectory_ids", data=np.array([t["trajectory_id"] for t in trajs]))
        f.create_dataset("dataset", data=np.array([t["dataset"] for t in trajs], dtype="S"))
        f.create_dataset("language", data=np.array([t["language"] for t in trajs], dtype=h5py.string_dtype()))
        f.create_dataset("own_instruction_in_training_pool", data=own_in_pool)
        f.create_dataset("actions/orig", data=np.stack([np.stack([t["first_action"], t["last_action"]]) for t in trajs]).astype(np.float32))
        key = "q" if is_ensemble else "p"
        for c, o in out_store.items():
            f.create_dataset(f"{key}/{c}", data=o)
        for c, x in feat_store.items():
            f.create_dataset(f"feat/{c}", data=x)
        for c, s in swap_lang.items():
            f.create_dataset(f"swap_language/{c}", data=np.array(s, dtype=h5py.string_dtype()))
        if "lang_swap_near" in swap_lang:
            f.create_dataset("near_swap_cosine", data=near_cos)
        for c, a in act_store.items():
            f.create_dataset(f"actions/{c}", data=a)
        for c, d in delta_store.items():
            f.create_dataset(f"image_median_abs_delta/{c}", data=d)
        for r, m in masks.items():
            f.create_dataset(f"mask_area_fraction/r{r}", data=m.reshape(n, 2, -1).mean(axis=-1).astype(np.float32))
        if "act_far" in conditions:
            f.create_dataset("act_far_rank", data=far_rank)
        if sweep:
            f.create_dataset("sweep/q", data=sweep[0])
            f.create_dataset("sweep/values", data=sweep[1])
            f.create_dataset("sweep/traj_index", data=np.arange(sweep[2]))
    logging.info("wrote %s", FLAGS.out)


if __name__ == "__main__":
    app.run(main)
