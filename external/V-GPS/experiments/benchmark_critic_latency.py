"""Wall-clock latency of the Q-ensemble critic at deployment batch sizes.

One image + one instruction, B candidate actions (B = 1 or an action-chunk length), scored by
1 member or all members in one jitted call. The critic re-encodes the image for every row, so
B > 1 measures the implementation as it is, not an encoder-shared variant. MUSE text encoding
(CPU, TF hub) is timed on its own because a deployed policy encodes the instruction once per
episode. Run by slurm/benchmark-critic-latency.sbatch; writes one JSON.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import jax
import numpy as np
from absl import app, flags, logging

from experiments.utils.model_utils import load_ensemble_agent
from experiments.utils.ood_utils import load_configs_for_jupyter
from experiments.utils.perturbations import _members_q
from jaxrl_m.data.text_processing import text_processors

FLAGS = flags.FLAGS
flags.DEFINE_string("resume_path", None, "ensemble checkpoint directory")
flags.DEFINE_integer("ensemble_size", 8, "")
flags.DEFINE_integer("batch_size", 1024, "training batch size of the run (config consistency only)")
flags.DEFINE_string("data_dir", "/V-GPS/datasets/open_x", "")
flags.DEFINE_list("rows", ["1", "4", "16", "50"], "candidate actions per call")
flags.DEFINE_integer("warmup", 20, "")
flags.DEFINE_integer("n_calls", 200, "")
flags.DEFINE_string("out", None, "output JSON")
flags.mark_flags_as_required(["resume_path", "out"])


def timed(fn, n: int) -> np.ndarray:
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ms.append(1e3 * (time.perf_counter() - t))
    return np.array(ms)


def summary(ms: np.ndarray) -> dict:
    med = float(np.median(ms))
    return {"median_ms": med, "p90_ms": float(np.percentile(ms, 90)), "min_ms": float(ms.min()), "hz_at_median": 1e3 / med}


def main(_):
    train_config, data_config = load_configs_for_jupyter(
        algorithm="ensemble_sarsa", data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size,
        ensemble_size=FLAGS.ensemble_size, seed=44, resume_path=FLAGS.resume_path)
    cfg = train_config
    cfg.config = train_config.to_dict()
    cfg.oxedata_config = data_config
    tp = text_processors[cfg.text_processor](**cfg.text_processor_kwargs)
    agents = load_ensemble_agent(resume_path=FLAGS.resume_path, config=cfg, data_dir=FLAGS.data_dir)[0]

    instruction = "put the eggplant in the pot"
    for _ in range(3):
        tp.encode([instruction])
    muse = summary(timed(lambda: tp.encode([instruction]), FLAGS.n_calls))
    lang = tp.encode([instruction])[0]

    fwd = jax.jit(_members_q)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    results = []
    for members in (1, FLAGS.ensemble_size):
        sub = jax.tree_map(lambda x: x[:members], agents)
        rngs = jax.random.split(jax.random.PRNGKey(0), members)
        for b in (int(r) for r in FLAGS.rows):
            batch = jax.device_put({"actions": rng.uniform(-1, 1, (b, 7)).astype(np.float32),
                                    "observations": {"image": np.repeat(image[None], b, 0)},
                                    "goals": {"language": np.repeat(lang[None], b, 0)}})
            call = lambda: fwd(sub, batch, rngs)
            q = call()
            if q.shape != (members, 2, b):
                raise ValueError(f"unexpected output {q.shape}")
            timed(call, FLAGS.warmup)
            row = {"members": members, "rows": b, **summary(timed(call, FLAGS.n_calls))}
            logging.info("%s", row)
            results.append(row)

    dev = jax.devices()[0]
    stats = dev.memory_stats()
    out = {"model_path": FLAGS.resume_path, "device": dev.device_kind, "jax": jax.__version__,
           "peak_device_bytes_in_use": stats["peak_bytes_in_use"], "device_bytes_limit": stats["bytes_limit"],
           "n_calls": FLAGS.n_calls, "warmup": FLAGS.warmup, "image": "256x256x3 uint8, device-resident",
           "muse_encode_one_instruction_cpu": muse, "critic": results,
           "rendered": datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(FLAGS.out) or ".", exist_ok=True)
    with open(FLAGS.out, "w") as fh:
        json.dump(out, fh, indent=2)
    logging.info("wrote %s", FLAGS.out)


if __name__ == "__main__":
    app.run(main)
