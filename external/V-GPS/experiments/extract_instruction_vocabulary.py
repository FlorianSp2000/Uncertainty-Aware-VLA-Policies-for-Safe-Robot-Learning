"""Which language instructions does the critic see in training, and which only at evaluation?

Walks every Bridge/Fractal trajectory of the training and the validation split once, in file
order, through the same `make_dataset_from_rlds` call and dataset kwargs the training scripts use
(`make_oxe_dataset_kwargs_and_weights` on the `data_config.py` OXE config), with empty
instructions dropped as in training. No shuffle, no repeat, no frame flattening: one element is
one trajectory. `shuffle=False` and `num_parallel_reads=4` match `create_validation_dataset`,
which built the object-removal pkls, so the i-th trajectory here is `trajectory_id=i` there.

Artifact: one HDF5 with a group per `<dataset>/<split>` holding, per trajectory, the raw
instruction string, the trajectory length and the first (normalised) action -- the fingerprint
used to check the index mapping against the pkls. Instruction normalisation and overlap counts
are left to the local analysis. Run by slurm/extract-instruction-vocabulary.sbatch.
"""
from __future__ import annotations

import inspect
import time
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags, logging
from ml_collections import config_flags

FLAGS = flags.FLAGS
flags.DEFINE_string("out_path", None, "output HDF5 path")
flags.DEFINE_list("splits", ["train", "val"], "which splits to walk")
flags.DEFINE_integer("max_trajectories", 0, "0 = whole split; >0 = stop after N (smoke test)")
config_flags.DEFINE_config_file("oxedata_config", None, "data config", lock_config=False)
flags.mark_flags_as_required(["out_path", "oxedata_config"])


def _keep_fingerprint(traj):
    """Drop images before leaving the graph; keep what the walk records and checks."""
    return {
        "language": traj["task"]["language_instruction"],
        "timestep": traj["observation"]["timestep"],
        "first_action": traj["action"][0],
    }


def walk_split(dataset_kwargs: dict, train: bool):
    from octo.data.dataset import make_dataset_from_rlds

    ds, _ = make_dataset_from_rlds(
        **{**dataset_kwargs, "shuffle": False, "num_parallel_reads": 4, "num_parallel_calls": 4},
        train=train, exclude_empty_lang_instr=True,
    )
    ds = ds.traj_map(_keep_fingerprint, 4)
    languages, lengths, first_actions = [], [], []
    t0 = time.time()
    for i, traj in enumerate(ds.as_numpy_iterator()):
        lang, timestep = traj["language"], traj["timestep"]
        # one element must be one whole trajectory: timesteps 0..T-1, one instruction throughout
        if not np.array_equal(timestep, np.arange(len(timestep))):
            raise ValueError(f"trajectory {i}: timesteps are not 0..T-1: {timestep[:10]}")
        if not np.all(lang == lang[0]):
            raise ValueError(f"trajectory {i}: instruction changes within the trajectory")
        if lang[0] == b"":
            raise ValueError(f"trajectory {i}: empty instruction survived the filter")
        languages.append(lang[0].decode("utf-8"))
        lengths.append(len(timestep))
        first_actions.append(traj["first_action"])
        if (i + 1) % 5000 == 0:
            logging.info("  %d trajectories, %d unique instructions, %.0fs",
                         i + 1, len(set(languages)), time.time() - t0)
        if FLAGS.max_trajectories and i + 1 >= FLAGS.max_trajectories:
            break
    if not languages:
        raise ValueError("split yielded no trajectories")
    return languages, np.asarray(lengths), np.stack(first_actions)


def main(_):
    import dlimp as dl
    from octo.data.oxe import make_oxe_dataset_kwargs_and_weights

    logging.info("dlimp.DLataset.from_rlds source:\n%s", inspect.getsource(dl.DLataset.from_rlds))
    dataset_kwargs_list, _ = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config.oxe_kwargs)
    with h5py.File(FLAGS.out_path, "w") as f:
        f.attrs["created_at"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["max_trajectories"] = FLAGS.max_trajectories
        for kwargs in dataset_kwargs_list:
            name = kwargs["name"]
            available = sorted(tfds.builder(name, data_dir=kwargs["data_dir"]).info.splits.keys())
            logging.info("%s: TFDS splits %s", name, available)
            for split in FLAGS.splits:
                if split not in ("train", "val"):
                    raise ValueError(f"unknown split {split}")
                logging.info("walking %s / %s", name, split)
                languages, lengths, first_actions = walk_split(kwargs, train=(split == "train"))
                g = f.create_group(f"{name}/{split}")
                g.create_dataset("language", data=np.array(languages, dtype=object),
                                 dtype=h5py.string_dtype("utf-8"))
                g.create_dataset("traj_len", data=lengths)
                g.create_dataset("first_action", data=first_actions)
                g.attrs["tfds_splits_available"] = available
                logging.info("%s / %s: %d trajectories, %d unique instructions",
                             name, split, len(languages), len(set(languages)))
    logging.info("wrote %s", FLAGS.out_path)


if __name__ == "__main__":
    app.run(main)
