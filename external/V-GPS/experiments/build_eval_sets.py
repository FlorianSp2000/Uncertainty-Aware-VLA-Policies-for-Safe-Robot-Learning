"""Build the Bridge/Fractal evaluation trajectory sets (pkl) used by eval_perturbation_sensitivity.py.

Each pkl holds first/last frame, actions and instruction of N trajectories, half Bridge, half Fractal,
in the deterministic loader order (seed 44, no shuffling):

    bridge_fractal_val_n200.pkl   --split val   --n 200   (language relabeling + action perturbations)
    bridge_fractal_n300.pkl       --split val   --n 300   (object removal; inpainted afterwards, see docs/DATA.md)
    bridge_fractal_train_n200.pkl --split train --n 200   (reference set for encoder novelty, seen/unseen split)

The val sets use `utils.ood_utils.extract_trajectory_data` unchanged. The train set uses the
train-split variant of the same function, which lived in a notebook when the thesis sets were built
and is moved here verbatim.

Run inside the train_q.sif container with the repo's external/V-GPS mounted at /V-GPS:
    python /V-GPS/experiments/build_eval_sets.py --split val --n 200 --out /V-GPS/experiments/data/bridge_fractal_val_n200.pkl
"""
import argparse
import os
import pickle
from pathlib import Path

import tensorflow as tf

from octo.data.dataset import make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_utils import filter_eval_datasets

from experiments.utils.constants import ENSEMBLE_MODEL_PATH
from experiments.utils.ood_utils import (
    _extract_from_single_dataset,
    _get_full_dataset_name,
    extract_trajectory_data,
    load_configs_for_jupyter,
)


def extract_trajectory_data_split(dataset_name: str, max_trajectories: int = None,
                                  keep_full_trajectory=False, keep_raw_instruction=False,
                                  save_trajectories: bool = False, save_path: str = None,
                                  save_format: str = "pkl", mix_ratio: dict = None,
                                  seed: int = 44, skip_missing_instruction: bool = True,
                                  train: bool = False):
    """Extract trajectory data from single or mixed OXE datasets.

    :param dataset_name: "bridge", "fractal", or "bridge_fractal" for mixed
    :param max_trajectories: Max trajectories to extract
    :param keep_full_trajectory: Keep full trajectory data
    :param keep_raw_instruction: Keep raw instruction bytes
    :param save_trajectories: Save extracted data to disk
    :param save_path: Path to save file
    :param save_format: Save format (pkl)
    :param mix_ratio: Dict like {"bridge": 0.6, "fractal": 0.4}
    :param seed: Random seed for deterministic ordering
    :param skip_missing_instruction: Skip trajectories without instructions
    :param train: If True, use training set; if False, use validation set
    :returns: List of extracted trajectories
    """
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    FLAGS, oxe_config = load_configs_for_jupyter(
        algorithm="ensemble_sarsa",
        data_dir="/V-GPS/datasets/open_x",
        batch_size=1024,
        ensemble_size=8,
        seed=seed,
        resume_path=ENSEMBLE_MODEL_PATH
    )

    # Create dataset kwargs
    (dataset_kwargs_list, sample_weights) = make_oxe_dataset_kwargs_and_weights(**oxe_config["oxe_kwargs"])

    # Handle dataset selection
    if dataset_name == "bridge_fractal":
        datasets_to_process = ["bridge", "fractal"]
        if mix_ratio is None:
            mix_ratio = {"bridge": 0.5, "fractal": 0.5}
    elif dataset_name in ["bridge", "fractal"]:
        datasets_to_process = [dataset_name]
        mix_ratio = {dataset_name: 1.0}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    all_trajectories = []

    for dataset_short_name in datasets_to_process:
        print(f"Processing {dataset_short_name} dataset ({'train' if train else 'val'})...")

        # Filter for specific dataset
        filter_name = _get_full_dataset_name(dataset_short_name)

        if train:
            filtered_kwargs = [k for k in dataset_kwargs_list if k['name'] == filter_name]
            if not filtered_kwargs:
                print(f"Warning: No data found for {dataset_short_name}")
                continue
            dataset_kwargs = filtered_kwargs[0]
        else:
            val_datasets_kwargs_list, _ = filter_eval_datasets(
                dataset_kwargs_list, sample_weights, [filter_name]
            )
            if not val_datasets_kwargs_list:
                print(f"Warning: No data found for {dataset_short_name}")
                continue
            dataset_kwargs = val_datasets_kwargs_list[0]

        # Calculate trajectory count for this dataset
        expected_count = int(max_trajectories * mix_ratio.get(dataset_short_name, 0)) if max_trajectories else None
        print(f"Target trajectories for {dataset_short_name}: {expected_count}")

        # Create dataset using make_single_dataset for both train and val
        dataset = make_single_dataset(
            dataset_kwargs={
                **dataset_kwargs,
                "num_parallel_reads": 4,
                "num_parallel_calls": 4,
                "shuffle": False,  # Keep deterministic order
            },
            traj_transform_kwargs={
                **oxe_config["traj_transform_kwargs"],
                "num_parallel_calls": 4,
            },
            frame_transform_kwargs={
                **oxe_config["frame_transform_kwargs"],
                "num_parallel_calls": 16,
            },
            train=train
        )

        dataset_trajectories = _extract_from_single_dataset(
            dataset, dataset_short_name, expected_count,
            keep_full_trajectory, keep_raw_instruction,
            skip_missing_instruction=skip_missing_instruction
        )

        all_trajectories.extend(dataset_trajectories)
        print(f"Extracted {len(dataset_trajectories)} trajectories from {dataset_short_name}")

    print(f"Total extracted: {len(all_trajectories)} trajectories from {dataset_name} ({'train' if train else 'val'})")

    if save_trajectories:
        if save_path is None:
            split_str = "train" if train else "val"
            num_tra_str = f"_n{max_trajectories}" if max_trajectories else "_all"
            save_path = f"data/{dataset_name}_{split_str}{num_tra_str}.pkl"
        elif not save_path.endswith('.pkl'):
            save_path += '.pkl'

        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving {len(all_trajectories)} trajectories to {save_path}...")

        save_data = {
            'trajectories': all_trajectories,
            'metadata': {
                'dataset_name': dataset_name,
                'split': 'train' if train else 'val',
                'num_trajectories': len(all_trajectories),
                'dataset_distribution': mix_ratio,
                'image_shape': all_trajectories[0]['first_image'].shape if all_trajectories else None,
                'action_shape': all_trajectories[0]['first_action'].shape if all_trajectories else None,
                'format_version': '1.1',
            }
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved to: {save_path}")

    return all_trajectories


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["val", "train"], required=True)
    ap.add_argument("--n", type=int, required=True, help="number of trajectories (half Bridge, half Fractal)")
    ap.add_argument("--dataset", default="bridge_fractal", choices=["bridge_fractal", "bridge", "fractal"])
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--out", required=True, help="output .pkl path")
    args = ap.parse_args()
    if not args.out.endswith(".pkl"):
        raise ValueError(f"--out must end in .pkl, got {args.out}")
    if Path(args.out).exists():
        raise FileExistsError(f"{args.out} exists; refusing to overwrite")

    if args.split == "val":
        extract_trajectory_data(args.dataset, max_trajectories=args.n, seed=args.seed,
                                save_trajectories=True, save_path=args.out)
    else:
        extract_trajectory_data_split(args.dataset, max_trajectories=args.n, seed=args.seed,
                                      save_trajectories=True, save_path=args.out, train=True)


if __name__ == "__main__":
    main()
