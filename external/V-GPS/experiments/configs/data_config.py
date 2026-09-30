# From V-GPS (https://github.com/nakamotoo/V-GPS) experiments/configs/data_config.py. Modified for this thesis.
from copy import deepcopy
import imp
import os
from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from octo.data.utils.data_utils import NormalizationType
from absl import logging

def get_dataset_config(window_size=1):
    task_augmentation = dict(
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=0.5,
        ),
    )

    return {
        # oxe_kwargs will generate dataset_kwargs_list and sampling weights
        "oxe_kwargs": dict(
            data_mix=placeholder(str),
            data_dir=placeholder(str),
            load_camera_views=("primary", "wrist"),
            load_depth=False,
        ),
        "traj_transform_kwargs": dict(
            window_size=window_size,
            action_horizon=1,
            goal_relabeling_strategy="uniform",
            subsample_length=100,
            **task_augmentation,
        ),
        "frame_transform_kwargs": dict(
            resize_size=(256, 256),
            image_dropout_prob=0.0,
            image_augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.1],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
            num_parallel_calls=200,
        ),
        "traj_transform_threads": 48,  # shared between all datasets
        "traj_read_threads": 48,  # shared between all datasets
        "shuffle_buffer_size": 100000,  # shared between all datasets
        "batch_size": 1024,
        "balance_weights": True,
    }


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):

    config = get_dataset_config(window_size=1)
    action_dim = FieldReference(7)

    primary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
   
    del config["frame_transform_kwargs"]["resize_size"]
    del config["frame_transform_kwargs"]["image_augment_kwargs"]

    config["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
    }
    config["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
    }

    if config_string == "baseline_classifier":
        
        config["frame_transform_kwargs"]["image_dropout_prob"] = 0.0

        # Call update_config with ALL Octo-specific settings
        config = update_config(
            config,
            oxe_kwargs=dict(
                data_dir=placeholder(str),
                data_mix="bridge_fractal",
                load_camera_views=("primary", ),
                load_depth=False,
                load_proprio=False,
                load_language=True,
                force_recompute_dataset_statistics=False,
                action_proprio_normalization_type=NormalizationType.NORMAL, # Octo uses z-score
                discount=0.98,
            ),
            traj_transform_kwargs=dict(
                window_size=1,
                action_horizon=4, # Match Octo's horizon
                max_action_dim=action_dim,
                task_augment_strategy="delete_task_conditioning",
                task_augment_kwargs=dict(
                    keep_image_prob=0.0, # 0% multimodal
                ),
                goal_relabeling_strategy=None, # Enable goal relabeling
            ),
            batch_size=256, # Smaller batch size for finetuning
            shuffle_buffer_size=10000, # Smaller shuffle buffer
            balance_weights=True,
            traj_transform_threads=16, # Balanced threading
            traj_read_threads=16,
        )
        
        return ConfigDict(config)

    config = update_config(
        config,
        oxe_kwargs=dict(
            data_dir=placeholder(str),
            data_mix="bridge_fractal",
            load_camera_views=("primary", ),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            force_recompute_dataset_statistics=False,
            discount=0.98,
            num_final_repeat=3,
            action_proprio_normalization_type=NormalizationType.BOUNDS, # we normalize actions to [-1, 1] with min max bounds,
        ),
        traj_transform_kwargs=dict(
            window_size=1,  # ADDED by me
            action_horizon=1,
            max_action_dim=action_dim,
            task_augment_strategy="delete_task_conditioning",
            # language_conditioned --> No goal relabeling, no image conditioning
            task_augment_kwargs=dict(
                keep_image_prob=0.0,
            ),
            goal_relabeling_strategy=None,
        ),
        frame_transform_kwargs=dict(
            image_dropout_prob=0.0,
        ),
        batch_size=512,
        shuffle_buffer_size=50000,
        balance_weights=True,
    )

    # SigLIP variant: 224x224 for So400m/14 (patch_size=14, 224/14=16 patches)
    # Must be AFTER update_config which overwrites frame_transform_kwargs
    if config_string == "siglip":
        config["frame_transform_kwargs"]["resize_size"] = {
            "primary": (224, 224),
        }

    return ConfigDict(config)