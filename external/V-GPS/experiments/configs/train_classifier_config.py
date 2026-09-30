# Config adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/configs/train_config.py.
"""
Configuration for binary classifier training.
Uses ResNet+MUSE architecture for feasibility classification.
"""
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string):

    base_config = dict(
        batch_size=256,
        num_steps=100000,
        log_interval=100,
        eval_interval=1000,
        save_interval=2500,
        save_dir=placeholder(str),
        seed=42,
        # Negative demonstration settings
        create_negative_demos=True,
        negative_demo_ratio=0.50,
        exclude_empty_lang_instr=True,
    )

    possible_structures = {
        "binary_classifier": ConfigDict(
            dict(
                agent="binary_classifier",
                agent_kwargs=dict(
                    # Goal conditioning (same as SARSA)
                    language_conditioned=True,
                    goal_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,

                    # Learning parameters
                    learning_rate=3e-4,
                    warmup_steps=2000,

                    # Network architecture (same as SARSA)
                    network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                ),

                # Text processing (same as SARSA)
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),

                # Encoder (same as SARSA)
                encoder="resnetv1-34-bridge-film",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),

                **base_config,
            )
        ),
    }

    return possible_structures[config_string]
