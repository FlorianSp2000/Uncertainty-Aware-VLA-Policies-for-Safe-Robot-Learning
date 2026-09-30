# Config adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/configs/train_config.py.
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(config_string):
    """
    Corrected ensemble configuration with proper agent naming.
    """
    base_real_config = dict(
        batch_size=256,  # This will be divided by ensemble_size
        num_steps=int(1e6),
        log_interval=1000,
        eval_interval=20000,
        save_interval=50000,
        save_dir=placeholder(str),
        resume_path="",
        resume_wandb_id=placeholder(str),
        seed=42,
        ensemble_size=8,  # Number of ensembles
        create_negative_demos=False,  # Include Trajectories with non-matching language prompts
        negative_demo_ratio=0.05,
        negative_final_reward=-1.0,  # Reward for final 3 steps of negative demos (ablation param)
        num_train_samples=0,  # 0 = use all, >0 = .take(N) subset for ablations
        exclude_empty_lang_instr=False,  # Exclude empty language instructions from training
        contrastive_negative_ratio=0.1,  # Fraction of batch to swap for contrastive loss
        contrastive_pairing_mode="random",  # "random" | "paired" | "hybrid_paired"
    )

    # reward_scheme used by train_ensemble_roboreward.py only — NOT part of base_real_config
    # so train_ensemble.py (bridge/fractal) is completely unaffected.
    _roboreward_reward_scheme = ConfigDict({
        "per_step_reward": -1.0,        # reward at non-terminal steps
        "terminal_transform": "raw",    # "raw" | "binary" | "normalize_01" | "normalize_11"
        "binary_pos_threshold": 4,      # binary: score >= threshold → pos_val
        "binary_pos_val": 1.0,          # binary: positive class value
        "binary_neg_val": -1.0,         # binary: negative class value
        "num_final_repeat": 3,          # how many terminal steps get terminal reward
        "normalize_by_traj_len": False, # divide only non-terminal per-step rewards by their count
        # td_mask: controls where Bellman bootstrap is disabled (mask=0 → no bootstrap)
        "td_mask_mode": "fixed",        # "fixed" | "all_ones" | "score_threshold"
        # "fixed":           [1,1,...,0,0,0]  last num_final_repeat steps always terminal
        # "all_ones":        [1,1,...,1,1,1]  always bootstrap (timeout/truncated episodes)
        # "score_threshold": [1,...,0,0,0] if score>=threshold else [1,...,1,1,1]
        "td_mask_score_threshold": 5,   # score_threshold only: min score to treat as terminal
    })

    # Plain dict so both ensemble_sarsa and ensemble_sarsa_roboreward can spread it
    # without copying a ConfigDict (which would lose placeholder type info).
    _sarsa_base_dict = dict(
        agent="sarsa_ensemble",
        agent_kwargs=dict(
            # Goal conditioning
            language_conditioned=True,
            goal_conditioned=True,
            early_goal_concat=None,
            shared_goal_encoder=None,
            shared_encoder=False,
            # Learning parameters
            learning_rate=3e-4,
            warmup_steps=2000,
            discount=0.98,
            soft_target_update_rate=5e-3,
            # Critic configuration
            critic_ensemble_size=2,
            critic_subsample_size=None,
            use_min_q=True,
            # Network architecture
            network_kwargs=dict(
                hidden_dims=[256, 256],
                activate_final=True,
                use_layer_norm=False,
            ),
            # OOD evaluation
            num_ood_actions=10,
            ood_action_sample_method="uniform",
            # Contrastive loss (disabled by default)
            contrastive_weight=0.0,
            contrastive_margin=1.0,
            # Expose the critic's encoder output (obs_encoding) as a Flax intermediate; needed for
            # feature dumps at eval time. Implied when contrastive_weight > 0. Adds no parameters.
            sow_embeddings=False,
        ),
        # Text processing
        text_processor="muse_embedding",
        text_processor_kwargs=dict(),
        # Encoder
        encoder="resnetv1-34-bridge-film",
        encoder_kwargs=dict(
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
        ),
        **base_real_config,
    )

    possible_structures = {
        "ensemble_sarsa": ConfigDict(_sarsa_base_dict),

        "ensemble_lc_cql": ConfigDict(
            dict(
                agent="cql",  # Keep as standard CQL for now
                agent_kwargs=dict(
                    language_conditioned=True,
                    goal_conditioned=True,
                    early_goal_concat=None,
                    shared_goal_encoder=None,
                    shared_encoder=False,
                    learning_rate=3e-4,
                    warmup_steps=2000,
                    discount=0.98,
                    cql_alpha=5.0,
                    target_update_rate=5e-3,
                    gc_kwargs=dict(negative_proportion=0.0),
                    use_calql=False,
                    critic_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_network_kwargs=dict(
                        hidden_dims=[256, 256],
                        activate_final=True,
                        use_layer_norm=False,
                    ),
                    policy_kwargs=dict(
                        tanh_squash_distribution=True,
                        std_parameterization="exp",
                    ),
                    actor_optimizer_kwargs=dict(
                        learning_rate=1e-4,
                        warmup_steps=2000,
                    ),
                    critic_optimizer_kwargs=dict(
                        learning_rate=3e-4,
                        warmup_steps=2000,
                    ),
                ),
                text_processor="muse_embedding",
                text_processor_kwargs=dict(),
                encoder="resnetv1-34-bridge-film",
                encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                ),
                **base_real_config,
            )
        ),
    }

    possible_structures["ensemble_sarsa_roboreward"] = ConfigDict({
        **_sarsa_base_dict,
        "reward_scheme": _roboreward_reward_scheme,
        "eval_inpainting": False,
        "eval_inpainting_data_paths": "",  # colon-separated pkl paths
    })

    # SigLIP So400m/14 backbone (frozen) from Pi0 checkpoint
    _siglip_agent_kwargs = dict(_sarsa_base_dict["agent_kwargs"])
    _siglip_agent_kwargs["use_siglip_wrapper"] = True
    _siglip_agent_kwargs["freeze_siglip_backbone"] = True  # CLI override: --config.agent_kwargs.freeze_siglip_backbone=False
    possible_structures["ensemble_sarsa_roboreward_siglip"] = ConfigDict({
        **{k: v for k, v in _sarsa_base_dict.items()
           if k not in ("encoder", "encoder_kwargs", "agent_kwargs", "agent")},
        "agent": "sarsa_ensemble_siglip",
        "encoder": "siglip-so400m-14",
        "encoder_kwargs": dict(),
        "agent_kwargs": _siglip_agent_kwargs,
        "reward_scheme": _roboreward_reward_scheme,
        "eval_inpainting": False,
        "eval_inpainting_data_paths": "",
        "siglip_checkpoint_path": "/V-GPS/checkpoints/pi0_base/params",
        "ensemble_size": 4,  # 4 GPUs for SigLIP (400M params per member)
    })

    return possible_structures[config_string]
