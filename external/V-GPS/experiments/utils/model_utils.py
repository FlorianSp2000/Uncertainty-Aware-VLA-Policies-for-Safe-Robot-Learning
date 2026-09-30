import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from functools import partial
import yaml
import wandb

from ml_collections import ConfigDict
from jaxrl_m.agents import agents
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors
from flax.training import checkpoints
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights

def process_text(batch, keep_language_str=False, text_processor=None):
    decoded_strings = [s.decode("utf-8") for s in batch["goals"]["language"]]
    if text_processor is not None:
        batch["goals"]["language"] = text_processor.encode(decoded_strings)
    if keep_language_str:
        batch["goals"]["language_str"] = decoded_strings
    return batch

def process_oxe_batch(batch, keep_language_str=False, training=False, text_processor=None, ensemble_size=8, batch_size_per_member=None):
    """Preprocess training or validation batch. Set keep_language_str=True for validation."""
    
    def reshape_to_ensemble(x):
        return x.reshape(ensemble_size, batch_size_per_member, *x.shape[1:])

    pre_batch = {
        "actions": batch["action"].squeeze(),
        "next_actions": batch["next_action"].squeeze(),
        "goals": {
            "language": batch["task"]["language_instruction"]
        },
        "mc_returns": batch["mc_return"],
        "observations": {"image": batch["observation"]["image_primary"].squeeze()},
        "next_observations": {"image": batch["next_observation"]["image_primary"].squeeze()},
        "rewards": batch["reward"],
        "masks": batch["td_mask"],
    }

    processed_batch = process_text(pre_batch, keep_language_str=keep_language_str, text_processor=text_processor)

    if training: # in training, we want to reshape (batch_size,) → (ensemble_size, batch_size // ensemble_size)
        ensemble_batch = jax.tree_map(reshape_to_ensemble, processed_batch)
    else: # in validation, we want to keep the batch size as is as we are duplicating the batch
        ensemble_batch = processed_batch 

    return ensemble_batch

def create_ensemble_agents(ensemble_size, rng, template_batch, encoder_def, agent_class, agent_kwargs):
    """
    Create ensemble agents with vectorized creation if available.
    """
    print("Creating ensemble agents...")
    
    # Check if agent class supports vectorized ensemble creation
    if hasattr(agent_class, 'create_ensemble_vectorized'):
        print("Using vectorized ensemble agent creation")
        ensemble_agents = agent_class.create_ensemble_vectorized(
            base_rng=rng,
            ensemble_size=ensemble_size,
            observations=template_batch["observations"],
            actions=template_batch["actions"],
            encoder_def=encoder_def,
            goals=template_batch["goals"],
            **agent_kwargs
        )
    else:
        print("Using standard ensemble agent creation")
        raise ValueError('create_ensemble_vectorized not implemented for this agent class')
    
    return ensemble_agents

def load_ensemble_agent(resume_path: str, config: ConfigDict, data_mix: str = "bridge_fractal", data_dir: str = None, step: int = None):
    """Load ensemble agent from checkpoint using your existing pipeline.

    Args:
        resume_path: Path to ensemble checkpoint directory
        config: Training config
        data_mix: Dataset mix name
        data_dir: Dataset directory
        step: Specific checkpoint step to load (None = latest)
    """

    print(f"Loading ensemble from {resume_path}..." + (f" (step={step})" if step else " (latest)"))
    
    ensemble_size = config.ensemble_size
    batch_size_per_member = config.batch_size // ensemble_size
    
    # Create encoder
    encoder_def = encoders[config.encoder](**config.encoder_kwargs)
    
    # Setup text processor
    text_processor = None
    if config.get("text_processor"):
        text_processor = text_processors[config.text_processor](**config.text_processor_kwargs)
    
    # Create dataset to get real template batch
    if "oxe_kwargs" in config.oxedata_config:
        (
            config.oxedata_config["dataset_kwargs_list"],
            config.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(**config.oxedata_config["oxe_kwargs"])
        oxe_kwargs = config.oxedata_config["oxe_kwargs"]
        del config.oxedata_config["oxe_kwargs"]
    
    train_data = make_interleaved_dataset(
        dataset_kwargs_list=config.oxedata_config["dataset_kwargs_list"],
        sample_weights=config.oxedata_config["sample_weights"],
        train=True,
        shuffle_buffer_size=1000,
        batch_size=config.batch_size,
        balance_weights=True,
        traj_transform_kwargs={},
        frame_transform_kwargs={}
    )
    
    # Create iterator and get example batch
    train_iterator = map(partial(process_oxe_batch, training=True, 
        text_processor=text_processor, ensemble_size=ensemble_size, batch_size_per_member=batch_size_per_member), train_data.iterator(prefetch=0))

    example_batch = next(train_iterator)  # Already ensemble-shaped because of process_oxe_batch's reshape !
    template_batch = jax.tree_map(lambda x: x[0], example_batch)  # Extract single member for template

    print(f"example_batch batch shape: {jax.tree_map(lambda x: x.shape, example_batch)}")
    print(f"Template batch shape: {jax.tree_map(lambda x: x.shape, template_batch)}")
    
    # Get agent class
    agent_class = agents[config.agent]
    
    # Create ensemble agents
    rng = jax.random.PRNGKey(config.seed)
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, template_batch, encoder_def, agent_class, config.agent_kwargs
    )
    
    # Restore from checkpoints following your exact pattern
    print("Restoring ensemble agents from checkpoint..." + (f" (step={step})" if step else " (latest)"))
    restored_agents = []
    for member_idx in range(ensemble_size):
        member_checkpoint_path = os.path.join(resume_path, f"ensemble_member_{member_idx}")
        if tf.io.gfile.exists(member_checkpoint_path):
            agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
            agent = checkpoints.restore_checkpoint(member_checkpoint_path, target=agent, step=step)
            print(f"Restored ensemble member {member_idx} from {member_checkpoint_path}" + (f" step={step}" if step else ""))
        else:
            print(f"Checkpoint not found for member {member_idx}, using random init")
            agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
        restored_agents.append(agent)
    
    ensemble_agents = jax.tree_map(lambda *args: jnp.stack(args, axis=0), *restored_agents)

    print("Ensemble loading complete!")
    return ensemble_agents, train_iterator

@partial(jax.vmap, in_axes=(0, None, 0)) # will use broadcasting TODO: test equivalence with 0,0,0 and duplicating
def _ensemble_forward_critic(agent, batch, rng_key):
    """Vectorized critic forward pass across ensemble members on same batch.
    Efficient when evaluating ensemble on same batch; If each agent should see different batches use in_axes(0,0,0) which
    creates 1-1 mapping of agents to batches.
    """
    obs_with_goals = agent._include_goals_in_obs(batch, "observations")
    return agent.forward_critic(obs_with_goals, batch["actions"], rng=rng_key, train=False)

def compute_ensemble_uncertainty(ensemble_agent, batch, ensemble_size=8):
    """
    expects traj dict, jax tree-like structure with values of dimension: (ensemble_size, batch_size, action_dim/obs_dim/reward_dim)
    :param ensemble_agents: Array of ensemble agent instances (8,)
    :param batch: Single batch dictionary (no duplication needed)
    :param ensemble_size: Number of ensemble members
    :returns: Scalar disagreement measure
    """
    rng = jax.random.PRNGKey(0) # TODO: should we use passed rng or a fixed one? cql.py also uses PRNGKey(0) here
    val_rngs = jax.random.split(rng, ensemble_size)
    q_values = _ensemble_forward_critic(ensemble_agent, batch, val_rngs)
    assert jnp.ndim(q_values) == 3 # We get back (ensemble_size, 2, batch_size) where I assume 2 is from number of critic networks
    mean_q_value_per_member = jnp.mean(q_values, axis=[1,2])
    disagreement = jnp.std(mean_q_value_per_member)
    return disagreement, mean_q_value_per_member


def load_roboreward_ensemble_agent(
    resume_path: str,
    config,
    action_dim: int = 7,
    step: int = None,
):
    """Load roboreward ensemble agent without loading the dataset.

    Uses synthetic template batch (zeros) — same approach as load_classifier_agent().
    Config should be loaded via get_config("ensemble_sarsa_roboreward").

    Args:
        resume_path: Path to ensemble checkpoint directory (contains ensemble_member_0..N)
        config: Training ConfigDict (ensemble_sarsa_roboreward); must have ensemble_size set
        action_dim: Action dimension (default 7 for bridge/fractal); TODO: store in config
        step: Specific checkpoint step to load (None = latest)

    Returns:
        (ensemble_agents, text_processor)
    """
    ensemble_size = config.ensemble_size
    print(f"Loading roboreward ensemble ({ensemble_size} members) from {resume_path}...")

    text_processor = text_processors[config.text_processor](**config.text_processor_kwargs)
    encoder_def = encoders[config.encoder](**config.encoder_kwargs)

    # TODO: Move to read from config / wandb
    # Synthetic template batch — no dataset loading needed
    template_batch = {
        "observations":      {"image": np.zeros((1, 256, 256, 3), dtype=np.uint8)},
        "next_observations": {"image": np.zeros((1, 256, 256, 3), dtype=np.uint8)},
        "actions":           np.zeros((1, action_dim), dtype=np.float32),
        "next_actions":      np.zeros((1, action_dim), dtype=np.float32),
        "goals":             {"language": np.zeros((1, 512), dtype=np.float32)},
        "rewards":           np.zeros((1,), dtype=np.float32),
        "masks":             np.zeros((1,), dtype=np.float32),
        "mc_returns":        np.zeros((1,), dtype=np.float32),
    }

    rng = jax.random.PRNGKey(config.seed)
    agent_class = agents[config.agent]
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, template_batch, encoder_def, agent_class, config.agent_kwargs
    )

    restored = []
    for idx in range(ensemble_size):
        member_path = os.path.join(resume_path, f"ensemble_member_{idx}")
        agent = jax.tree_map(lambda x, i=idx: x[i], ensemble_agents)
        agent = checkpoints.restore_checkpoint(member_path, target=agent, step=step)
        assert agent.state.step != 0, f"Checkpoint restore failed for member {idx} (step=0)"
        print(f"  Restored member {idx} (step={agent.state.step})")
        restored.append(agent)

    ensemble_agents = jax.tree_map(lambda *args: jnp.stack(args, axis=0), *restored)
    print("Roboreward ensemble loading complete!")
    return ensemble_agents, text_processor


def load_classifier_agent(checkpoint_path, wandb_run_name="", config_yaml_path=None):
    """
    Load binary classifier from checkpoint with config from wandb or yaml.

    Args:
        checkpoint_path: Path to checkpoint directory or specific checkpoint file.
                        Supports both:
                        - Directory: .../checkpoints (finds latest checkpoint)
                        - Specific: .../checkpoints/checkpoint_100000
        wandb_run_name: WandB run identifier (e.g., "username/project/run_id")
        config_yaml_path: Fallback YAML config path if no wandb

    Returns:
        (agent, text_processor) tuple
    """
    # 1. Load config (prioritize wandb, fallback to yaml)
    if wandb_run_name:
        api = wandb.Api()
        run = api.run(wandb_run_name)
        config = run.config
    elif config_yaml_path:
        with open(config_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Must provide wandb_run_name or config_yaml_path")

    # 2. Create encoder
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])

    # 3. Create agent (example batch for init) TODO: generalize obs/action shapes
    agent = agents[config["agent"]].create(
        rng=jax.random.PRNGKey(0),
        encoder_def=encoder_def,
        observations={"image": np.zeros((1, 256, 256, 3), dtype=np.uint8)},
        goals={"language": np.zeros((1, 512), dtype=np.float32)},
        **config["agent_kwargs"],
    )

    # 4. Restore checkpoint
    # Flax's restore_checkpoint expects a directory, not a specific checkpoint file
    # If path ends with checkpoint_*, use parent directory
    checkpoint_path_obj = Path(checkpoint_path)
    if checkpoint_path_obj.name.startswith('checkpoint_'):
        checkpoint_dir = str(checkpoint_path_obj.parent)
    else:
        checkpoint_dir = str(checkpoint_path_obj)

    # ASSERTION: Checkpoint directory must exist
    assert os.path.exists(checkpoint_dir), f"Checkpoint directory not found: {checkpoint_dir}"

    agent = checkpoints.restore_checkpoint(checkpoint_dir, agent)

    # ASSERTION: Verify checkpoint was actually loaded (not untrained model)
    assert agent.state.step != 0, "Checkpoint loading failed: model remains untrained (step=0)"

    # 5. Load text processor
    text_processor = None
    if config.get("text_processor"):
        text_processor = text_processors[config["text_processor"]](
            **config.get("text_processor_kwargs", {})
        )

    return agent, text_processor


"""
Example Computation
# q_values = compute_ensemble_uncertainty(ensemble_agents, duplicate_batch, ensemble_size) # for (0,0,0), right now use we replace duplicate batch with single mini per-member batch
"""


def load_octo_scratch_model(checkpoint_path: str, step: int = None):
    """Load an Octo model trained from scratch (not used for the thesis results)

    Uses OctoModel.load_pretrained() which loads:
    - config.json
    - example_batch.msgpack
    - dataset_statistics.json
    - checkpoint/ (params via orbax)

    Args:
        checkpoint_path: Path to experiment directory with step subdirs
                        e.g., "/V-GPS/results/octo_scratch_baseline/OctoScratch_vit_s_b256_s42_neg0.50"
        step: Specific checkpoint step (e.g., 240000). If None, loads latest.

    Returns:
        (model, text_processor) tuple where model is OctoModel instance
    """
    from octo.model.octo_model import OctoModel

    print(f"Loading Octo model from: {checkpoint_path}")
    model = OctoModel.load_pretrained(checkpoint_path, step=step)
    text_processor = model.text_processor

    print(f"✓ Loaded model with heads: {list(model.config['model']['heads'].keys())}")
    print(f"✓ Checkpoint step: {step}")

    return model, text_processor


# Alias: works for both scratch-trained and finetuned Octo models
load_octo_model = load_octo_scratch_model