import dlimp as dl
import jax
import numpy as np
import tensorflow as tf


# =============================================================================
# V2 Functions: Per-Batch Random Swapping (Prevents Memorization)
# =============================================================================

def collect_prompt_pool(dataset, prompt_pool_size: int = 10_000):
    """
    V2: Collects prompt pool from any dataset (trajectory-level or flattened samples).
    Works with both make_single_dataset and make_interleaved_dataset outputs.

    NOTE: swap_batch_prompts_tf creates task_feasible field, so no need to pre-add it.

    :param dataset: Any TF dataset with 'task'/'language_instruction' field
    :param prompt_pool_size: Number of samples to collect prompts from
    :returns: List of unique prompts (bytes)
    """
    print("V2: Collecting language prompts for batch-level swapping...")
    language_prompts = []
    # TODO: find out what type really is coming out of the dataset
    for example in dataset.take(prompt_pool_size):
        lang = example['task']['language_instruction']
        # Handle both trajectory (shape [S]) and sample (shape [] or scalar) cases
        if hasattr(lang, 'numpy'):
            lang = lang.numpy()
        if hasattr(lang, '__len__') and len(lang.shape) > 0:
            lang = lang[0]  # Take first element if sequence
        if lang:  # Skip empty
            language_prompts.append(lang)

    unique_prompts = list(set(language_prompts))
    print(f"V2: Collected {len(unique_prompts)} unique prompts from {len(language_prompts)} samples")
    return unique_prompts


def prepare_prompt_pool_and_dataset(dataset: dl.DLataset, prompt_pool_size: int = 10_000):
    """
    V2: Collects prompt pool and adds task_feasible=1.0 to all trajectories.
    Swapping happens at batch processing time via swap_batch_prompts_tf().

    This prevents memorization because:
    - V1: tf.random.set_seed() made same trajectory → same label every epoch
    - V2: Random decision made fresh per batch, same traj can be pos/neg

    :param dataset: DLataset containing trajectories
    :param prompt_pool_size: Number of trajectories to sample prompts from
    :returns: (dataset_with_feasibility_field, unique_prompts_list)
    """
    # Step 1: Collect unique prompts for later batch-level swapping
    print("V2: Collecting language prompts for batch-level swapping...")
    language_prompts = []

    for example in dataset.take(prompt_pool_size):
        lang_instruction = example['task']['language_instruction'].numpy()[0]
        if lang_instruction:  # Skip empty
            language_prompts.append(lang_instruction)

    unique_prompts = list(set(language_prompts))
    print(f"V2: Collected {len(unique_prompts)} unique prompts from {len(language_prompts)} samples")

    # Step 2: Add task_feasible=1.0 to all trajectories (will be modified at batch level)
    def add_feasibility_field(traj):
        traj = dict(traj)
        traj['task'] = dict(traj['task'])
        traj_length = tf.shape(traj['task']['language_instruction'])[0]
        traj['task_feasible'] = tf.ones([traj_length], dtype=tf.float32)
        return traj

    dataset_with_feasibility = dataset.traj_map(add_feasibility_field)

    return dataset_with_feasibility, unique_prompts


def swap_batch_prompts_tf(batch, prompt_pool_tensor, negative_ratio: float = 0.50):
    """
    Apply random prompt swapping at batch level. Called per-batch during training.

    Uses fresh random state each call → same trajectory gets different labels
    across epochs → no memorization possible.

    NOTE: Only used on batched data after .unbatch().batch() - shape is (B,).
    Trajectory-level data (S,) uses V1 create_negative_demonstrations_classifier.

    :param batch: Dict with batch['task']['language_instruction'] shape (B,)
                  and batch['task_feasible'] shape (B,)
    :param prompt_pool_tensor: tf.constant of unique prompts, shape (num_prompts,)
    :param negative_ratio: Fraction of samples to swap to infeasible
    :returns: Modified batch with random swaps applied
    """
    batch_size = tf.shape(batch['task']['language_instruction'])[0]
    num_prompts = tf.shape(prompt_pool_tensor)[0]

    # Fresh random mask each call (no seed → truly random)
    swap_mask = tf.random.uniform([batch_size]) < negative_ratio

    # Random prompt indices
    random_indices = tf.random.uniform([batch_size], 0, num_prompts, dtype=tf.int32)
    random_prompts = tf.gather(prompt_pool_tensor, random_indices)

    # Apply swap: shape (B,)
    original_prompts = batch['task']['language_instruction']
    swapped_prompts = tf.where(swap_mask, random_prompts, original_prompts)

    # Update feasibility: swapped = infeasible (0), kept = feasible (1)
    new_feasibility = tf.where(
        swap_mask,
        tf.zeros([batch_size], dtype=tf.float32),
        tf.ones([batch_size], dtype=tf.float32)
    )

    # Create new batch (don't mutate input)
    # Convert to numpy: tf.where returns EagerTensors that break Python iteration
    # (iterating TF dataset tensors auto-converts to numpy; tf.where output doesn't)
    new_batch = dict(batch)
    new_batch['task'] = dict(batch['task'])
    new_batch['task']['language_instruction'] = swapped_prompts.numpy()
    new_batch['task_feasible'] = new_feasibility.numpy()

    return new_batch


def _slice_batch(batch, start, end):
    """Slice all numpy-array leaves of a pytree along axis 0."""
    return jax.tree_map(lambda x: x[start:end].copy(), batch)


def _concat_batches(batches):
    """Concatenate a list of same-structure pytrees along axis 0."""
    return jax.tree_map(lambda *xs: np.concatenate(xs, axis=0), *batches)


def _sample_random_prompts(n, prompt_pool_tensor):
    """Sample n random prompts from pool; returns numpy bytes array shape (n,)."""
    pool = prompt_pool_tensor.numpy()
    indices = np.random.randint(0, len(pool), size=n)
    return pool[indices]


# =============================================================================
# V1 Functions (Original - Has Memorization Bug with Fixed Seed)
# Keep for backward compatibility, but prefer V2 for new experiments
# =============================================================================

def create_negative_demonstrations(dataset: dl.DLataset,
    negative_ratio: float = 0.05,
    seed: int = 42,
    prompt_pool_size: int = 10_000,
    use_bootstrap=True,
    negative_final_reward: float = -1.0,
    ):
    """
    Takes a dlimp dataset of non-flattened, consistent trajectories.
    Add negative demonstrations by modifying language prompts and rewards for a percentage of trajectories.
    This adds a new key 'original_language_instruction' to each trajectory for reference.

    :param dataset: DLataset containing trajectories
    :param negative_ratio: Fraction of trajectories to convert to negative demos (e.g., 0.05 for 5%)
    :param seed: Random seed for reproducible sampling
    :param use_bootstrap: If True, set td_mask to all ones for negative demos
    :returns: Dataset with negative demonstrations added
    """
    
    # Step 1: Collect all unique language prompts for random sampling
    print("Collecting language prompts...")
    language_prompts = []

    for example in dataset.take(prompt_pool_size):
        lang_instruction = example['task']['language_instruction'].numpy()[0]
        # assume we have filtered out all empty language instructions for building the prompt pool
        language_prompts.append(lang_instruction)
    
    print(f"len(language_prompts) is {len(language_prompts)}")
    # Remove duplicates and convert to tensor
    unique_prompts = list(set(language_prompts))
    language_prompts_tensor = tf.constant(unique_prompts)
    print(f"Collected {len(unique_prompts)} unique language prompts")
    
    def modify_trajectory(traj):
        """Randomly decide whether to convert trajectory to negative demo."""
        
        # Random decision based on negative_ratio
        should_modify = tf.random.uniform(()) < negative_ratio
        
        def create_negative_demo():
            """Convert to negative demo: random prompt + all rewards = -1"""
            # Random prompt selection
            random_idx = tf.random.uniform([], 0, len(unique_prompts), dtype=tf.int32)
            random_prompt = language_prompts_tensor[random_idx]
            
            # Create modified traj
            modified_example = dict(traj)
            modified_example['task'] = dict(traj['task'])
            modified_example['task']['original_language_instruction'] = traj['task']['language_instruction']

            # Replace language instruction with random prompt
            traj_length = tf.shape(traj['task']['language_instruction'])[0]
            modified_example['task']['language_instruction'] = tf.fill([traj_length], random_prompt)
            
            # Set rewards for negative demonstration
            # Base: -1 for all steps, final 3 steps: negative_final_reward
            traj_length = tf.shape(traj['reward'])[0]
            num_final = tf.minimum(3, traj_length)

            # Create mask: 0 for non-final, 1 for final 3 steps
            final_mask = tf.concat([
                tf.zeros(traj_length - num_final, dtype=tf.float32),
                tf.ones(num_final, dtype=tf.float32)
            ], axis=0)

            # Apply: base reward (-1) for most steps, negative_final_reward for last 3
            base_reward = tf.ones_like(traj['reward']) * -1.0
            modified_example['reward'] = (
                base_reward * (1.0 - final_mask) +
                negative_final_reward * final_mask
            )

            if use_bootstrap:
                modified_example['td_mask'] = tf.ones_like(traj['reward'])  # bootstrap through
            # TODO: This lets 'mc_return' that is computed in make_dataset_from_rlds become invalid 
            return modified_example
        
        def keep_original():
            """Keep original trajectory unchanged. Just add original_language_instruction key for consistency """
            unchanged_traj = dict(traj)
            unchanged_traj['task'] = dict(traj['task'])
            unchanged_traj['task']['original_language_instruction'] = traj['task']['language_instruction']
            return unchanged_traj
        
        # Conditionally apply modification
        return tf.cond(should_modify, create_negative_demo, keep_original)
    
    # Step 3: Apply the transformation
    print(f"Creating negative demonstrations ({negative_ratio*100:.1f}% of trajectories)...")
    modified_dataset = dataset.traj_map(modify_trajectory)
    
    return modified_dataset, language_prompts_tensor

def create_negative_demonstrations_classifier(dataset: dl.DLataset, 
    negative_ratio: float = 0.50, 
    seed: int = 42, 
    prompt_pool_size: int = 10_000,
    ):
    """
    Creates negative demonstrations for binary classifier baseline by swapping language prompts and adding feasibility labels.
    
    Key differences from V-GPS ensemble version:
    - No reward modification
    - Adds explicit 'task_feasible' flag instead of relying on key detection
    - Cleaner distinction between positive and negative samples
    
    :param dataset: DLataset containing trajectories
    :param negative_ratio: Fraction of trajectories to convert to negative demos
    :param seed: Random seed for reproducible sampling
    :param prompt_pool_size: Number of prompts to collect for random sampling
    :returns: Dataset with negative demonstrations and feasibility labels
    """
    
    # Step 1: Collect unique language prompts for random sampling
    print("Collecting language prompts for binary classifier negative demonstrations...")
    language_prompts = []

    for example in dataset.take(prompt_pool_size):
        lang_instruction = example['task']['language_instruction'].numpy()[0]
        language_prompts.append(lang_instruction)
    
    # Remove duplicates and convert to tensor
    unique_prompts = list(set(language_prompts))
    language_prompts_tensor = tf.constant(unique_prompts)
    print(f"Collected {len(unique_prompts)} unique language prompts out of {len(language_prompts)} for binary classifier")
    
    # Set up random selection
    tf.random.set_seed(seed)

    def modify_trajectory(traj):
        """Add feasibility labels and optionally swap language prompt for negative demos."""
        
        # Random decision based on negative_ratio
        should_create_negative = tf.random.uniform(()) < negative_ratio
        
        def create_negative_demo():
            """Create infeasible demo: swap language prompt, mark as infeasible."""
            # Random prompt selection (different from original)
            random_idx = tf.random.uniform([], 0, len(unique_prompts), dtype=tf.int32)
            random_prompt = language_prompts_tensor[random_idx]
            
            # Create modified trajectory
            modified_traj = dict(traj)
            modified_traj['task'] = dict(traj['task'])
            
            # Replace language instruction with random mismatched prompt
            traj_length = tf.shape(traj['task']['language_instruction'])[0]
            modified_traj['task']['language_instruction'] = tf.fill([traj_length], random_prompt)
            
            # Mark as infeasible
            modified_traj['task_feasible'] = tf.zeros([traj_length], dtype=tf.float32)
            
            return modified_traj
        
        def keep_positive_demo():
            """Keep original trajectory as feasible demo."""
            positive_traj = dict(traj)
            positive_traj['task'] = dict(traj['task'])
            
            # Mark as feasible
            traj_length = tf.shape(traj['task']['language_instruction'])[0]
            positive_traj['task_feasible'] = tf.ones([traj_length], dtype=tf.float32)
            
            return positive_traj
        
        # Conditionally apply modification
        return tf.cond(should_create_negative, create_negative_demo, keep_positive_demo)
    
    # Apply the transformation
    print(f"Creating binary classifier negative demonstrations ({negative_ratio*100:.1f}% infeasible trajectories)...")
    modified_dataset = dataset.traj_map(modify_trajectory)

    return modified_dataset, language_prompts_tensor


def create_negative_demonstrations_octo_debug(dataset: dl.DLataset,
    negative_ratio: float = 0.50,
    seed: int = 42,
    ):
    """
    Debug version: Replace language with label strings ("feasible" / "infeasible").

    This makes the task trivially easy - model only needs to learn:
    - "feasible" → task_feasible = 1
    - "infeasible" → task_feasible = 0

    Use this to test if model CAN learn at all, independent of visual reasoning.

    :param dataset: DLataset containing trajectories
    :param negative_ratio: Fraction of trajectories to mark as infeasible
    :param seed: Random seed for reproducible sampling
    :returns: Dataset with label-string language prompts
    """

    # Set up random selection
    tf.random.set_seed(seed)

    def modify_trajectory(traj):
        """Replace language with label string based on random assignment."""

        # Random decision based on negative_ratio
        should_create_negative = tf.random.uniform(()) < negative_ratio

        def create_negative_demo():
            """Mark as infeasible: language = "infeasible"."""
            modified_traj = dict(traj)
            modified_traj['task'] = dict(traj['task'])

            # Replace language instruction with label string
            traj_length = tf.shape(traj['task']['language_instruction'])[0]
            modified_traj['task']['language_instruction'] = tf.fill([traj_length], b'infeasible')

            # Mark as infeasible
            modified_traj['task_feasible'] = tf.zeros([traj_length], dtype=tf.float32)

            return modified_traj

        def keep_positive_demo():
            """Mark as feasible: language = "feasible"."""
            positive_traj = dict(traj)
            positive_traj['task'] = dict(traj['task'])

            # Replace language instruction with label string
            traj_length = tf.shape(traj['task']['language_instruction'])[0]
            positive_traj['task']['language_instruction'] = tf.fill([traj_length], b'feasible')

            # Mark as feasible
            positive_traj['task_feasible'] = tf.ones([traj_length], dtype=tf.float32)

            return positive_traj

        # Conditionally apply modification
        return tf.cond(should_create_negative, create_negative_demo, keep_positive_demo)

    # Apply the transformation
    print(f"DEBUG MODE: Replacing language with label strings (feasible/infeasible)")
    print(f"Creating {negative_ratio*100:.1f}% infeasible, {(1-negative_ratio)*100:.1f}% feasible trajectories...")
    modified_dataset = dataset.traj_map(modify_trajectory)

    return modified_dataset, None  # No language prompts tensor needed
