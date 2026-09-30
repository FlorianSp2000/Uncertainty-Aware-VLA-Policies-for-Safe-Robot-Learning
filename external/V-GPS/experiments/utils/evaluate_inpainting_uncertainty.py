from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from textwrap import wrap
# import pandas as pd
import h5py
import json
from datetime import datetime
from tqdm import tqdm
import re
from absl import logging

from experiments.utils.model_utils import load_ensemble_agent, compute_ensemble_uncertainty

def extract_timestamp_from_path(path):
    """Extract timestamp from model path to use as model ID.

    Searches for pattern YYYYMMDD_HHMMSS anywhere in path.
    Works for both:
    - Ensemble: .../VGPS_ensemble_sarsa_..._YYYYMMDD_HHMMSS
    - Classifier: .../ResNet_MUSE_classifier_..._YYYYMMDD_HHMMSS/checkpoints/checkpoint_100000
    """
    match = re.search(r'(\d{8}_\d{6})', path)
    return match.group(1) if match else None


def extract_unique_model_id_from_path(path):
    """Extract a unique model ID by including the path component that contains the timestamp.

    Unlike extract_timestamp_from_path, this includes the surrounding directory name so that
    models sharing a launch timestamp (e.g. reward ablations run together) get distinct IDs.

    Example: .../reward_ablation_neg1_YYYYMMDD_HHMMSS/... → reward_ablation_neg1_YYYYMMDD_HHMMSS
    Falls back to timestamp-only if no surrounding component is found.
    """
    match = re.search(r'([^/\\]+_\d{8}_\d{6})', path)
    if match:
        return match.group(1)
    return extract_timestamp_from_path(path)

def save_inference_results_hdf5(results, metadata, filepath):
    """Save inference results to HDF5 format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Save metadata as JSON string
        f.attrs['metadata'] = json.dumps(metadata)
        
        # Save all arrays uniformly
        for key, values in results.items():
            # Filter out None values and convert to numpy
            if key == 'failed_ids':
                # failed_ids is already a list of IDs, no None filtering needed
                f.create_dataset(key, data=np.array(values) if values else np.array([]))
            else:
                # For other arrays, filter None values
                clean_values = [v for v in values if v is not None]
                f.create_dataset(key, data=np.array(clean_values) if clean_values else np.array([]))

def load_inference_results_hdf5(filepath):
    """Load inference results from HDF5 format."""
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata = json.loads(f.attrs['metadata'])
        
        # Load all results uniformly
        results = {}
        for key in f.keys():
            results[key] = f[key][:].tolist()
        
        return results, metadata

def run_ensemble_inference_on_inpaintings(
    trajectories, ensemble_agents, text_processor,
    model_path, dataset_name,
    output_path="experiments",
    force_recompute=False,
    ensemble_size=8,
    model_type_tag="ensemble",
    save_results: bool = True,
):
    """
    Evaluate ensemble uncertainty on original vs inpainted observations.

    :param trajectories: List of trajectory dicts with original and inpainted first/last frames
    :param ensemble_agents: Loaded ensemble agent
    :param text_processor: Text processor for language encoding
    :param model_path: Path to model (for metadata)
    :param dataset_name: Name of dataset (for metadata and caching)
    :param output_path: Base output directory for experiments
    :param force_recompute: If True, recompute even if results file exists
    :param ensemble_size: Number of ensemble members
    :param model_type_tag: Tag used in HDF5 filename to avoid collisions (e.g. "roboreward_ensemble")
    :param save_results: If False, skip HDF5 caching entirely (in-memory only)
    :returns: Dict with original and inpainted uncertainty scores
    """

    if save_results:
        cache_path = Path(output_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        model_id = extract_unique_model_id_from_path(model_path)
        results_filename = f"inference_results_{dataset_name}_{model_type_tag}_{model_id}.hdf5"
        results_path = cache_path / results_filename

        if results_path.exists() and not force_recompute:
            logging.info(f"Loading existing results from: {results_path}")
            try:
                results, metadata = load_inference_results_hdf5(results_path)
                logging.info(f"Loaded {len(results.get('trajectory_ids', []))} results")
                return results
            except Exception as e:
                logging.warning(f"Failed to load results: {e}. Recomputing...")

    # Compute inference results
    logging.info(f"Computing inference results for {dataset_name} ({len(trajectories)} trajectories)...")

    def process_language_strings(batch):
        """Custom text processing for string input (not bytes)."""
        # Language is already decoded strings, just encode directly
        language_strings = batch["goals"]["language"]
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode(language_strings)
        return batch
    
    def create_batch_from_trajectory(traj, use_inpainted=False):
        """Convert trajectory dict to model input format."""
        # Select image source
        if use_inpainted:
            first_image = traj['first_image_inpainted'] 
            last_image = traj['last_image_inpainted']
        else:
            first_image = traj['first_image']
            last_image = traj['last_image']
        
        # Stack first and last frames to create 2-step trajectory
        images = np.stack([first_image, last_image])  # (2, 256, 256, 3)
        
        # Stack actions
        first_action = traj['first_action']
        last_action = traj['last_action'] 
        actions = np.stack([first_action, last_action])  # (2, action_dim)
        
        # Create language array (same for both steps) - already strings, not bytes
        language_strings = [traj['language'], traj['language']]  # List of strings
        
        # Create batch in expected format
        batch = {
            "actions": actions,
            "observations": {"image": images},
            "goals": {"language": language_strings}  # Strings, not bytes
        }
        
        # Apply custom text processing (handles strings directly)
        return process_language_strings(batch)
            
    # Storage for results
    results = {
        'original_uncertainties': [],
        'inpainted_uncertainties': [],
        'original_q_means': [],
        'original_q_maxes': [],
        'original_q_mins': [],
        'inpainted_q_means': [],
        'inpainted_q_maxes': [],
        'inpainted_q_mins': [],
        'trajectory_ids': [],
        'failed_ids': [] # # TODO: are currently only unique per dataset
    }
    
    logging.info(f"Evaluating {len(trajectories)} trajectories...")
    
    for traj in tqdm(trajectories, desc="Processing trajectories"):
        traj_id = traj.get('trajectory_id', len(results['trajectory_ids']))
        results['trajectory_ids'].append(traj_id)

        try:
            # 1. Inference with original observations
            original_batch = create_batch_from_trajectory(traj, use_inpainted=False)
            original_uncertainty, original_q_vals = compute_ensemble_uncertainty(ensemble_agents, original_batch, ensemble_size)
            
            results['original_uncertainties'].append(float(original_uncertainty))
            results['original_q_means'].append(float(jnp.mean(original_q_vals)))
            results['original_q_maxes'].append(float(jnp.max(original_q_vals)))
            results['original_q_mins'].append(float(jnp.min(original_q_vals)))

            # 2. Inference with inpainted observations
            inpainted_batch = create_batch_from_trajectory(traj, use_inpainted=True)
            inpainted_uncertainty, inpainted_q_vals = compute_ensemble_uncertainty(ensemble_agents, inpainted_batch, ensemble_size)

            results['inpainted_uncertainties'].append(float(inpainted_uncertainty))
            results['inpainted_q_means'].append(float(jnp.mean(inpainted_q_vals)))
            results['inpainted_q_maxes'].append(float(jnp.max(inpainted_q_vals)))
            results['inpainted_q_mins'].append(float(jnp.min(inpainted_q_vals)))

        except Exception as e:
            logging.error(f"Failed on trajectory {traj_id}: {e}")
            for key in results.keys():
                if key not in ['trajectory_ids', 'failed_ids']:
                    results[key].append(None)
            results['failed_ids'].append(traj_id)

    logging.info(f"Completed! Failed on {len(results['failed_ids'])} trajectories")

    if save_results:
        metadata = {
            'model_path': model_path,
            'model_id': model_id,
            'model_type': 'ensemble',
            'dataset_name': dataset_name,
            'n_trajectories': len(trajectories),
            'n_successful': len([x for x in results['original_uncertainties'] if x is not None]),
            'ensemble_size': ensemble_size,
            'timestamp': datetime.now().isoformat()
        }
        save_inference_results_hdf5(results, metadata, results_path)
        logging.info(f"Saved inference results to: {results_path}")

    return results


def plot_inpainting_separation(results: dict, dataset_name: str):
    """
    Lightweight 2-panel plot for inline training evaluation.

    Returns (fig, metrics) where metrics includes:
    - mean_delta_unc: mean paired uncertainty shift, inp - orig
    - pct_paired_increase: fraction of matched scenes where inp > orig
    - auc_unc: cross-pair dominance P(inp_j > orig_i), i.e. Mann-Whitney / AUC view

    pct_paired_increase and auc_unc are complementary:
    - pct_paired_increase answers the within-scene question
    - auc_unc answers the distributional separation question
    Caller should plt.close(fig) after logging.
    """
    orig_unc = np.array([v for v in results["original_uncertainties"] if v is not None])
    inp_unc  = np.array([v for v in results["inpainted_uncertainties"] if v is not None])
    orig_q   = np.array([v for v in results["original_q_means"] if v is not None])
    inp_q    = np.array([v for v in results["inpainted_q_means"] if v is not None])

    assert len(orig_unc) > 0, f"No valid results for {dataset_name}"

    # labels = np.concatenate([np.zeros(len(orig_unc)), np.ones(len(inp_unc))])
    # scores = np.concatenate([orig_unc, inp_unc])
    mean_delta       = float(np.mean(inp_unc - orig_unc))
    # Paired within-scene effect: how often does inpainting increase uncertainty
    # for the exact same trajectory?
    pct_paired_increase = float(np.mean(inp_unc > orig_unc) * 100)
    # Mann-Whitney U equivalence: AUC-ROC = P(score_+ > score_-) over all cross-pairs.
    # Outer product gives N×N boolean matrix; mean = U / (N*N). No sklearn needed.
    # 0.5 = chance, 1.0 = inpainted always more uncertain than original.
    # Distributional separation view over all cross-pairs:
    # P(inp_j > orig_i). This complements the paired metric above rather than
    # replacing it, because it answers a different question.
    auc_unc          = float(np.mean(inp_unc[:, None] > orig_unc[None, :]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"{dataset_name}  | Δunc={mean_delta:.4f}  %paired↑={pct_paired_increase:.1f}%  AUC={auc_unc:.3f}",
        fontsize=13,
    )

    axes[0].hist(orig_unc, alpha=0.6, label="original", bins=30, color="#2E86C1")
    axes[0].hist(inp_unc,  alpha=0.6, label="inpainted", bins=30, color="#E74C3C")
    axes[0].set_title("Uncertainty (disagreement std)")
    axes[0].set_xlabel("std of Q-values across members")
    axes[0].legend()

    axes[1].hist(orig_q, alpha=0.6, label="original Q", bins=30, color="#2E86C1")
    axes[1].hist(inp_q,  alpha=0.6, label="inpainted Q", bins=30, color="#E74C3C")
    axes[1].set_title("Mean Q-value")
    axes[1].set_xlabel("mean Q across members")
    axes[1].legend()

    fig.tight_layout()
    metrics = {"mean_delta_unc": mean_delta, "pct_paired_increase": pct_paired_increase, "auc_unc": auc_unc}
    return fig, metrics


def plot_inpainting_outliers(
    trajectories,
    results: dict,
    dataset_name: str,
    n_positive: int = 3,
    n_negative: int = 3,
):
    """
    Plot qualitative examples for the strongest positive and negative uncertainty deltas.

    Positive delta means inpainted uncertainty > original uncertainty.
    Negative delta means original uncertainty > inpainted uncertainty.
    """

    valid_examples = []
    for traj, orig_unc, inp_unc, orig_q, inp_q, traj_id in zip(
        trajectories,
        results["original_uncertainties"],
        results["inpainted_uncertainties"],
        results["original_q_means"],
        results["inpainted_q_means"],
        results["trajectory_ids"],
    ):
        if orig_unc is None or inp_unc is None:
            continue
        valid_examples.append({
            "traj": traj,
            "traj_id": traj_id,
            "orig_unc": float(orig_unc),
            "inp_unc": float(inp_unc),
            "orig_q": None if orig_q is None else float(orig_q),
            "inp_q": None if inp_q is None else float(inp_q),
            "delta_unc": float(inp_unc - orig_unc),
        })

    if not valid_examples:
        raise ValueError(f"No valid inpainting examples available for {dataset_name}")

    sorted_examples = sorted(valid_examples, key=lambda item: item["delta_unc"])
    negative_examples = [ex for ex in sorted_examples if ex["delta_unc"] < 0][:n_negative]
    positive_examples = [ex for ex in reversed(sorted_examples) if ex["delta_unc"] > 0][:n_positive]
    selected_examples = positive_examples + negative_examples

    if not selected_examples:
        # Degenerate case where all deltas are exactly zero.
        selected_examples = list(reversed(sorted_examples[-min(len(sorted_examples), n_positive + n_negative):]))

    fig, axes = plt.subplots(len(selected_examples), 5, figsize=(18, 3.2 * len(selected_examples)))
    if len(selected_examples) == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"{dataset_name} | qualitative uncertainty outliers",
        fontsize=14,
    )

    for row_axes, example in zip(axes, selected_examples):
        traj = example["traj"]
        image_panels = [
            (traj["first_image"], "orig first"),
            (traj["first_image_inpainted"], "inpaint first"),
            (traj["last_image"], "orig last"),
            (traj["last_image_inpainted"], "inpaint last"),
        ]

        for ax, (image, title) in zip(row_axes[:4], image_panels):
            ax.imshow(image)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

            if "inpaint" in title:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#E74C3C")
                    spine.set_linewidth(3)
                    spine.set_visible(True)

        delta_unc = example["delta_unc"]
        case_label = "positive delta" if delta_unc > 0 else "negative delta"
        language = traj.get("language", "No language")
        wrapped_language = "\n".join(wrap(str(language), width=32))
        text_lines = [
            f"id: {example['traj_id']}",
            case_label,
            f"orig_unc={example['orig_unc']:.4f}",
            f"inp_unc={example['inp_unc']:.4f}",
            f"delta_unc={delta_unc:.4f}",
        ]
        if example["orig_q"] is not None and example["inp_q"] is not None:
            text_lines.extend([
                f"orig_q={example['orig_q']:.4f}",
                f"inp_q={example['inp_q']:.4f}",
            ])
        text_lines.extend(["", wrapped_language])

        row_axes[4].text(
            0.02,
            0.98,
            "\n".join(text_lines),
            ha="left",
            va="top",
            transform=row_axes[4].transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7", edgecolor="#555555"),
        )
        row_axes[4].set_title("details", fontsize=10)
        row_axes[4].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    return fig


def run_classifier_inference_on_orig_and_inpaintings(
    trajectories, classifier_agent, text_processor,
    model_path, dataset_name, output_path="experiments/results",
    force_recompute=False
):
    """
    Evaluate binary classifier on original vs inpainted observations.

    :param trajectories: List of trajectory dicts with original and inpainted first/last frames
    :param classifier_agent: Loaded binary classifier agent
    :param text_processor: Text processor for language encoding
    :param model_path: Path to model (for metadata)
    :param dataset_name: Name of dataset (for metadata and caching)
    :param output_path: Base output directory for experiments
    :param force_recompute: If True, recompute even if results file exists
    :returns: Dict with classifier probabilities and predictions
    """

    # Create output directory and results path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_id = extract_timestamp_from_path(model_path)
    results_filename = f"inference_results_{dataset_name}_classifier_{model_id}.hdf5"
    results_path = output_path / results_filename

    # Check if results file exists
    if results_path.exists() and not force_recompute:
        logging.info(f"Loading existing results from: {results_path}")
        try:
            results, metadata = load_inference_results_hdf5(results_path)
            logging.info(f"Loaded {len(results.get('trajectory_ids', []))} results")
            return results
        except Exception as e:
            logging.warning(f"Failed to load results: {e}. Recomputing...")

    # Compute inference results
    logging.info(f"Computing inference results (output: {results_path})")

    def process_language_strings(batch):
        """Custom text processing for string input (not bytes)."""
        # Language is already decoded strings, just encode directly
        language_strings = batch["goals"]["language"]
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode(language_strings)
        return batch

    def create_batch_from_trajectory(traj, use_inpainted=False):
        """Convert trajectory dict to model input format."""
        # Select image source
        if use_inpainted:
            first_image = traj['first_image_inpainted']
            last_image = traj['last_image_inpainted']
        else:
            first_image = traj['first_image']
            last_image = traj['last_image']

        # Stack first and last frames to create 2-step trajectory
        images = np.stack([first_image, last_image])  # (2, 256, 256, 3)

        # Create language array (same for both steps) - already strings, not bytes
        language_strings = [traj['language'], traj['language']]  # List of strings

        # Create batch in expected format
        batch = {
            "observations": {"image": images},
            "goals": {"language": language_strings}  # Strings, not bytes
        }

        # Apply custom text processing (handles strings directly)
        return process_language_strings(batch)

    # Storage for results
    results = {
        'original_probs': [],      # P(feasible | original)
        'inpainted_probs': [],     # P(feasible | inpainted)
        'original_preds': [],      # Binary: 1=feasible, 0=infeasible
        'inpainted_preds': [],
        'trajectory_ids': [],
        'failed_ids': []
    }

    logging.info(f"Evaluating {len(trajectories)} trajectories...")

    rng = jax.random.PRNGKey(0)

    for traj in tqdm(trajectories, desc="Processing trajectories"):
        traj_id = traj.get('trajectory_id', len(results['trajectory_ids'])) # TODO: what default is that supposed to be?
        results['trajectory_ids'].append(traj_id)

        try:
            # 1. Original inference
            orig_batch = create_batch_from_trajectory(traj, use_inpainted=False)
            obs_goals = (orig_batch["observations"], orig_batch["goals"])

            orig_logits = classifier_agent.forward_classifier(obs_goals, rng=rng, train=False)
            orig_probs = jax.nn.sigmoid(orig_logits)
            orig_pred = (jnp.mean(orig_probs) > 0.5).astype(jnp.float32)

            results['original_probs'].append(float(jnp.mean(orig_probs)))
            results['original_preds'].append(float(orig_pred))

            # 2. Inpainted inference
            inp_batch = create_batch_from_trajectory(traj, use_inpainted=True)
            obs_goals = (inp_batch["observations"], inp_batch["goals"])

            inp_logits = classifier_agent.forward_classifier(obs_goals, rng=rng, train=False)
            inp_probs = jax.nn.sigmoid(inp_logits)
            inp_pred = (jnp.mean(inp_probs) > 0.5).astype(jnp.float32)

            results['inpainted_probs'].append(float(jnp.mean(inp_probs)))
            results['inpainted_preds'].append(float(inp_pred))

        except Exception as e:
            logging.error(f"Failed on trajectory {traj_id}: {e}")
            for key in ['original_probs', 'inpainted_probs', 'original_preds', 'inpainted_preds']:
                results[key].append(None)
            results['failed_ids'].append(traj_id)

    logging.info(f"Completed! Failed on {len(results['failed_ids'])} trajectories")

    # Save results
    metadata = {
        'model_path': model_path,
        'model_id': model_id,
        'model_type': 'classifier',
        'dataset_name': dataset_name,
        'n_trajectories': len(trajectories),
        'n_successful': len([x for x in results['original_probs'] if x is not None]),
        'timestamp': datetime.now().isoformat()
    }

    save_inference_results_hdf5(results, metadata, results_path)
    logging.info(f"Saved inference results to: {results_path}")

    return results


def analyze_extreme_cases(trajectories, inference_results, n_examples=3, model_path="", dataset_name="bridge_fractal_n300_with_inpainted", output_path="experiments"):
    """
    Create in-depth analysis and plotting of Q value results during Inference: summary analysis and extreme cases.
    """
    
    # Create plots output directory
    plots_dir = Path(output_path) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Get dataset info and uncertainty arrays
    datasets = [traj['dataset'] for traj in trajectories]
    orig_all = np.array(inference_results['original_uncertainties'])
    inpaint_all = np.array(inference_results['inpainted_uncertainties'])
    uncertainty_differences = inpaint_all - orig_all
    
    # Dataset masks
    bridge_mask = np.array([d == 'bridge' for d in datasets])
    fractal_mask = np.array([d == 'fractal' for d in datasets])
    
    # extract q values
    orig_q_means = np.array(inference_results['original_q_means'])
    orig_q_maxes = np.array(inference_results['original_q_maxes'])
    orig_q_mins = np.array(inference_results['original_q_mins'])
    inpaint_q_means = np.array(inference_results['inpainted_q_means'])
    inpaint_q_maxes = np.array(inference_results['inpainted_q_maxes'])
    inpaint_q_mins = np.array(inference_results['inpainted_q_mins'])

    # Extract trajectory lengths
    traj_lengths = np.array([traj['traj_length'] for traj in trajectories])
    
    # Calculate ground truth Q-values (reward-to-go with discount=0.98)
    discount_factor = 0.98 # TODO: make configurable
    def calculate_ground_truth_q(traj_length):
        """Calculate ground truth Q-value using reward structure: -1 per step except last 3 (reward=0)"""
        # Reward structure: -1 for each step except last 3 steps get 0
        rewards = np.ones(traj_length) * (-1)
        if traj_length >= 3:
            rewards[-3:] = 0  # Last 3 steps have 0 reward
        
        # Calculate discounted return (reward-to-go from first step)
        q_ground_truth = 0
        for t in range(traj_length):
            q_ground_truth += (discount_factor ** t) * rewards[t]
        return q_ground_truth
        
    # ============ PLOT 1: SUMMARY ANALYSIS ============
    # fig1, axes = plt.subplots(4, 3, figsize=(28, 20))  # Changed from 2,3 to 4,3
    fig1, axes = plt.subplots(6, 3, figsize=(28, 30))  # Changed from 4,3 to 6,3

    # Summary statistics box
    summary_text = f"""
ENSEMBLE UNCERTAINTY ON IMPOSSIBLE TASKS

Model: {model_path}
Dataset: {dataset_name}

Question: Does ensemble disagreement correlate with task impossibility?
Method: Remove target objects via inpainting to create impossible tasks

Overall Results (N={len(orig_all)}):
- Mean uncertainty increase: {np.mean(uncertainty_differences):.2f} ± {np.std(uncertainty_differences):.2f}
- Trajectories with higher uncertainty: {np.mean(uncertainty_differences > 0)*100:.1f}%

Bridge Dataset (N={np.sum(bridge_mask)}):
- Avg. Uncertainty (orig -> inpaint): {np.mean(orig_all[bridge_mask]):.2f} → {np.mean(inpaint_all[bridge_mask]):.2f}
- Relative increase: {((np.mean(inpaint_all[bridge_mask])/np.mean(orig_all[bridge_mask])-1)*100):.0f}%

Fractal Dataset (N={np.sum(fractal_mask)}):
- Avg. Uncertainty (orig -> inpaint): {np.mean(orig_all[fractal_mask]):.2f} → {np.mean(inpaint_all[fractal_mask]):.2f}  
- Relative increase: {((np.mean(inpaint_all[fractal_mask])/np.mean(orig_all[fractal_mask])-1)*100):.0f}%
    """.strip()
    
    # Place summary in top-left, spanning 2 columns
    axes[0,0].text(0.05, 0.95, summary_text, transform=axes[0,0].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[0,0].axis('off')
    axes[0,1].axis('off')
    
    # Scatter plot (top-right)
    bridge_points = axes[0,2].scatter(orig_all[bridge_mask], inpaint_all[bridge_mask], 
                                    alpha=0.7, c='#2E86C1', s=60, edgecolors='white', linewidth=0.5, label='Bridge')
    fractal_points = axes[0,2].scatter(orig_all[fractal_mask], inpaint_all[fractal_mask], 
                                    alpha=0.7, c='#E74C3C', s=60, edgecolors='white', linewidth=0.5, label='Fractal')
    max_val = max(orig_all.max(), inpaint_all.max())
    axes[0,2].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Uncertainty')
    axes[0,2].set_xlabel('Original Task Uncertainty', fontweight='bold')
    axes[0,2].set_ylabel('Impossible Task Uncertainty', fontweight='bold') 
    axes[0,2].set_title('Uncertainty: Possible vs Impossible Tasks', fontweight='bold')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)


    # Dataset-specific uncertainty_differences (bottom-left)
    bridge_diff = uncertainty_differences[bridge_mask]
    fractal_diff = uncertainty_differences[fractal_mask]
    
    axes[1,0].hist(bridge_diff, bins=20, alpha=0.7, label='Bridge', color='#2E86C1', edgecolor='black')
    axes[1,0].hist(fractal_diff, bins=20, alpha=0.7, label='Fractal', color='#E74C3C', edgecolor='black')
    axes[1,0].axvline(0, color='black', linestyle='--', label='No difference')
    axes[1,0].set_xlabel('Uncertainty Increase (Impossible - Possible)', fontweight='bold')
    axes[1,0].set_ylabel('Count', fontweight='bold')
    axes[1,0].set_title('Uncertainty Increases by Dataset', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Dataset comparison bars (bottom-middle)
    datasets_list = ['Bridge', 'Fractal']
    means = [np.mean(bridge_diff), np.mean(fractal_diff)]
    stds = [np.std(bridge_diff), np.std(fractal_diff)]
    colors_bar = ['#2E86C1', '#E74C3C']
    
    bars = axes[1,1].bar(datasets_list, means, yerr=stds, capsize=5, 
                        color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1)
    axes[1,1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_ylabel('Mean Uncertainty Increase', fontweight='bold')
    axes[1,1].set_title('Mean Uncertainty Increase by Dataset', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                      f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Box plot comparison (bottom-right)
    data_to_plot = [orig_all[bridge_mask], inpaint_all[bridge_mask], 
                    orig_all[fractal_mask], inpaint_all[fractal_mask]]
    labels = ['Bridge\nPossible', 'Bridge\nImpossible', 'Fractal\nPossible', 'Fractal\nImpossible']
    
    axes[1,2].boxplot(data_to_plot, labels=labels)
    axes[1,2].set_ylabel('Uncertainty', fontweight='bold')
    axes[1,2].set_title('Uncertainty Distributions by Task Type', fontweight='bold')
    axes[1,2].tick_params(axis='x', rotation=45)
    axes[1,2].grid(True, alpha=0.3)
    
    # Q-value histograms by dataset (row 3, left)
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    gs_sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=axes[2,0].get_subplotspec(), hspace=0.3, wspace=0.3)
    axes[2,0].remove()

    # Create subplots with shared x-axis within columns
    ax_bridge_poss = fig1.add_subplot(gs_sub[0, 0])
    ax_bridge_imp = fig1.add_subplot(gs_sub[1, 0], sharex=ax_bridge_poss)  # Share with bridge_poss
    ax_fractal_poss = fig1.add_subplot(gs_sub[0, 1])
    ax_fractal_imp = fig1.add_subplot(gs_sub[1, 1], sharex=ax_fractal_poss)  # Share with fractal_poss

    # Your histogram plotting code remains the same
    ax_bridge_poss.hist(orig_q_means[bridge_mask], bins=15, alpha=0.8, color='#2E86C1', edgecolor='black')
    ax_bridge_poss.set_title('Bridge Possible', fontsize=10, fontweight='bold')
    ax_bridge_poss.set_ylabel('Count', fontsize=9)
    ax_bridge_poss.grid(True, alpha=0.3)

    ax_bridge_imp.hist(inpaint_q_means[bridge_mask], bins=15, alpha=0.8, color='#85C1E9', edgecolor='black')
    ax_bridge_imp.set_title('Bridge Impossible', fontsize=10, fontweight='bold')
    ax_bridge_imp.set_xlabel('Mean Q-Value', fontsize=9)  # Only bottom plot needs xlabel
    ax_bridge_imp.set_ylabel('Count', fontsize=9)
    ax_bridge_imp.grid(True, alpha=0.3)

    ax_fractal_poss.hist(orig_q_means[fractal_mask], bins=15, alpha=0.8, color='#E74C3C', edgecolor='black')
    ax_fractal_poss.set_title('Fractal Possible', fontsize=10, fontweight='bold')
    ax_fractal_poss.set_ylabel('Count', fontsize=9)
    ax_fractal_poss.grid(True, alpha=0.3)

    ax_fractal_imp.hist(inpaint_q_means[fractal_mask], bins=15, alpha=0.8, color='#F1948A', edgecolor='black')
    ax_fractal_imp.set_title('Fractal Impossible', fontsize=10, fontweight='bold')
    ax_fractal_imp.set_xlabel('Mean Q-Value', fontsize=9)  # Only bottom plot needs xlabel
    ax_fractal_imp.grid(True, alpha=0.3)

    fig1.text(0.165, 0.47, 'Q-Value Distributions by Dataset & Task Type',
            fontsize=14, fontweight='bold', ha='center')

    bridge_q_points = axes[2,1].scatter(orig_q_means[bridge_mask], inpaint_q_means[bridge_mask], 
                                    alpha=0.7, c='#2E86C1', s=60, edgecolors='white', linewidth=0.5, label='Bridge')
    fractal_q_points = axes[2,1].scatter(orig_q_means[fractal_mask], inpaint_q_means[fractal_mask], 
                                        alpha=0.7, c='#E74C3C', s=60, edgecolors='white', linewidth=0.5, label='Fractal')
    min_q = min(orig_q_means.min(), inpaint_q_means.min())
    max_q = max(orig_q_means.max(), inpaint_q_means.max())
    axes[2,1].plot([min_q, max_q], [min_q, max_q], 'k--', alpha=0.5, linewidth=2, label='Equal Q-Values')
    axes[2,1].set_xlabel('Original Task Mean Q-Value', fontweight='bold')
    axes[2,1].set_ylabel('Impossible Task Mean Q-Value', fontweight='bold')
    axes[2,1].set_title('Q-Values: Possible vs Impossible Tasks', fontweight='bold')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)

    # Q-value range (max-min) comparison (row 3, right)
    orig_q_range = orig_q_maxes - orig_q_mins
    inpaint_q_range = inpaint_q_maxes - inpaint_q_mins
    
    bridge_orig_range = orig_q_range[bridge_mask]
    bridge_inpaint_range = inpaint_q_range[bridge_mask]
    fractal_orig_range = orig_q_range[fractal_mask]
    fractal_inpaint_range = inpaint_q_range[fractal_mask]
    
    x_pos = np.arange(4)
    ranges_means = [np.mean(bridge_orig_range), np.mean(bridge_inpaint_range),
                   np.mean(fractal_orig_range), np.mean(fractal_inpaint_range)]
    ranges_stds = [np.std(bridge_orig_range), np.std(bridge_inpaint_range),
                  np.std(fractal_orig_range), np.std(fractal_inpaint_range)]
    range_colors = ['#2E86C1', '#85C1E9', '#E74C3C', '#F1948A']
    
    bars = axes[2,2].bar(x_pos, ranges_means, yerr=ranges_stds, capsize=5,
                        color=range_colors, alpha=0.7, edgecolor='black', linewidth=1)
    axes[2,2].set_xticks(x_pos)
    axes[2,2].set_xticklabels(['Bridge\nPossible', 'Bridge\nImpossible', 
                              'Fractal\nPossible', 'Fractal\nImpossible'], rotation=0)
    axes[2,2].set_ylabel('Q-Value Range (Max - Min)', fontweight='bold')
    axes[2,2].set_title('Ensemble Q-Value Range by Task Type', fontweight='bold')
    axes[2,2].grid(True, alpha=0.3)
    
    # Row 4: Q-value changes and correlations
    
    # Q-value changes histogram (row 4, left)
    q_mean_diff = inpaint_q_means - orig_q_means
    bridge_q_diff = q_mean_diff[bridge_mask]
    fractal_q_diff = q_mean_diff[fractal_mask]
    
    axes[3,0].hist(bridge_q_diff, bins=20, alpha=0.7, label='Bridge', color='#2E86C1', edgecolor='black')
    axes[3,0].hist(fractal_q_diff, bins=20, alpha=0.7, label='Fractal', color='#E74C3C', edgecolor='black')
    axes[3,0].axvline(0, color='black', linestyle='--', label='No change')
    axes[3,0].set_xlabel('Q-Value Change (Impossible - Possible)', fontweight='bold')
    axes[3,0].set_ylabel('Count', fontweight='bold')
    axes[3,0].set_title('Q-Value Changes by Dataset', fontweight='bold')
    axes[3,0].legend()
    axes[3,0].grid(True, alpha=0.3)
    
    # Correlation: Uncertainty vs Q-value changes (row 4, middle)
    bridge_corr_points = axes[3,1].scatter(uncertainty_differences[bridge_mask], q_mean_diff[bridge_mask], 
                                        alpha=0.7, c='#2E86C1', s=60, edgecolors='white', linewidth=0.5, label='Bridge')
    fractal_corr_points = axes[3,1].scatter(uncertainty_differences[fractal_mask], q_mean_diff[fractal_mask], 
                                        alpha=0.7, c='#E74C3C', s=60, edgecolors='white', linewidth=0.5, label='Fractal')
    axes[3,1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[3,1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[3,1].set_xlabel('Uncertainty Change (Impossible - Possible)\n[Std of Q-values over ensemble]', fontweight='bold')
    axes[3,1].set_ylabel('Q-Value Change (Impossible - Possible)', fontweight='bold')
    axes[3,1].set_title('Uncertainty vs Q-Value Changes', fontweight='bold')
    axes[3,1].legend()
    axes[3,1].grid(True, alpha=0.3)

    q_stats_text = f"""
Q-VALUE ANALYSIS SUMMARY

Bridge Dataset:
• Possible Tasks Mean Q: {np.mean(orig_q_means[bridge_mask]):.2f} ± {np.std(orig_q_means[bridge_mask]):.2f}
• Impossible Tasks Mean Q: {np.mean(inpaint_q_means[bridge_mask]):.2f} ± {np.std(inpaint_q_means[bridge_mask]):.2f}
• Mean Q-value change: {np.mean(bridge_q_diff):.2f} ± {np.std(bridge_q_diff):.2f}
• Mean trajectory length: {np.mean(traj_lengths[bridge_mask]):.1f} ± {np.std(traj_lengths[bridge_mask]):.1f}

Fractal Dataset:
• Possible Tasks Mean Q: {np.mean(orig_q_means[fractal_mask]):.2f} ± {np.std(orig_q_means[fractal_mask]):.2f}
• Impossible Tasks Mean Q: {np.mean(inpaint_q_means[fractal_mask]):.2f} ± {np.std(inpaint_q_means[fractal_mask]):.2f}
• Mean Q-value change: {np.mean(fractal_q_diff):.2f} ± {np.std(fractal_q_diff):.2f}
• Mean trajectory length: {np.mean(traj_lengths[fractal_mask]):.1f} ± {np.std(traj_lengths[fractal_mask]):.1f}

Overall:
• Tasks with decreased Q-values: {np.mean(q_mean_diff < 0)*100:.1f}%
• Tasks with increased Q-value range: {np.mean((inpaint_q_range - orig_q_range) > 0)*100:.1f}%
• Correlation (Uncertainty ↔ Q-change): {np.corrcoef(uncertainty_differences, q_mean_diff)[0,1]:.3f}
• Mean trajectory length: {np.mean(traj_lengths):.1f} ± {np.std(traj_lengths):.1f}
    """.strip()
    
    axes[3,2].text(0.05, 0.95, q_stats_text, transform=axes[3,2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    axes[3,2].axis('off')
    
    # ============ ROW 5: TRAJECTORY LENGTH ANALYSIS ============
    # Length distribution by dataset (row 5, left)
    # Remove outliers for following plots
    length_percentiles = np.percentile(traj_lengths, [2, 98])
    valid_mask = (traj_lengths >= length_percentiles[0]) & (traj_lengths <= length_percentiles[1])

    # Q-value summary statistics (row 4, right)
    ground_truth_q_vals = np.array([calculate_ground_truth_q(length) for length in traj_lengths])
    
    # (Q_inpainted - Q_original) / |Q_original|
    relative_q_changes = (inpaint_q_means - orig_q_means) / np.abs(orig_q_means)
    
    # Length-normalized Q-values
    orig_q_normalized = orig_q_means / traj_lengths
    inpaint_q_normalized = inpaint_q_means / traj_lengths
    
    # Ground truth normalized changes
    gt_normalized_orig = orig_q_means / np.abs(ground_truth_q_vals)
    gt_normalized_inpaint = inpaint_q_means / np.abs(ground_truth_q_vals)
    gt_relative_changes = (gt_normalized_inpaint - gt_normalized_orig)

    # Filter for length-based analysis (create new variables, don't mutate originals)
    traj_lengths_f = traj_lengths[valid_mask]
    bridge_mask_f = bridge_mask[valid_mask]
    fractal_mask_f = fractal_mask[valid_mask]
    orig_q_means_f = orig_q_means[valid_mask]
    inpaint_q_means_f = inpaint_q_means[valid_mask]
    orig_q_normalized_f = orig_q_normalized[valid_mask]
    inpaint_q_normalized_f = inpaint_q_normalized[valid_mask]
    ground_truth_q_vals_f = ground_truth_q_vals[valid_mask]
    relative_q_changes_f = relative_q_changes[valid_mask]
    gt_normalized_orig_f = gt_normalized_orig[valid_mask]
    gt_normalized_inpaint_f = gt_normalized_inpaint[valid_mask]
    gt_relative_changes_f = gt_relative_changes[valid_mask]

    bridge_lengths = traj_lengths_f[bridge_mask_f]
    fractal_lengths = traj_lengths_f[fractal_mask_f]
    
    axes[4,0].hist(bridge_lengths, bins=15, alpha=0.7, label='Bridge', color='#2E86C1', edgecolor='black')
    axes[4,0].hist(fractal_lengths, bins=15, alpha=0.7, label='Fractal', color='#E74C3C', edgecolor='black')
    axes[4,0].set_xlabel('Trajectory Length', fontweight='bold')
    axes[4,0].set_ylabel('Count', fontweight='bold')
    axes[4,0].set_title('Trajectory Length Distribution by Dataset', fontweight='bold')
    axes[4,0].legend()
    axes[4,0].grid(True, alpha=0.3)
    
    # Length-normalized Q-values scatter (row 5, middle)
    bridge_norm_points = axes[4,1].scatter(orig_q_normalized_f[bridge_mask_f], inpaint_q_normalized_f[bridge_mask_f],
                                        alpha=0.7, c='#2E86C1', s=60, edgecolors='white', linewidth=0.5, label='Bridge')
    fractal_norm_points = axes[4,1].scatter(orig_q_normalized_f[fractal_mask_f], inpaint_q_normalized_f[fractal_mask_f],
                                        alpha=0.7, c='#E74C3C', s=60, edgecolors='white', linewidth=0.5, label='Fractal')
    min_norm_q = min(orig_q_normalized_f.min(), inpaint_q_normalized_f.min())
    max_norm_q = max(orig_q_normalized_f.max(), inpaint_q_normalized_f.max())
    axes[4,1].plot([min_norm_q, max_norm_q], [min_norm_q, max_norm_q], 'k--', alpha=0.5, linewidth=2, label='Equal Q-Values')
    axes[4,1].set_xlabel('Original Task Q-Value / Length', fontweight='bold')
    axes[4,1].set_ylabel('Impossible Task Q-Value / Length', fontweight='bold')
    axes[4,1].set_title('Length-Normalized Q-Values: Possible vs Impossible', fontweight='bold')
    axes[4,1].legend()
    axes[4,1].grid(True, alpha=0.3)
    
    # Relative Q-value changes histogram (row 5, right)
    bridge_rel_changes = relative_q_changes_f[bridge_mask_f]
    fractal_rel_changes = relative_q_changes_f[fractal_mask_f]
    
    axes[4,2].hist(bridge_rel_changes, bins=20, alpha=0.7, label='Bridge', color='#2E86C1', edgecolor='black')
    axes[4,2].hist(fractal_rel_changes, bins=20, alpha=0.7, label='Fractal', color='#E74C3C', edgecolor='black')
    axes[4,2].axvline(0, color='black', linestyle='--', label='No change')
    axes[4,2].set_xlabel('Relative Q-Value Change\n(Q_impossible - Q_possible) / |Q_possible|', fontweight='bold')
    axes[4,2].set_ylabel('Count', fontweight='bold')
    axes[4,2].set_title('Relative Q-Value Changes by Dataset', fontweight='bold')
    axes[4,2].legend()
    axes[4,2].grid(True, alpha=0.3)
    
    # ============ ROW 6: GROUND TRUTH NORMALIZED ANALYSIS ============
    
    # Ground truth Q-values vs trajectory length (row 6, left)
    bridge_gt_points = axes[5,0].scatter(traj_lengths_f[bridge_mask_f], ground_truth_q_vals_f[bridge_mask_f],
                                        alpha=0.7, c='#2E86C1', s=60, edgecolors='white', linewidth=0.5, label='Bridge')
    fractal_gt_points = axes[5,0].scatter(traj_lengths_f[fractal_mask_f], ground_truth_q_vals_f[fractal_mask_f],
                                        alpha=0.7, c='#E74C3C', s=60, edgecolors='white', linewidth=0.5, label='Fractal')
    axes[5,0].set_xlabel('Trajectory Length', fontweight='bold')
    axes[5,0].set_ylabel('Ground Truth Q-Value\n(Discounted Reward-to-Go)', fontweight='bold')
    axes[5,0].set_title('Ground Truth Q-Values vs Trajectory Length', fontweight='bold')
    axes[5,0].legend()
    axes[5,0].grid(True, alpha=0.3)
    
    # Ground truth normalized Q-values scatter (row 6, middle)
    bridge_gt_norm_points = axes[5,1].scatter(gt_normalized_orig_f[bridge_mask_f], gt_normalized_inpaint_f[bridge_mask_f],
                                            alpha=0.7, c='#2E86C1', s=60, edgecolors='white', linewidth=0.5, label='Bridge')
    fractal_gt_norm_points = axes[5,1].scatter(gt_normalized_orig_f[fractal_mask_f], gt_normalized_inpaint_f[fractal_mask_f],
                                            alpha=0.7, c='#E74C3C', s=60, edgecolors='white', linewidth=0.5, label='Fractal')
    min_gt_norm = min(gt_normalized_orig_f.min(), gt_normalized_inpaint_f.min())
    max_gt_norm = max(gt_normalized_orig_f.max(), gt_normalized_inpaint_f.max())
    axes[5,1].plot([min_gt_norm, max_gt_norm], [min_gt_norm, max_gt_norm], 'k--', alpha=0.5, linewidth=2, label='Equal Normalized Q')
    axes[5,1].set_xlabel('Original Q / |Ground Truth Q|', fontweight='bold')
    axes[5,1].set_ylabel('Impossible Q / |Ground Truth Q|', fontweight='bold')
    axes[5,1].set_title('Ground Truth Normalized Q-Values', fontweight='bold')
    axes[5,1].legend()
    axes[5,1].grid(True, alpha=0.3)
    
    # Ground truth normalized changes histogram (row 6, right)
    bridge_gt_changes = gt_relative_changes_f[bridge_mask_f]
    fractal_gt_changes = gt_relative_changes_f[fractal_mask_f]
    
    axes[5,2].hist(bridge_gt_changes, bins=20, alpha=0.7, label='Bridge', color='#2E86C1', edgecolor='black')
    axes[5,2].hist(fractal_gt_changes, bins=20, alpha=0.7, label='Fractal', color='#E74C3C', edgecolor='black')
    axes[5,2].axvline(0, color='black', linestyle='--', label='No change')
    axes[5,2].set_xlabel('Ground Truth Normalized Change\n(Q_impossible/|Q_gt| - Q_possible/|Q_gt|)', fontweight='bold')
    axes[5,2].set_ylabel('Count', fontweight='bold')
    axes[5,2].set_title('Ground Truth Normalized Q-Value Changes', fontweight='bold')
    axes[5,2].legend()
    axes[5,2].grid(True, alpha=0.3)

    fig1.suptitle('Ensemble Uncertainty: Response to Task Impossibility', 
                  fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.6)
    plt.subplots_adjust(top=0.92)
    # extract model id = timestamp:
    model_id = extract_timestamp_from_path(model_path)

    summary_path = plots_dir / f"ensemble_uncertainty_summary_{dataset_name}_ensemble_{model_id}.pdf"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    logging.info(f"Saved summary plot: {summary_path}")
    plt.show()
    
    # ============ PLOT 2: EXTREME CASES ============
    def analyze_dataset_extremes_both(dataset_name, n_examples):
        """Find extreme cases for a specific dataset - both increases and decreases."""
        dataset_mask = np.array([d == dataset_name for d in datasets])
        dataset_indices = np.where(dataset_mask)[0]
        dataset_diffs = uncertainty_differences[dataset_mask]
        
        # Highest increases
        highest_idx = np.argsort(dataset_diffs)[-min(n_examples, len(dataset_diffs)):][::-1]
        highest_global_idx = dataset_indices[highest_idx]
        
        # Lowest (most negative) - biggest decreases
        lowest_idx = np.argsort(dataset_diffs)[:min(n_examples, len(dataset_diffs))]
        lowest_global_idx = dataset_indices[lowest_idx]
        
        return highest_global_idx, lowest_global_idx
    
    def plot_trajectory_row(traj, orig_unc, inp_unc, row_axes, title_prefix, dataset_name):
        """Plot a trajectory in a row of 5 subplots."""
        # Color code by dataset
        border_color = "#2E86C1" if dataset_name == "bridge" else "#E74C3C"
        
        # Plot images
        images = [
            (traj['first_image'], 'First\nPossible'),
            (traj['first_image_inpainted'], 'First\nImpossible'),
            (traj['last_image'], 'Last\nPossible'),
            (traj['last_image_inpainted'], 'Last\nImpossible')
        ]
        
        for i, (img, title) in enumerate(images):
            row_axes[i].imshow(img)
            row_axes[i].set_title(title, fontsize=11, fontweight='bold')
            row_axes[i].axis('off')
            
            # Add border for impossible task images
            if 'impossible' in title.lower():
                for spine in row_axes[i].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(4)
                    spine.set_visible(True)
        
        # Language prompt
        language_prompt = traj.get('language', 'No language instruction available')
        wrapped_prompt = '\n'.join(wrap(language_prompt, width=30))
        
        bg_color = "lightblue" if dataset_name == "bridge" else "lightcoral"
        
        row_axes[4].text(0.5, 0.5, wrapped_prompt,
                        ha='center', va='center', transform=row_axes[4].transAxes,
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, edgecolor=border_color))
        row_axes[4].set_title('Language Prompt', fontsize=11, fontweight='bold')
        row_axes[4].axis('off')
        
        # Add uncertainty info as subplot title
        unc_diff = inp_unc - orig_unc
        direction = "↑" if unc_diff > 0 else "↓"
        row_title = f'{title_prefix}: {orig_unc:.2f} → \n {inp_unc:.2f} ({direction}{abs(unc_diff):.2f})'
        
        return row_title
    
    # Find extreme cases - both increases and decreases
    bridge_high, bridge_low = analyze_dataset_extremes_both('bridge', n_examples)
    fractal_high, fractal_low = analyze_dataset_extremes_both('fractal', n_examples)
    
    # Create figure for extreme cases - all in one plot with separator
    total_rows = len(bridge_high) + len(fractal_high) + 1 + len(bridge_low) + len(fractal_low)  # +1 for separator
    fig2, axes = plt.subplots(total_rows, 5, figsize=(24, 3.2 * total_rows))
    
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    
    current_row = 0
    row_titles = []
    
    # Plot Bridge examples - highest increases
    for i, global_idx in enumerate(bridge_high):
        traj = trajectories[global_idx]
        orig_unc = orig_all[global_idx]
        inp_unc = inpaint_all[global_idx]
        
        row_title = plot_trajectory_row(traj, orig_unc, inp_unc, axes[current_row], 
                                       f"Bridge Increase #{i+1}", "bridge")
        row_titles.append(row_title)
        current_row += 1
    
    # Plot Fractal examples - highest increases
    for i, global_idx in enumerate(fractal_high):
        traj = trajectories[global_idx]
        orig_unc = orig_all[global_idx]
        inp_unc = inpaint_all[global_idx]
        
        row_title = plot_trajectory_row(traj, orig_unc, inp_unc, axes[current_row], 
                                       f"Fractal Increase #{i+1}", "fractal")
        row_titles.append(row_title)
        current_row += 1
    
    # Add separator row
    for j in range(5):
        axes[current_row, j].axis('off')
        if j == 2:  # Middle subplot
            axes[current_row, j].text(0.5, 0.5, '━' * 50 + '\nUNCERTAINTY DECREASES\n' + '━' * 50,
                                     ha='center', va='center', transform=axes[current_row, j].transAxes,
                                     fontsize=14, fontweight='bold', color='red')
    row_titles.append("SEPARATOR")
    current_row += 1
    
    # Plot Bridge examples - largest decreases
    for i, global_idx in enumerate(bridge_low):
        traj = trajectories[global_idx]
        orig_unc = orig_all[global_idx]
        inp_unc = inpaint_all[global_idx]
        
        row_title = plot_trajectory_row(traj, orig_unc, inp_unc, axes[current_row], 
                                       f"Bridge Decrease #{i+1}", "bridge")
        row_titles.append(row_title)
        current_row += 1
    
    # Plot Fractal examples - largest decreases
    for i, global_idx in enumerate(fractal_low):
        traj = trajectories[global_idx]
        orig_unc = orig_all[global_idx]
        inp_unc = inpaint_all[global_idx]
        
        row_title = plot_trajectory_row(traj, orig_unc, inp_unc, axes[current_row], 
                                       f"Fractal Decrease #{i+1}", "fractal")
        row_titles.append(row_title)
        current_row += 1
    
    # Add row titles on the left (skip separator)
    for i, title in enumerate(row_titles):
        if title != "SEPARATOR":
            fig2.text(0.02, 0.95 - (i / total_rows), title, fontsize=12, fontweight='bold', 
                     rotation=0, va='center')
    
    fig2.suptitle('Extreme Cases: Largest Uncertainty Changes on Impossible Tasks', 
                  fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.2)
    plt.subplots_adjust(left=0.10, top=0.96, wspace=0.1)
    # extract model id: already done above?
    # model_id = extract_timestamp_from_path(model_path)
    
    samples_path = plots_dir / f"ensemble_uncertainty_samples_{dataset_name}_ensemble_{model_id}.pdf"
    plt.savefig(samples_path, dpi=300, bbox_inches='tight', facecolor='white')
    logging.info(f"Saved samples plot: {samples_path}")

    plt.show()
    
    # ============ STATISTICAL SUMMARY TABLE ============
    summary_stats = {
        'Dataset': ['Overall', 'Bridge', 'Fractal'],
        'N': [len(orig_all), np.sum(bridge_mask), np.sum(fractal_mask)],
        'Possible_Task_Mean': [np.mean(orig_all), np.mean(orig_all[bridge_mask]), np.mean(orig_all[fractal_mask])],
        'Impossible_Task_Mean': [np.mean(inpaint_all), np.mean(inpaint_all[bridge_mask]), np.mean(inpaint_all[fractal_mask])],
        'Uncertainty_Increase': [np.mean(uncertainty_differences), np.mean(uncertainty_differences[bridge_mask]), np.mean(uncertainty_differences[fractal_mask])],
        'Percent_Higher': [np.mean(uncertainty_differences > 0)*100, np.mean(uncertainty_differences[bridge_mask] > 0)*100, np.mean(uncertainty_differences[fractal_mask] > 0)*100],
        'Q_Possible_Mean': [np.mean(orig_q_means), np.mean(orig_q_means[bridge_mask]), np.mean(orig_q_means[fractal_mask])],
        'Q_Impossible_Mean': [np.mean(inpaint_q_means), np.mean(inpaint_q_means[bridge_mask]), np.mean(inpaint_q_means[fractal_mask])],
        'Q_Change_Mean': [np.mean(q_mean_diff), np.mean(bridge_q_diff), np.mean(fractal_q_diff)],
        'Q_Decreased_Percent': [np.mean(q_mean_diff < 0)*100, np.mean(bridge_q_diff < 0)*100, np.mean(fractal_q_diff < 0)*100]
    }
    summary_stats.update({
        'Avg_Traj_Length': [np.mean(traj_lengths_f), np.mean(traj_lengths_f[bridge_mask_f]), np.mean(traj_lengths_f[fractal_mask_f])],
        'Length_Norm_Q_Orig': [np.mean(orig_q_normalized_f), np.mean(orig_q_normalized_f[bridge_mask_f]), np.mean(orig_q_normalized_f[fractal_mask_f])],
        'Length_Norm_Q_Inpaint': [np.mean(inpaint_q_normalized_f), np.mean(inpaint_q_normalized_f[bridge_mask_f]), np.mean(inpaint_q_normalized_f[fractal_mask_f])],
        'Relative_Q_Change_Mean': [np.mean(relative_q_changes_f), np.mean(bridge_rel_changes), np.mean(fractal_rel_changes)],
        'GT_Norm_Q_Orig': [np.mean(gt_normalized_orig_f), np.mean(gt_normalized_orig_f[bridge_mask_f]), np.mean(gt_normalized_orig_f[fractal_mask_f])],
        'GT_Norm_Q_Inpaint': [np.mean(gt_normalized_inpaint_f), np.mean(gt_normalized_inpaint_f[bridge_mask_f]), np.mean(gt_normalized_inpaint_f[fractal_mask_f])],
        'GT_Norm_Change_Mean': [np.mean(gt_relative_changes_f), np.mean(bridge_gt_changes), np.mean(fractal_gt_changes)]
    })

    # df = pd.DataFrame(summary_stats)
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY TABLE - UNCERTAINTY CHANGES")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print("="*80)
    # print(df.round(3).to_string(index=False))
    print("="*80)
    
    # return df

# summary_df = analyze_extreme_cases(filtered_trajectories, inference_results, n_examples=3,
#                                  model_path=RESUME_PATH, dataset_name="bridge_fractal_n300_with_inpainted")


def run_octo_scratch_inference_on_orig_and_inpaintings(
    trajectories, model, text_processor,
    model_path, dataset_name, output_path="experiments/results",
    force_recompute=False
):
    """Run Octo scratch model inference on original vs inpainted observations.

    The Octo model has a 'feasibility' head that outputs binary classification logits.
    This function processes trajectories and extracts P(feasible) for both original
    and inpainted images.

    Args:
        trajectories: List of trajectory dicts with first/last (inpainted) images
        model: OctoModel instance (from load_octo_scratch_model)
        text_processor: Text processor from model
        model_path: Path for metadata
        dataset_name: For caching
        output_path: Where to save HDF5 results
        force_recompute: Skip cache

    Returns:
        Dict with original_probs, inpainted_probs, trajectory_ids, failed_ids
    """
    # Log model's max_horizon from config (before cache check for visibility)
    model_max_horizon = model.config.get('model', {}).get('max_horizon', None)
    print(f"Model max_horizon from config: {model_max_horizon}")

    # Set up output path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_id = extract_timestamp_from_path(model_path)
    results_filename = f"inference_results_{dataset_name}_octo_scratch_{model_id}.hdf5"
    results_path = output_path / results_filename

    # Check cache
    if results_path.exists() and not force_recompute:
        logging.info(f"Loading cached results: {results_path}")
        return load_inference_results_hdf5(results_path)[0]

    logging.info(f"Computing Octo scratch inference (output: {results_path})")

    # Storage
    results = {
        'original_probs': [],
        'inpainted_probs': [],
        'trajectory_ids': [],
        'failed_ids': []
    }

    def create_batch_and_predict(traj, use_inpainted=False):
        """Create batch from trajectory and get P(feasible)."""
        # Select images
        if use_inpainted:
            images = jnp.array(np.stack([traj['first_image_inpainted'], traj['last_image_inpainted']]))
        else:
            images = jnp.array(np.stack([traj['first_image'], traj['last_image']]))

        # Add window dimension for single-frame inference: (2, H, W, C) -> (2, 1, H, W, C)
        # Note: Model trained with window_size=2 on consecutive frames, but first/last aren't consecutive
        # so we use window=1 to evaluate each frame independently
        images = images[:, None, ...]  # (batch=2, window=1, H, W, C)

        # Create observation batch
        batch_size = images.shape[0]
        timestep_pad_mask = jnp.ones((batch_size, 1), dtype=bool)  # (batch=2, window=1)
        observations = {
            'image_primary': images,
            'timestep_pad_mask': timestep_pad_mask,
            'pad_mask_dict': {'image_primary': jnp.ones((batch_size, 1), dtype=bool)}
        }

        # Debug: print shapes (only first trajectory)
        if traj.get('trajectory_id', 0) == 0:
            print(f"DEBUG: images.shape = {images.shape}")
            print(f"DEBUG: timestep_pad_mask.shape = {timestep_pad_mask.shape}, values = {timestep_pad_mask}")

        # Create task inputs - duplicate for batch_size=2 (first + last image)
        task_inputs = model.create_tasks(texts=[traj['language'], traj['language']])

        # Bind module and run forward pass
        bound_module = model.module.bind(
            {"params": model.params},
            rngs={"dropout": jax.random.PRNGKey(0)}
        )

        # Get transformer embeddings
        transformer_embeddings = bound_module.octo_transformer(
            observations,
            task_inputs,
            observations['timestep_pad_mask'],
            train=False
        )

        # Get feasibility head output - __call__ signature: (transformer_outputs, train)
        logits = bound_module.heads['feasibility'](transformer_embeddings, train=False)

        # Fail fast: verify logits are valid
        assert logits is not None, "Feasibility head returned None"
        assert logits.shape[-1] == 1, f"Expected logits shape (..., 1), got {logits.shape}"
        assert not jnp.any(jnp.isnan(logits)), "Logits contain NaN"

        # Convert to probabilities
        probs = jax.nn.sigmoid(logits)

        # Fail fast: verify probs are in valid range
        assert jnp.all((probs >= 0) & (probs <= 1)), f"Probs out of range: min={probs.min()}, max={probs.max()}"

        mean_prob = float(jnp.mean(probs))

        return mean_prob

    # Run inference on all trajectories
    for traj in tqdm(trajectories, desc="Octo scratch inference"):
        traj_id = traj['trajectory_id']  # fail fast if missing
        results['trajectory_ids'].append(traj_id)

        try:
            # Original
            orig_prob = create_batch_and_predict(traj, use_inpainted=False)
            results['original_probs'].append(orig_prob)

            # Inpainted
            inp_prob = create_batch_and_predict(traj, use_inpainted=True)
            results['inpainted_probs'].append(inp_prob)

        except Exception as e:
            logging.error(f"Failed on trajectory {traj_id}: {e}")
            results['original_probs'].append(None)
            results['inpainted_probs'].append(None)
            results['failed_ids'].append(traj_id)

    # Save results
    metadata = {
        'model_path': model_path,
        'model_id': model_id,
        'model_type': 'octo_scratch',
        'dataset_name': dataset_name,
        'n_trajectories': len(trajectories),
        'n_successful': len([x for x in results['original_probs'] if x is not None]),
        'timestamp': datetime.now().isoformat()
    }

    save_inference_results_hdf5(results, metadata, results_path)
    logging.info(f"Saved results: {results_path}")

    return results


# Alias: works for both scratch-trained and finetuned Octo models with feasibility head
run_octo_inference_on_orig_and_inpaintings = run_octo_scratch_inference_on_orig_and_inpaintings
