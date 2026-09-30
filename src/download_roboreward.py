"""Download RoboReward dataset from HuggingFace.

Usage:
    python download_roboreward.py [--cache_dir PATH] [--streaming]

Examples:
    # Download to default location (~/datasets/roboreward)
    python download_roboreward.py

    # Download to custom location
    python download_roboreward.py --cache_dir /path/to/datasets/roboreward

    # Stream only (no download)
    python download_roboreward.py --streaming
"""

import argparse
import os
from pathlib import Path


def _disable_video_decoding(dataset):
    """Cast video columns to decode=False to avoid torchcodec/FFmpeg dependency."""
    from datasets import Video
    for col_name, feature in dataset['train'].features.items():
        if isinstance(feature, Video):
            print(f"  Disabling video decoding for column: {col_name}")
            for split in dataset:
                dataset[split] = dataset[split].cast_column(col_name, Video(decode=False))
    return dataset


def download_roboreward(cache_dir: str = None, streaming: bool = False):
    """Download RoboReward dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: HuggingFace datasets library not found!")
        print("Install: pip install datasets")
        return

    # Set default cache directory
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/datasets/roboreward")

    cache_dir = os.path.abspath(cache_dir)

    if streaming:
        print("Loading RoboReward in streaming mode (no download)...")
        dataset = load_dataset("teetone/RoboReward", streaming=True)

        print("\n===== Streaming Preview =====")
        for split in ['train', 'validation', 'test']:
            print(f"\n{split.upper()} split:")
            split_data = dataset[split]
            for i, example in enumerate(split_data.take(3)):
                task_preview = example['task'][:60] + "..." if len(example['task']) > 60 else example['task']
                print(f"  Example {i}: Reward={example['reward']}, Task=\"{task_preview}\"")
    else:
        print(f"Downloading RoboReward to: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

        # Download dataset
        dataset = load_dataset(
            "teetone/RoboReward",
            cache_dir=cache_dir,
        )

        # Disable video decoding (avoids torchcodec/FFmpeg dependency)
        dataset = _disable_video_decoding(dataset)

        # Print statistics
        print("\n===== Dataset Statistics =====")
        total_size = 0
        for split, data in dataset.items():
            print(f"{split:12s}: {len(data):6d} examples")
            total_size += len(data)
        print(f"{'TOTAL':12s}: {total_size:6d} examples")

        # Print example structure
        print("\n===== Example Structure =====")
        example = dataset['train'][0]
        for key, value in example.items():
            print(f"  {key:20s}: {type(value).__name__}")

        # Reward distribution
        print("\n===== Reward Distribution (train split, first 1000) =====")
        rewards = [dataset['train'][i]['reward'] for i in range(min(1000, len(dataset['train'])))]
        from collections import Counter
        reward_counts = Counter(rewards)
        for reward in sorted(reward_counts.keys()):
            count = reward_counts[reward]
            pct = 100 * count / len(rewards)
            print(f"  Reward {reward}: {count:4d} ({pct:5.1f}%)")

        # Show sample examples
        print("\n===== Sample Examples =====")
        for i in range(min(5, len(dataset['train']))):
            ex = dataset['train'][i]
            task_preview = ex['task'][:70] + "..." if len(ex['task']) > 70 else ex['task']
            print(f"Example {i}: Reward={ex['reward']}, Task=\"{task_preview}\"")

        # Disk usage
        print(f"\n===== Download Complete =====")
        print(f"Dataset cached at: {cache_dir}")

        # Try to show disk usage
        try:
            import subprocess
            result = subprocess.run(['du', '-sh', cache_dir], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Disk usage: {result.stdout.strip().split()[0]}")
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RoboReward dataset")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache dataset (default: ~/datasets/roboreward)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset without downloading"
    )

    args = parser.parse_args()
    download_roboreward(args.cache_dir, args.streaming)
