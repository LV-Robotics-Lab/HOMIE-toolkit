#!/usr/bin/env python3
"""
Download Xperience-10M datasets from HuggingFace Hub.

This script supports:
1. xperience-10m-sample: public sample episode, no auth required
2. xperience-10m: gated full dataset, download a single episode after auth
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

SAMPLE_FILES = [
    "README.md",
    "annotation.hdf5",
    "stereo_left.mp4",
    "stereo_right.mp4",
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
]


def ensure_huggingface_hub():
    """Ensure huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install huggingface-hub: {e}")
            return False


def _is_gated_access_error(exc):
    text = str(exc).lower()
    return "gated" in text or "401" in text or "403" in text or "authentication" in text


def download_dataset(dataset_id, local_path, allow_patterns=None):
    """Download a dataset from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download

        print(f"\n{'='*60}")
        print(f"Downloading: {dataset_id}")
        print(f"Target location: {local_path}")
        if allow_patterns:
            print("Patterns:")
            for pattern in allow_patterns:
                print(f"  - {pattern}")
        print(f"{'='*60}")

        # Create parent directory if it doesn't exist
        Path(local_path).mkdir(parents=True, exist_ok=True)

        # Download the dataset
        repo_path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=local_path,
            allow_patterns=allow_patterns,
        )

        print(f"✓ Successfully downloaded to: {repo_path}")
        return True

    except Exception as e:
        if _is_gated_access_error(e):
            print(f"✗ Access denied for {dataset_id}.")
            print("  This dataset requires Hugging Face auth and approved access.")
            print("  Run one of the following first:")
            print("    hf auth login")
            print("    export HF_TOKEN=...")
            return False
        print(f"✗ Error downloading {dataset_id}: {e}")
        return False


def download_sample_dataset(local_path):
    """Download the public sample episode."""
    print("\nThe sample dataset is public and does not require authorization.")
    return download_dataset(
        "ropedia-ai/xperience-10m-sample",
        local_path,
        allow_patterns=SAMPLE_FILES,
    )


def download_full_episode(local_path):
    """Download one episode from the gated full dataset."""
    print("\nThe full dataset is gated.")
    print("Download a single episode path such as: 003dcaf0-edba-4787-ada0-187d2748f684/ep1")
    episode_path = input("Episode path: ").strip().strip("/")
    if not episode_path:
        print("No episode path provided. Skipping full dataset download.")
        return False, None

    allow_patterns = [f"{episode_path}/*"]
    ok = download_dataset(
        "ropedia-ai/xperience-10m",
        local_path,
        allow_patterns=allow_patterns,
    )
    return ok, episode_path


def main():
    """Main function to download sample and/or full dataset."""

    # Check if huggingface_hub is available
    if not ensure_huggingface_hub():
        print("Please install huggingface-hub manually:")
        print("  pip install huggingface-hub")
        sys.exit(1)

    # Define datasets to download
    datasets = [
        {
            "id": "ropedia-ai/xperience-10m-sample",
            "path": str(DATA_ROOT / "xperience-10m-sample"),
            "description": "Public sample episode (no auth required)",
        },
        {
            "id": "ropedia-ai/xperience-10m",
            "path": str(DATA_ROOT / "xperience-10m"),
            "description": "Gated full dataset (download one episode after auth)",
        },
    ]

    print("\n" + "="*60)
    print("HOMIE-toolkit: Dataset Downloader")
    print("="*60)

    # Ask user which datasets to download
    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds['id']}")
        print(f"   Description: {ds['description']}")
        print(f"   Location: {ds['path']}")

    print("\nEnter the number of datasets to download (e.g., '1' or '1,2' for both):")
    user_input = input("Your choice: ").strip()

    if not user_input:
        print("No datasets selected. Exiting.")
        return

    try:
        selected_indices = [int(x.strip()) - 1 for x in user_input.split(",")]
        selected_datasets = [datasets[i] for i in selected_indices if 0 <= i < len(datasets)]
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    if not selected_datasets:
        print("No valid datasets selected. Exiting.")
        return

    # Download selected datasets
    successful = []
    failed = []

    for dataset_info in selected_datasets:
        print(f"\nDownloading {dataset_info['id']}...")
        if dataset_info["id"] == "ropedia-ai/xperience-10m-sample":
            if download_sample_dataset(dataset_info["path"]):
                successful.append(dataset_info["id"])
            else:
                failed.append(dataset_info["id"])
        else:
            ok, episode_path = download_full_episode(dataset_info["path"])
            if ok:
                successful.append(f"{dataset_info['id']} ({episode_path})")
            else:
                failed.append(dataset_info["id"])

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    if successful:
        print(f"\n✓ Successfully downloaded ({len(successful)}):")
        for ds in successful:
            print(f"  - {ds}")

    if failed:
        print(f"\n✗ Failed downloads ({len(failed)}):")
        for ds in failed:
            print(f"  - {ds}")

    if successful:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Public sample episode:")
        print(f"   .venv/bin/python examples/example_load_annotation.py --data_root {DATA_ROOT / 'xperience-10m-sample'}")
        print("\n2. Full dataset episode:")
        print(f"   .venv/bin/python examples/example_load_annotation.py --data_root {DATA_ROOT / 'xperience-10m' / '<session_id>' / 'epN'}")
        print("\n3. Visualize with Rerun:")
        print("   .venv/bin/python examples/example_visualize_rrd.py --data_root data/xperience-10m-sample --output_rrd visualization.rrd")
        print(f"   rerun {DATA_ROOT / 'xperience-10m-sample' / 'visualization.rrd'}")


if __name__ == "__main__":
    main()
