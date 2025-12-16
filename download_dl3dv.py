#!/usr/bin/env python3
"""
Download DL3DV dataset videos from Hugging Face
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import argparse

# Configuration
DL3DV_REPO = "DL3DV/DL3DV-10K-Sample"  # Start with sample, can change to full dataset
OUTPUT_DIR = "/home/junjie/i2v/datasets/dl3dv"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")

def download_dl3dv_sample(num_videos=None):
    """
    Download DL3DV sample dataset from Hugging Face

    Args:
        num_videos: Number of videos to download (None = all)
    """
    print(f"Downloading DL3DV dataset to {OUTPUT_DIR}...")

    # Create output directory
    Path(VIDEO_DIR).mkdir(parents=True, exist_ok=True)

    try:
        # Download the dataset
        # For sample: downloads 11 scenes
        local_dir = snapshot_download(
            repo_id=DL3DV_REPO,
            repo_type="dataset",
            local_dir=OUTPUT_DIR,
            allow_patterns="*.mp4",  # Only download video files
        )

        print(f"✓ Dataset downloaded to: {local_dir}")

        # List downloaded videos
        videos = list(Path(local_dir).rglob("*.mp4"))
        print(f"✓ Found {len(videos)} videos")

        # Copy/move videos to a flat structure
        for i, video in enumerate(sorted(videos)[:num_videos] if num_videos else sorted(videos)):
            print(f"  - {video.name}")

        return videos

    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nNote: You may need to:")
        print("1. Accept the dataset terms on Hugging Face")
        print("2. Login with: huggingface-cli login")
        return []

def download_dl3dv_480p(num_videos=100):
    """
    Download DL3DV-480P subset (more efficient for processing)

    Args:
        num_videos: Number of videos to download
    """
    print(f"Downloading {num_videos} videos from DL3DV-480P...")

    # For 480P version, you would use the main dataset repo
    # This requires more setup and is larger
    repo_id = "DL3DV/DL3DV-ALL-480P"

    print(f"Note: For full dataset, please follow instructions at:")
    print(f"https://github.com/DL3DV-10K/Dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DL3DV dataset")
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of videos to download (default: all)")
    parser.add_argument("--dataset", choices=["sample", "480p", "4k"], default="sample",
                        help="Dataset version to download")

    args = parser.parse_args()

    if args.dataset == "sample":
        videos = download_dl3dv_sample(args.num_videos)
        print(f"\n✓ Downloaded {len(videos)} videos to {OUTPUT_DIR}")
    elif args.dataset == "480p":
        download_dl3dv_480p(args.num_videos)
    else:
        print("4K version requires manual download from Hugging Face")
        print("Visit: https://huggingface.co/datasets/DL3DV/DL3DV-ALL-4K")
