#!/usr/bin/env python3
"""
Download RealEstate10K dataset videos from YouTube
Based on: https://github.com/cashiwamochi/RealEstate10K_Downloader
"""
import os
import subprocess
from pathlib import Path
import requests
import tarfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
METADATA_URL = "https://storage.cloud.google.com/realestate10k-public-files/RealEstate10K.tar.gz"
OUTPUT_DIR = "/home/junjie/i2v/datasets/realestate10k"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

def download_metadata():
    """Download and extract RealEstate10K metadata (camera poses)"""
    print("Downloading RealEstate10K metadata...")

    Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)

    # Download tar.gz
    tar_path = os.path.join(OUTPUT_DIR, "RealEstate10K.tar.gz")

    if not os.path.exists(tar_path):
        response = requests.get(METADATA_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(tar_path, 'wb') as f, tqdm(
            desc="Downloading metadata",
            total=total_size,
            unit='B',
            unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"✓ Downloaded to {tar_path}")

    # Extract
    print("Extracting metadata...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(METADATA_DIR)

    print(f"✓ Extracted to {METADATA_DIR}")
    return METADATA_DIR

def parse_video_id(txt_file):
    """Extract YouTube video ID from metadata filename"""
    # Format: <youtube_id>.txt
    return Path(txt_file).stem

def download_youtube_video(video_id, output_path):
    """
    Download a single YouTube video using yt-dlp

    Args:
        video_id: YouTube video ID
        output_path: Where to save the video
    """
    try:
        # Use yt-dlp (better than youtube-dl)
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "-o", output_path,
            f"https://www.youtube.com/watch?v={video_id}",
            "--no-warnings",
            "--quiet"
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        return True, video_id

    except subprocess.CalledProcessError as e:
        return False, video_id
    except Exception as e:
        return False, video_id

def download_videos(num_videos=100, split="train", max_workers=4):
    """
    Download RealEstate10K videos from YouTube

    Args:
        num_videos: Number of videos to download
        split: 'train' or 'test'
        max_workers: Number of parallel downloads
    """
    print(f"Downloading {num_videos} videos from RealEstate10K ({split} split)...")

    Path(VIDEO_DIR).mkdir(parents=True, exist_ok=True)

    # Get metadata files
    split_dir = os.path.join(METADATA_DIR, split)
    if not os.path.exists(split_dir):
        print(f"✗ Metadata not found. Run download_metadata() first.")
        return

    txt_files = sorted(list(Path(split_dir).glob("*.txt")))[:num_videos]
    print(f"Found {len(txt_files)} metadata files")

    # Download videos in parallel
    successful = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for txt_file in txt_files:
            video_id = parse_video_id(txt_file)
            output_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

            # Skip if already downloaded
            if os.path.exists(output_path):
                print(f"  ✓ Already exists: {video_id}")
                successful += 1
                continue

            future = executor.submit(download_youtube_video, video_id, output_path)
            futures[future] = video_id

        # Progress bar
        with tqdm(total=len(futures), desc="Downloading videos") as pbar:
            for future in as_completed(futures):
                success, video_id = future.result()
                if success:
                    successful += 1
                    pbar.set_postfix({"success": successful, "failed": len(failed)})
                else:
                    failed.append(video_id)
                pbar.update(1)

    print(f"\n✓ Successfully downloaded: {successful} videos")
    if failed:
        print(f"✗ Failed to download: {len(failed)} videos")
        print(f"  Failed IDs: {', '.join(failed[:10])}" + ("..." if len(failed) > 10 else ""))

    return successful

def install_dependencies():
    """Install required dependencies"""
    print("Installing yt-dlp...")
    try:
        subprocess.run(["pip", "install", "-U", "yt-dlp"], check=True)
        print("✓ yt-dlp installed")
    except Exception as e:
        print(f"✗ Failed to install yt-dlp: {e}")
        print("Please install manually: pip install -U yt-dlp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RealEstate10K dataset")
    parser.add_argument("--num_videos", type=int, default=100,
                        help="Number of videos to download")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="Dataset split to download")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel downloads")
    parser.add_argument("--install_deps", action="store_true",
                        help="Install yt-dlp dependency")

    args = parser.parse_args()

    if args.install_deps:
        install_dependencies()

    # Step 1: Download metadata
    download_metadata()

    # Step 2: Download videos
    download_videos(args.num_videos, args.split, args.workers)
