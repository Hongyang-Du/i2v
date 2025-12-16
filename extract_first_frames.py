#!/usr/bin/env python3
"""
Extract first frames from DL3DV and RealEstate10K videos
Save them to first_frames directory with numerical ordering
"""
import os
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

# Configuration
DL3DV_DIR = "/home/junjie/i2v/datasets/dl3dv"
REALESTATE_DIR = "/home/junjie/i2v/datasets/realestate10k/videos"
OUTPUT_DIR = "/home/junjie/i2v/first_frames"

def extract_first_frame(video_path, output_path, max_retries=3):
    """
    Extract the first frame from a video file

    Args:
        video_path: Path to input video
        output_path: Path to save the first frame
        max_retries: Number of retries if frame extraction fails

    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                if attempt < max_retries - 1:
                    continue
                return False

            # Read first frame
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                if attempt < max_retries - 1:
                    continue
                return False

            # Save frame as JPEG
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"  ✗ Error extracting frame from {video_path}: {e}")
            return False

    return False

def get_next_frame_number(output_dir):
    """Get the next available frame number based on existing files"""
    existing_frames = list(Path(output_dir).glob("*.jpg"))
    if not existing_frames:
        return 0

    # Extract numbers from filenames
    numbers = []
    for frame in existing_frames:
        try:
            num = int(frame.stem)
            numbers.append(num)
        except ValueError:
            continue

    return max(numbers) + 1 if numbers else 0

def process_dl3dv_videos(start_number=0, max_videos=None):
    """
    Process DL3DV videos and extract first frames

    Args:
        start_number: Starting number for frame naming
        max_videos: Maximum number of videos to process

    Returns:
        int: Next available frame number
    """
    print("\n=== Processing DL3DV Videos ===")

    # Find all videos recursively
    video_paths = []
    if os.path.exists(DL3DV_DIR):
        video_paths = sorted(list(Path(DL3DV_DIR).rglob("*.mp4")))
    else:
        print(f"✗ DL3DV directory not found: {DL3DV_DIR}")
        print("  Please run download_dl3dv.py first")
        return start_number

    if not video_paths:
        print(f"✗ No videos found in {DL3DV_DIR}")
        return start_number

    # Limit number of videos
    if max_videos:
        video_paths = video_paths[:max_videos]

    print(f"Found {len(video_paths)} DL3DV videos")

    # Process videos
    current_number = start_number
    successful = 0
    failed = 0

    for video_path in tqdm(video_paths, desc="Extracting DL3DV frames"):
        output_path = Path(OUTPUT_DIR) / f"{current_number}.jpg"

        if extract_first_frame(video_path, output_path):
            successful += 1
            current_number += 1
        else:
            failed += 1

    print(f"✓ DL3DV: Extracted {successful} frames, {failed} failed")
    return current_number

def process_realestate_videos(start_number=0, max_videos=None):
    """
    Process RealEstate10K videos and extract first frames

    Args:
        start_number: Starting number for frame naming
        max_videos: Maximum number of videos to process

    Returns:
        int: Next available frame number
    """
    print("\n=== Processing RealEstate10K Videos ===")

    # Find all videos
    if not os.path.exists(REALESTATE_DIR):
        print(f"✗ RealEstate10K directory not found: {REALESTATE_DIR}")
        print("  Please run download_realestate10k.py first")
        return start_number

    video_paths = sorted(list(Path(REALESTATE_DIR).glob("*.mp4")))

    if not video_paths:
        print(f"✗ No videos found in {REALESTATE_DIR}")
        return start_number

    # Limit number of videos
    if max_videos:
        video_paths = video_paths[:max_videos]

    print(f"Found {len(video_paths)} RealEstate10K videos")

    # Process videos
    current_number = start_number
    successful = 0
    failed = 0

    for video_path in tqdm(video_paths, desc="Extracting RealEstate frames"):
        output_path = Path(OUTPUT_DIR) / f"{current_number}.jpg"

        if extract_first_frame(video_path, output_path):
            successful += 1
            current_number += 1
        else:
            failed += 1

    print(f"✓ RealEstate10K: Extracted {successful} frames, {failed} failed")
    return current_number

def main():
    parser = argparse.ArgumentParser(
        description="Extract first frames from DL3DV and RealEstate10K datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["dl3dv", "realestate", "all"],
        default=["all"],
        help="Which datasets to process"
    )
    parser.add_argument(
        "--max_videos_per_dataset",
        type=int,
        default=None,
        help="Maximum number of videos per dataset (default: all)"
    )
    parser.add_argument(
        "--clear_existing",
        action="store_true",
        help="Clear existing frames before extracting"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for first frames (default: {OUTPUT_DIR})"
    )

    args = parser.parse_args()

    # Update output directory
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Clear existing frames if requested
    if args.clear_existing:
        print(f"Clearing existing frames in {OUTPUT_DIR}...")
        for frame in Path(OUTPUT_DIR).glob("*.jpg"):
            frame.unlink()
        print("✓ Cleared existing frames")
        current_number = 0
    else:
        # Get next available number
        current_number = get_next_frame_number(OUTPUT_DIR)
        print(f"Starting from frame number: {current_number}")

    # Process datasets
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["dl3dv", "realestate"]

    if "dl3dv" in datasets:
        current_number = process_dl3dv_videos(
            start_number=current_number,
            max_videos=args.max_videos_per_dataset
        )

    if "realestate" in datasets:
        current_number = process_realestate_videos(
            start_number=current_number,
            max_videos=args.max_videos_per_dataset
        )

    # Summary
    total_frames = len(list(Path(OUTPUT_DIR).glob("*.jpg")))
    print(f"\n{'='*50}")
    print(f"✓ Extraction complete!")
    print(f"✓ Total frames in {OUTPUT_DIR}: {total_frames}")
    print(f"✓ Frame numbers: 0 to {current_number - 1}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
