#!/usr/bin/env python3
"""
Master script to set up I2V datasets (DL3DV + RealEstate10K)
This script:
1. Downloads DL3DV and RealEstate10K datasets
2. Extracts first frames from all videos
3. Saves frames with numerical ordering to first_frames/
4. Generates camera motion prompts for each frame
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f">>> {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, shell=isinstance(cmd, str))
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n" + "="*60)
    print("Checking dependencies...")
    print("="*60)

    dependencies = {
        "opencv-python": "cv2",
        "huggingface-hub": "huggingface_hub",
        "tqdm": "tqdm",
        "requests": "requests",
    }

    missing = []
    for package, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)

    # Check yt-dlp separately
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        print("✓ yt-dlp")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ yt-dlp - MISSING")
        missing.append("yt-dlp")

    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with:")
        print(f"  pip install {' '.join([d for d in missing if d != 'yt-dlp'])}")
        if "yt-dlp" in missing:
            print(f"  pip install yt-dlp")
        return False

    print("\n✓ All dependencies are installed")
    return True

def setup_dl3dv(num_videos=None, skip_download=False):
    """Download and process DL3DV dataset"""
    if skip_download:
        print("\n⏭  Skipping DL3DV download (--skip_download)")
        return True

    print("\n" + "="*60)
    print("STEP 1: DL3DV Dataset")
    print("="*60)

    cmd = ["python3", "download_dl3dv.py"]
    if num_videos:
        cmd.extend(["--num_videos", str(num_videos)])

    return run_command(cmd, "Download DL3DV dataset")

def setup_realestate10k(num_videos=100, skip_download=False):
    """Download and process RealEstate10K dataset"""
    if skip_download:
        print("\n⏭  Skipping RealEstate10K download (--skip_download)")
        return True

    print("\n" + "="*60)
    print("STEP 2: RealEstate10K Dataset")
    print("="*60)

    cmd = [
        "python3", "download_realestate10k.py",
        "--num_videos", str(num_videos),
        "--workers", "4"
    ]

    return run_command(cmd, "Download RealEstate10K dataset")

def extract_frames(datasets, max_videos_per_dataset=None, clear_existing=False):
    """Extract first frames from videos"""
    print("\n" + "="*60)
    print("STEP 3: Extract First Frames")
    print("="*60)

    cmd = ["python3", "extract_first_frames.py", "--datasets"] + datasets

    if max_videos_per_dataset:
        cmd.extend(["--max_videos_per_dataset", str(max_videos_per_dataset)])

    if clear_existing:
        cmd.append("--clear_existing")

    return run_command(cmd, "Extract first frames")

def generate_prompts():
    """Generate camera motion prompts"""
    print("\n" + "="*60)
    print("STEP 4: Generate Camera Motion Prompts")
    print("="*60)

    cmd = ["python3", "assign_cam_motion.py"]
    return run_command(cmd, "Generate camera motion prompts")

def main():
    parser = argparse.ArgumentParser(
        description="Set up I2V datasets with DL3DV and RealEstate10K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup with default settings (sample DL3DV + 100 RealEstate10K videos)
  python setup_i2v_datasets.py

  # Download more videos
  python setup_i2v_datasets.py --dl3dv_videos 50 --realestate_videos 200

  # Only extract frames (skip download)
  python setup_i2v_datasets.py --skip_download

  # Process only specific datasets
  python setup_i2v_datasets.py --datasets dl3dv --skip_download

  # Clear existing frames and start fresh
  python setup_i2v_datasets.py --clear_existing
        """
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["dl3dv", "realestate", "all"],
        default=["all"],
        help="Which datasets to process (default: all)"
    )

    parser.add_argument(
        "--dl3dv_videos",
        type=int,
        default=None,
        help="Number of DL3DV videos to download (default: all from sample)"
    )

    parser.add_argument(
        "--realestate_videos",
        type=int,
        default=100,
        help="Number of RealEstate10K videos to download (default: 100)"
    )

    parser.add_argument(
        "--max_videos_per_dataset",
        type=int,
        default=None,
        help="Max videos to process per dataset during extraction"
    )

    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading, only extract frames from existing videos"
    )

    parser.add_argument(
        "--clear_existing",
        action="store_true",
        help="Clear existing first_frames before extracting new ones"
    )

    parser.add_argument(
        "--skip_prompts",
        action="store_true",
        help="Skip camera motion prompt generation"
    )

    parser.add_argument(
        "--check_deps",
        action="store_true",
        help="Only check dependencies and exit"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*60)
    print("I2V Dataset Setup: DL3DV + RealEstate10K")
    print("="*60)

    # Check dependencies
    if not check_dependencies():
        if args.check_deps:
            sys.exit(1)
        print("\n⚠ Please install missing dependencies before continuing")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    elif args.check_deps:
        sys.exit(0)

    # Determine which datasets to process
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["dl3dv", "realestate"]

    success = True

    # Step 1 & 2: Download datasets
    if not args.skip_download:
        if "dl3dv" in datasets:
            if not setup_dl3dv(args.dl3dv_videos, args.skip_download):
                success = False
                print("\n⚠ DL3DV download failed, but continuing...")

        if "realestate" in datasets:
            if not setup_realestate10k(args.realestate_videos, args.skip_download):
                success = False
                print("\n⚠ RealEstate10K download failed, but continuing...")

    # Step 3: Extract first frames
    if not extract_frames(datasets, args.max_videos_per_dataset, args.clear_existing):
        success = False
        print("\n⚠ Frame extraction failed")

    # Step 4: Generate prompts
    if not args.skip_prompts:
        if not generate_prompts():
            success = False
            print("\n⚠ Prompt generation failed")

    # Final summary
    print("\n" + "="*60)
    if success:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Check first_frames/ directory for extracted frames")
        print("  2. Review generated_prompts.json for camera motion prompts")
        print("  3. Run your I2V generation pipeline")
    else:
        print("⚠ Setup completed with some errors")
        print("Please check the output above for details")
    print("="*60 + "\n")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
