import json
from pathlib import Path
import torch
from PIL import Image
import os
import argparse
import fcntl
import time

from wan import WanTI2V
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video


# ----------------------------
# PATHS
# ----------------------------
PROMPT_JSON = "generated_prompts.json"
FIRST_FRAMES_DIR = "first_frames"
OUTPUT_DIR = "wan_dpo_outputs"
OUTPUT_JSON = "wan_generation_results.json"

RESUME = True
MAX_VIDEOS = 1   # e.g. 10 to only run first 10 videos
K = 3  # Number of videos to generate per input
FIXED_SEEDS = [42, 123, 456, 789]  # Fixed seeds for k videos (must have K values)


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def skip(path):
    return RESUME and Path(path).exists()


def get_gpu_id():
    """Get GPU ID from environment variable or default to 0"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        return int(devices[0]) if devices else 0

    # Auto-select GPU with most free memory
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            free_memory = []
            for i in range(num_gpus):
                torch.cuda.set_device(i)
                free, total = torch.cuda.mem_get_info(i)
                free_memory.append((i, free))
            # Sort by free memory and pick the one with most free
            free_memory.sort(key=lambda x: x[1], reverse=True)
            return free_memory[0][0]
    return 0


def load_results_json():
    """Load existing results JSON or create new structure with file locking"""
    if not Path(OUTPUT_JSON).exists():
        return {"groups": []}

    max_retries = 10
    for attempt in range(max_retries):
        try:
            with open(OUTPUT_JSON, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data
        except (IOError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"Warning: Could not load {OUTPUT_JSON}, starting fresh")
                return {"groups": []}
    return {"groups": []}


def save_results_json(results):
    """Save results JSON with file locking for multi-GPU safety"""
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # Create temp file for atomic write
            temp_file = OUTPUT_JSON + ".tmp"

            with open(temp_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                try:
                    json.dump(results, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Atomic rename
            os.rename(temp_file, OUTPUT_JSON)
            return
        except IOError as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                raise Exception(f"Failed to save {OUTPUT_JSON} after {max_retries} attempts: {e}")


def main():
    # -----------------------------------
    # Parse command line arguments
    # -----------------------------------
    parser = argparse.ArgumentParser(description='Wan batch video generation')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Start index for processing (for multi-GPU)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index for processing (for multi-GPU)')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU ID to use (overrides auto-selection)')
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    ensure_dir("logs")  # For multi-GPU logs

    # -----------------------------------
    # Load JSON
    # -----------------------------------
    if not Path(PROMPT_JSON).exists():
        raise FileNotFoundError(f"Cannot find {PROMPT_JSON}")

    with open(PROMPT_JSON) as f:
        assignments = json.load(f)

    # Sorted keys → sequential execution
    keys = sorted(assignments.keys())

    # Apply start/end index for multi-GPU processing
    if args.start_idx is not None and args.end_idx is not None:
        keys = keys[args.start_idx:args.end_idx]
        print(f"Processing subset: indices {args.start_idx} to {args.end_idx - 1}")
    elif MAX_VIDEOS is not None:
        keys = keys[:MAX_VIDEOS]

    print(f"Running {len(keys)} groups with {K} videos each.")
    print(f"Total videos to generate: {len(keys) * K}")
    print(f"Keys: {keys}")

    # -----------------------------------
    # Auto-select GPU or use specified GPU
    # -----------------------------------
    if args.gpu_id is not None:
        gpu_id = args.gpu_id
        print(f"Using specified GPU: {gpu_id}")
    else:
        gpu_id = get_gpu_id()
        print(f"Auto-selected GPU: {gpu_id}")

    # -----------------------------------
    # Load Wan TI2V Model ONCE
    # -----------------------------------
    print("Loading Wan TI2V (5B)...")
    config = WAN_CONFIGS['ti2v-5B']
    wan_ti2v = WanTI2V(
        config=config,
        checkpoint_dir='./Wan2.2-TI2V-5B',
        device_id=gpu_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,  # Keep T5 on CPU to save VRAM
        convert_model_dtype=True  # Convert to bfloat16
    )
    print("Model loaded successfully!")

    # -----------------------------------
    # Load or create results JSON
    # -----------------------------------
    results = load_results_json()

    # -----------------------------------
    # Sequential generation with k videos per input
    # -----------------------------------
    all_keys = sorted(assignments.keys())

    for idx, key in enumerate(keys):
        data = assignments[key]
        img_path = data["image_prompt"]
        text_prompt = data["text_prompt"]

        # Use global index for group_id (consistent across all GPUs)
        global_idx = all_keys.index(key)
        group_id = global_idx + 1

        # Create subfolder for this group
        group_folder = Path(OUTPUT_DIR) / str(group_id)
        ensure_dir(group_folder)

        print("\n" + "=" * 60)
        print(f"Processing Group {group_id} (local {idx + 1}/{len(keys)}, key: {key})")
        print(f"Image:   {img_path}")
        print(f"Prompt:  {text_prompt[:80]}...")
        print(f"Folder:  {group_folder}")
        print("=" * 60)

        # Check if image exists
        if not Path(img_path).exists():
            print(f"[ERROR] Image not found: {img_path}")
            continue

        # Load image once for all k generations
        img = Image.open(img_path).convert('RGB')
        print(f"[WAN] Loaded image: {img.size}")

        # Create group entry for results JSON
        group_entry = {
            "group_id": group_id,
            "image_path": img_path,
            "text_prompt": text_prompt,
            "videos": []
        }

        # Generate k videos with fixed seeds
        for k_idx in range(1, K + 1):
            video_name = f"{k_idx}.mp4"
            out_path = group_folder / video_name

            # SKIP if exists
            if skip(out_path):
                print(f"[SKIP] Video {k_idx}/{K} already exists → {out_path}")
                group_entry["videos"].append({
                    "video_name": video_name,
                    "video_path": str(out_path)
                })
                continue

            # Use fixed seed from list
            seed = FIXED_SEEDS[k_idx - 1]

            print(f"[WAN] Generating video {k_idx}/{K} with seed {seed}...")
            video = wan_ti2v.generate(
                input_prompt=text_prompt,
                img=img,
                size=(1280, 704),
                max_area=704 * 1280,
                frame_num=49,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=50,
                guide_scale=6.0,
                n_prompt="",
                seed=seed,
                offload_model=True
            )

            # Save video
            print(f"[WAN] Saving video {k_idx}/{K}...")
            save_video(
                tensor=video[None],
                save_file=str(out_path),
                fps=24,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"[WAN] Saved → {out_path}")

            # Add to group entry (no seed needed since it's fixed)
            group_entry["videos"].append({
                "video_name": video_name,
                "video_path": str(out_path)
            })

        # Update results JSON
        # Reload to get latest changes from other GPUs, then update
        results = load_results_json()
        # Remove existing entry for this group if exists
        results["groups"] = [g for g in results["groups"] if g["group_id"] != group_id]
        results["groups"].append(group_entry)
        # Sort groups by group_id
        results["groups"].sort(key=lambda x: x["group_id"])

        # Save results JSON after each group
        save_results_json(results)
        print(f"[JSON] Updated results for group {group_id}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print(f"Results saved to {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
