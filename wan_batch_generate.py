import json
from pathlib import Path
import torch
from PIL import Image

from wan import WanTI2V
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video


# ----------------------------
# PATHS
# ----------------------------
PROMPT_JSON = "generated_prompts.json"
FIRST_FRAMES_DIR = "first_frames"
OUTPUT_DIR = "wan_outputs"

RESUME = True
MAX_VIDEOS = 10   # e.g. 10 to only run first 10 videos


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def skip(path):
    return RESUME and Path(path).exists()


def main():
    ensure_dir(OUTPUT_DIR)

    # -----------------------------------
    # Load JSON
    # -----------------------------------
    if not Path(PROMPT_JSON).exists():
        raise FileNotFoundError(f"Cannot find {PROMPT_JSON}")

    with open(PROMPT_JSON) as f:
        assignments = json.load(f)

    # Sorted keys → sequential execution
    keys = sorted(assignments.keys())

    if MAX_VIDEOS is not None:
        keys = keys[:MAX_VIDEOS]

    print(f"Running {len(keys)} videos.")
    print(keys)

    # -----------------------------------
    # Load Wan TI2V Model
    # -----------------------------------
    print("Loading Wan TI2V (5B)...")
    config = WAN_CONFIGS['ti2v-5B']
    wan_ti2v = WanTI2V(
        config=config,
        checkpoint_dir='./Wan2.2-TI2V-5B',
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,  # Keep T5 on CPU to save VRAM
        convert_model_dtype=True  # Convert to bfloat16
    )
    print("Model loaded successfully!")

    # -----------------------------------
    # Sequential generation
    # -----------------------------------
    for key in keys:
        data = assignments[key]
        img_path = data["image_prompt"]
        text_prompt = data["text_prompt"]

        out_path = Path(OUTPUT_DIR) / f"{key}.mp4"

        print("\n" + "=" * 60)
        print(f"Processing {key}")
        print(f"Image:   {img_path}")
        print(f"Prompt:  {text_prompt[:80]}...")
        print(f"Output:  {out_path}")
        print("=" * 60)

        # SKIP if exists
        if skip(out_path):
            print(f"[SKIP] Already exists → {out_path}")
        else:
            # Load image
            if not Path(img_path).exists():
                print(f"[ERROR] Image not found: {img_path}")
                continue

            img = Image.open(img_path).convert('RGB')
            print(f"[WAN] Loaded image: {img.size}")

            # Generate video
            print("[WAN] Generating video...")
            video = wan_ti2v.generate(
                input_prompt=text_prompt,
                img=img,
                size=(1280, 704),  # Same resolution as CogVideo
                max_area=704 * 1280,
                frame_num=49,  # Same as CogVideo (must be 4n+1)
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=50,
                guide_scale=6.0,  # Same as CogVideo
                n_prompt="",
                seed=42,  # Same seed as CogVideo for consistency
                offload_model=True  # Save VRAM
            )

            # Save video
            print("[WAN] Saving video...")
            save_video(
                tensor=video[None],  # Add batch dimension
                save_file=str(out_path),
                fps=24,  # Wan TI2V uses 24 fps
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"[WAN] Saved → {out_path}")

        # -----------------------------------
        # Update JSON for this item
        # -----------------------------------
        rel_path = str(out_path)

        if "generated_videos" not in data:
            data["generated_videos"] = []

        if rel_path not in data["generated_videos"]:
            data["generated_videos"].append(rel_path)

        assignments[key] = data

        # Write back to JSON immediately
        with open(PROMPT_JSON, "w") as f:
            json.dump(assignments, f, indent=2)

        print(f"[JSON] Updated entry for {key}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
