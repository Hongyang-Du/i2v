import json
from pathlib import Path
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video


# ----------------------------
# PATHS
# ----------------------------
PROMPT_JSON = "generated_prompts.json"
OUTPUT_DIR = "cog_outputs"

RESUME = True
MAX_VIDEOS = None   # e.g. 10 to only run first 10 videos


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
    # Load CogVideoX
    # -----------------------------------
    print("Loading CogVideoX (5B I2V)...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # -----------------------------------
    # Sequential generation
    # -----------------------------------
    for key in keys:
        data = assignments[key]
        img_path = data["image_prompt"]
        text_prompt = data["text_prompt"]

        out_path = Path(OUTPUT_DIR) / f"{key}.mp4"

        print("\n==============================")
        print(f"Processing {key}")
        print(f"Image:   {img_path}")
        print(f"Output:  {out_path}")
        print("==============================")

        # SKIP if exists
        if skip(out_path):
            print(f"[SKIP] Already exists → {out_path}")
        else:
            # Load image
            image = load_image(img_path)

            # Generate video
            print("[COG] Generating video...")
            video = pipe(
                prompt=text_prompt,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            export_to_video(video, str(out_path), fps=8)
            print(f"[COG] Saved → {out_path}")

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

    print("\n======== ALL DONE ========\n")


if __name__ == "__main__":
    main()
