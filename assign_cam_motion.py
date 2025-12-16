import json
import random
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
FIRST_FRAMES = "../dataset/first_frames"
OUTPUT_JSON = "../dataset/generated_prompts.json"

PREFIX_PROMPT = (
    "A realistic continuation of the reference indoor scene. "
    "Everything must remain completely static: no moving people, no shifting objects, "
    "and no dynamic elements. All furniture and geometry must stay identical. "
    "Only the camera is allowed to move. Render physically accurate multi-step camera motion. "
)

# ============================================================
# NATURAL CAMERA MOTION (NO NUMBERS, NO SYMBOLS)
# ============================================================

TRANSLATIONS = [
    "push forward into the scene",
    "pull back away from the scene",
    "slide sideways across the room",
    "move laterally along the furniture line",
    "drift across the space",
    "glide toward the room center",
    "shift through the foreground",
    "move diagonally through the space",
]

ROTATIONS = [
    "pan across the room",
    "pan toward the main subject",
    "scan across the shelves",
    "tilt upward toward the ceiling",
    "tilt downward toward the floor",
    "roll gently to one side",
    "look around the environment",
]

COMPLEX_PATHS = [
    "orbit around the scene",
    "arc around the center of the room",
    "circle around the main object",
    "swing around the room",
    "pivot around the viewpoint",
]


def random_motion_piece():
    """Return a single natural-language motion."""
    group = random.choice(["T", "R", "C"])
    if group == "T":
        return random.choice(TRANSLATIONS)
    if group == "R":
        return random.choice(ROTATIONS)
    if group == "C":
        return random.choice(COMPLEX_PATHS)


def generate_multi_stage_motion():
    """Generate 2–3 segments joined by natural connectors."""
    pieces = [random_motion_piece() for _ in range(random.choice([2, 3]))]

    if len(pieces) == 1:
        return pieces[0]
    elif len(pieces) == 2:
        return f"{pieces[0]}, then {pieces[1]}"
    else:
        return f"{pieces[0]}, then {pieces[1]}, followed by {pieces[2]}"


# ============================================================
# MAIN
# ============================================================
def main():
    frames = sorted(Path(FIRST_FRAMES).glob("*.jpg"))
    print(f"Found {len(frames)} images.")

    result = {}

    for img_path in frames:
        name = img_path.stem

        camera_motion = generate_multi_stage_motion()
        text_prompt = PREFIX_PROMPT + f" Camera motion: {camera_motion}."

        result[name] = {
            "image_prompt": str(img_path),
            "camera_motion": camera_motion,
            "text_prompt": text_prompt,
        }

        print(f"[{name}]  {camera_motion}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=========== DONE ===========")
    print(f"Saved prompts → {OUTPUT_JSON}\n")


if __name__ == "__main__":
    main()
