import json
from pathlib import Path
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
import random
import os
import sys
import fcntl 
import time
from typing import Dict, Any, List, Optional

# ----------------------------
# PATHS & CONFIGURATION
# ----------------------------
PROMPT_JSON = "generated_prompts.json"
OUTPUT_DIR = "../data_cog"  # 视频输出根目录
OUTPUT_JSON = "../data_cog/cog_generation_results.json"  # JSON 结果文件

# 配置
K = 3               # ⭐ 每个 Prompt 目标生成 K 个视频 (内部循环 K 次)
RESUME = True       # 启用断点续传
MAX_GROUPS = 1   

# ⭐ 关键配置: 设置你想使用的 GPU ID
# 如果你在命令行中使用 CUDA_VISIBLE_DEVICES=N python ... 运行，
# 你可以将这里的 GPU_ID 设置为 None 或一个默认值 (但命令行会覆盖它)
GPU_ID = 1          

# --- 辅助函数 (保持不变) ---
def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def load_results_json() -> Dict[str, Any]:
    """安全加载 JSON 文件。"""
    if not Path(OUTPUT_JSON).exists():
        return {"groups": []}
    try:
        with open(OUTPUT_JSON, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"[WARN] Could not load {OUTPUT_JSON}, starting fresh. Error: {e}")
        return {"groups": []}

def safe_save_json(results):
    """安全保存 JSON 文件。"""
    temp_file = OUTPUT_JSON + ".tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2)
        os.rename(temp_file, OUTPUT_JSON) 
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")

def skip(path: Path) -> bool:
    """检查文件是否存在"""
    return path.exists()

# --- 主函数 ---
def main():
    
    # ⭐ 步骤 1: 设置环境变量，隔离 GPU
    device = torch.device("cpu") # 默认值
    if GPU_ID is not None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID) 
        device = torch.device("cuda:0")
        print(f"[INFO] Using CUDA Device ID {GPU_ID}. PyTorch internal ID is cuda:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("[INFO] Using default CUDA Device ID 0.")
    else:
        print("[INFO] CUDA not available. Using CPU.")

    ensure_dir(OUTPUT_DIR)
    
    # -----------------------------------
    # Load Prompts and Setup Keys (保持不变)
    # -----------------------------------
    if not Path(PROMPT_JSON).exists():
        raise FileNotFoundError(f"Cannot find {PROMPT_JSON}")

    with open(PROMPT_JSON) as f:
        assignments = json.load(f)

    all_keys = sorted(assignments.keys())
    keys_to_process = all_keys[:MAX_GROUPS] if MAX_GROUPS is not None else all_keys
    
    # -----------------------------------
    # 步骤 2: Load CogVideoX 到指定 GPU
    # -----------------------------------
    print("Loading CogVideoX (5B I2V)...")
    try:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX1.5-5B-I2V", 
            torch_dtype=torch.bfloat16
        )

        pipe.to(device)
             
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

    except Exception as e:
        print(f"[FATAL ERROR] 加载 CogVideoX 失败: {e}")
        sys.exit(1)


    # -----------------------------------
    # Generation Loop (Group Level)
    # -----------------------------------
    results = load_results_json()
    
    for idx, key in enumerate(keys_to_process):
        data = assignments[key]
        img_path = data["image_prompt"]
        text_prompt = data["text_prompt"]

        group_id = int(key) 
        group_folder = Path(OUTPUT_DIR) / str(group_id)
        ensure_dir(group_folder)

        # 查找或创建 Group Entry (逻辑不变)
        group_list = results.get("groups", [])
        group_entry_list = [g for g in group_list if g.get("group_id") == group_id]
        
        if group_entry_list:
            group_entry = group_entry_list[0]
            results["groups"] = [g for g in group_list if g.get("group_id") != group_id]
        else:
            group_entry = {
                "group_id": group_id,
                "image_path": img_path,
                "text_prompt": text_prompt,
                "videos": []
            }
            
        print("\n==============================")
        print(f"Processing Group {group_id} (key: {key})")
        print(f"Image:   {img_path}")
        print(f"Videos recorded so far: {len(group_entry['videos'])}/{K}")
        print("==============================")
        
        try:
            image = load_image(img_path) 
        except Exception as e:
            print(f"[ERROR] Could not load image {img_path}: {e}")
            continue
        
        # -----------------------------------
        # 步骤 3: 内部循环生成 K 个 ( Batch=1 )
        # -----------------------------------
        
        # 缓存已记录的且文件存在的视频名
        recorded_videos = {
            v["video_name"] for v in group_entry["videos"] 
            if v.get("video_name") and skip(group_folder / v["video_name"])
        }
        
        videos_to_process_count = 0
        
        for k_idx in range(1, K + 1):
            video_name = f"{k_idx}.mp4"
            out_path = group_folder / video_name

            if video_name in recorded_videos:
                print(f"[SKIP] {video_name} already exists.")
                continue
            
            # --- GENERATION (Batch=1) ---
            generator = torch.Generator(device=device).manual_seed(random.randint(0, 2**32 - 1))
            
            print(f"[COG] Generating video {k_idx}/{K}...")

            try:
                # ⭐ 关键生成: num_videos_per_prompt=1
                output_frames = pipe(
                    prompt=text_prompt,
                    image=image,
                    num_videos_per_prompt=1, 
                    num_inference_steps=50,
                    num_frames=81,
                    guidance_scale=6,
                    generator=generator,
                ).frames 
                
                # output_frames 是一个包含一个视频帧列表的列表 [[frame1, frame2, ...]]
                output = output_frames[0] 
                
                # 保存视频到磁盘
                export_to_video(output, str(out_path), fps=8)
                print(f"[COG] Saved → {out_path}")
                
                # 创建新的视频记录 (不包含 seed)
                video_path_relative = Path(str(group_id)) / video_name 
                new_video_entry = {
                    "video_name": video_name,
                    "video_path": str(video_path_relative), 
                }
                
                # 4. 统一更新 Group Entry 和 JSON
                # 移除旧记录 (如果存在不完整的记录)
                group_entry["videos"] = [v for v in group_entry["videos"] if v.get("video_name") != video_name]
                group_entry["videos"].append(new_video_entry)
                group_entry["videos"].sort(key=lambda x: x["video_name"]) 

                results["groups"].append(group_entry)
                results["groups"].sort(key=lambda x: x["group_id"])
                safe_save_json(results)
                
                videos_to_process_count += 1
                
            except Exception as e:
                print(f"[ERROR] Generation failed for {video_name}: {e}. Skipping to next k_idx.")
                # 即使失败也要释放内存，防止 OOM 积累
                torch.cuda.empty_cache()
                continue

        print(f"[JSON] Finished processing Group {group_id}. {videos_to_process_count} new videos generated.")

        # -----------------------------------
        # Group cleanup 
        # -----------------------------------
        del image
        torch.cuda.empty_cache()


    print("\n======== ALL DONE ========\n")


if __name__ == "__main__":
    main()