#!/usr/bin/env python3
"""
专门用于 DL3DV 数据集：
从 DL3DV 的图像序列结构中，递归查找第一帧 (frame_00001.png)，并保存到输出目录。
"""
import os
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Optional

# ============================================================================
# ⭐ 配置区域
# ============================================================================

# DL3DV 的根目录（相对于运行脚本的当前目录）
DL3DV_DIR = "DL3DV-10K/1K"
# 输出目录（相对于运行脚本的当前目录）
OUTPUT_DIR = "../dataset/first_frames"

# ============================================================================
# 辅助函数
# ============================================================================

def get_next_frame_number(output_dir: Path) -> int:
    """获取基于现有文件的下一个可用帧编号"""
    existing_frames = list(output_dir.glob("*.jpg"))
    if not existing_frames:
        return 0

    numbers = []
    for frame in existing_frames:
        try:
            num = int(frame.stem)
            numbers.append(num)
        except ValueError:
            continue

    return max(numbers) + 1 if numbers else 0

# ============================================================================
# ✨ 核心处理函数：处理 DL3DV 图像序列
# ============================================================================

def process_dl3dv_images(start_number: int = 0, max_videos: Optional[int] = None) -> int:
    """
    递归查找 DL3DV 场景下的第一帧 (frame_00001.png)，并复制到输出目录。
    """
    print("\n=== Processing DL3DV First Frames (frame_00001.png) ===")
    
    dl3dv_path = Path(DL3DV_DIR)
    if not dl3dv_path.exists():
        print(f"✗ DL3DV directory not found: {DL3DV_DIR}")
        return start_number

    # 1. 查找所有场景目录下的第一帧图像文件
    first_frame_paths = list(dl3dv_path.rglob("frame_00001.png"))

    # 按场景路径排序，以保证处理顺序一致性
    first_frame_paths = sorted(first_frame_paths, key=lambda p: str(p.parent))

    if not first_frame_paths:
        print(f"✗ No first frame images (frame_00001.png) found in {DL3DV_DIR}")
        print("请检查 DL3DV_DIR 是否设置正确，且子目录中存在 frame_00001.png 文件。")
        return start_number

    # 2. 实现断点续传（Resume Logic）
    files_to_process = first_frame_paths
    
    # 如果 current_number > 0，则跳过前面已处理的文件。
    if start_number > 0 and start_number < len(first_frame_paths):
        # 假设文件是按序保存的，我们跳过前 start_number 个文件
        files_to_process = first_frame_paths[start_number:]
        print(f"Resume Mode: Skipping {start_number} already processed files.")
    elif start_number >= len(first_frame_paths):
        print("Resume Mode: All files appear to be processed.")
        return start_number

    # 3. 限制数量并复制文件
    if max_videos:
        files_to_process = files_to_process[:max_videos]

    print(f"Found {len(first_frame_paths)} total frames. Processing {len(files_to_process)} frames.")

    current_number = start_number
    successful = 0
    
    for image_path in tqdm(files_to_process, desc="Copying DL3DV first frames"):
        # 统一保存为 .jpg 格式
        output_path = Path(OUTPUT_DIR) / f"{current_number}.jpg"

        try:
            # 使用 cv2 读取 PNG 图像 (cv2.imread 可以正确处理 PNG 透明度)
            frame = cv2.imread(str(image_path))
            
            if frame is None:
                 raise ValueError(f"Failed to read image.")
                 
            # 统一保存为 JPEG 格式
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            successful += 1
            current_number += 1

        except Exception as e:
            print(f"  ✗ Error processing scene {image_path.parent.name} ({image_path.name}): {e}")
            continue

    print(f"✓ DL3DV: Copied {successful} frames")
    return current_number

# ============================================================================
# 主函数
# ============================================================================

def main():
    # 修正：将 global 声明放在函数开头（虽然我们更推荐避免使用 global，但这里为了兼容原结构）
    global OUTPUT_DIR 
    
    parser = argparse.ArgumentParser(
        description="Extract first frames from DL3DV dataset"
    )
    parser.add_argument(
        "--max_videos_per_dataset",
        type=int,
        default=None,
        help="Maximum number of scenes/frames to process"
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

    # 更新全局输出目录
    OUTPUT_DIR = args.output_dir
    output_path = Path(OUTPUT_DIR)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 清除现有帧
    if args.clear_existing:
        print(f"Clearing existing frames in {OUTPUT_DIR}...")
        for frame in output_path.glob("*.jpg"):
            frame.unlink()
        print("✓ Cleared existing frames")
        current_number = 0
    else:
        # 获取下一个可用编号
        current_number = get_next_frame_number(output_path)
        print(f"Starting from frame number: {current_number}")

    # 执行处理
    final_number = process_dl3dv_images(
        start_number=current_number,
        max_videos=args.max_videos_per_dataset
    )

    # 总结
    total_frames = len(list(output_path.glob("*.jpg")))
    print(f"\n{'='*50}")
    print(f"✓ Extraction complete!")
    print(f"✓ Total frames in {OUTPUT_DIR}: {total_frames}")
    print(f"✓ Frame numbers: 0 to {final_number - 1}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()