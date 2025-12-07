#!/usr/bin/env python3
"""
测试脚本 - 验证批量生成配置
"""

import json
from pathlib import Path
import torch

def check_files():
    """检查必要的文件是否存在"""
    print("=" * 60)
    print("检查文件...")
    print("=" * 60)

    required_files = {
        "generated_prompts.json": "输入JSON文件",
        "wan_batch_generate.py": "批量生成脚本",
        "run_multi_gpu.sh": "多卡运行脚本"
    }

    all_exist = True
    for file, desc in required_files.items():
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {desc}: {file}")
        if not exists:
            all_exist = False

    return all_exist


def check_json_structure():
    """检查JSON文件结构"""
    print("\n" + "=" * 60)
    print("检查JSON结构...")
    print("=" * 60)

    if not Path("generated_prompts.json").exists():
        print("✗ generated_prompts.json 不存在")
        return False

    try:
        with open("generated_prompts.json") as f:
            data = json.load(f)

        print(f"✓ 总样本数: {len(data)}")

        # 检查第一个样本的结构
        if data:
            first_key = list(data.keys())[0]
            first_item = data[first_key]

            print(f"\n示例样本 (key: {first_key}):")
            print(f"  - image_prompt: {first_item.get('image_prompt', 'N/A')}")
            print(f"  - text_prompt: {first_item.get('text_prompt', 'N/A')[:60]}...")

            # 检查图片是否存在
            img_path = first_item.get('image_prompt')
            if img_path:
                img_exists = Path(img_path).exists()
                status = "✓" if img_exists else "✗"
                print(f"  {status} 图片文件存在: {img_exists}")

        return True
    except Exception as e:
        print(f"✗ 读取JSON文件出错: {e}")
        return False


def check_gpu():
    """检查GPU状态"""
    print("\n" + "=" * 60)
    print("检查GPU...")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("✗ CUDA不可用")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"✓ 可用GPU数量: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        print(f"  GPU {i}: {props.name}")
        print(f"    - 总显存: {total_gb:.1f} GB")
        print(f"    - 可用显存: {free_gb:.1f} GB")

    return True


def estimate_workload():
    """估算工作量"""
    print("\n" + "=" * 60)
    print("工作量估算...")
    print("=" * 60)

    # 从脚本中读取K值
    k = 4  # 默认值
    try:
        with open("wan_batch_generate.py") as f:
            for line in f:
                if line.startswith("K = "):
                    k = int(line.split("=")[1].strip())
                    break
    except:
        pass

    try:
        with open("generated_prompts.json") as f:
            data = json.load(f)
        num_inputs = len(data)
    except:
        num_inputs = 0

    total_videos = num_inputs * k

    print(f"输入样本数: {num_inputs}")
    print(f"每个输入生成视频数 (K): {k}")
    print(f"总共将生成视频: {total_videos}")

    # 估算时间（假设每个视频2分钟）
    est_minutes = total_videos * 2
    est_hours = est_minutes / 60

    print(f"\n估算生成时间（单卡）:")
    print(f"  - 约 {est_minutes:.0f} 分钟 ({est_hours:.1f} 小时)")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            multi_gpu_hours = est_hours / num_gpus
            print(f"\n使用 {num_gpus} 个GPU并行:")
            print(f"  - 约 {multi_gpu_hours:.1f} 小时")

    return True


def show_commands():
    """显示运行命令"""
    print("\n" + "=" * 60)
    print("运行命令")
    print("=" * 60)

    print("\n1. 单卡运行:")
    print("   python wan_batch_generate.py")

    print("\n2. 指定GPU:")
    print("   CUDA_VISIBLE_DEVICES=0 python wan_batch_generate.py")

    print("\n3. 多卡并行:")
    print("   bash run_multi_gpu.sh")

    print("\n4. 测试运行（只处理前2个样本）:")
    print("   修改 wan_batch_generate.py 中的 MAX_VIDEOS = 2")
    print("   python wan_batch_generate.py")

    print("\n5. 查看结果:")
    print("   ls wan_dpo_outputs/")
    print("   python -c \"import json; print(json.dumps(json.load(open('wan_generation_results.json')), indent=2))\"")


def main():
    print("\nWan 批量生成 - 配置检查\n")

    all_ok = True

    all_ok &= check_files()
    all_ok &= check_json_structure()
    all_ok &= check_gpu()
    all_ok &= estimate_workload()

    show_commands()

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ 所有检查通过！可以开始运行")
    else:
        print("✗ 有些检查未通过，请先修复问题")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
