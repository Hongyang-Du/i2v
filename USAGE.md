# I2V数据集设置使用指南

## ✅ 已完成的工作

我已经为你创建了完整的DL3DV和RealEstate10K数据集下载和处理流程：

### 创建的脚本文件：

1. **[setup_i2v_datasets.py](file:///home/junjie/i2v/setup_i2v_datasets.py)** - 主脚本（一键运行所有步骤）
2. **[download_dl3dv.py](file:///home/junjie/i2v/download_dl3dv.py)** - 下载DL3DV数据集
3. **[download_realestate10k.py](file:///home/junjie/i2v/download_realestate10k.py)** - 下载RealEstate10K数据集
4. **[extract_first_frames.py](file:///home/junjie/i2v/extract_first_frames.py)** - 提取视频第一帧
5. **[assign_cam_motion.py](file:///home/junjie/i2v/assign_cam_motion.py)** - 已存在，无需修改
6. **[README_DATASETS.md](file:///home/junjie/i2v/README_DATASETS.md)** - 详细文档

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
# 进入i2v目录
cd /home/junjie/i2v

# 安装所需依赖
pip install opencv-python huggingface-hub tqdm requests yt-dlp
```

### 步骤2: 运行主脚本

```bash
# 默认配置：下载DL3DV样本(11个视频) + 100个RealEstate10K视频
python setup_i2v_datasets.py

# 或者自定义数量
python setup_i2v_datasets.py --dl3dv_videos 50 --realestate_videos 200
```

### 步骤3: 检查结果

运行完成后会生成：
- `first_frames/` - 包含所有提取的第一帧图片（按数字排序：0.jpg, 1.jpg, ...）
- `generated_prompts.json` - 包含每个帧对应的相机运动提示词

## 📊 工作流程

```
1. 下载DL3DV视频 (从Hugging Face)
   ↓
2. 下载RealEstate10K视频 (从YouTube)
   ↓
3. 提取每个视频的第一帧
   ↓
4. 按数字顺序保存到first_frames/
   ↓
5. 为每个帧生成相机运动提示词
   ↓
6. 保存到generated_prompts.json
```

## 💡 常用命令示例

```bash
# 只检查依赖是否安装
python setup_i2v_datasets.py --check_deps

# 只下载和处理DL3DV
python setup_i2v_datasets.py --datasets dl3dv

# 只下载和处理RealEstate10K
python setup_i2v_datasets.py --datasets realestate

# 跳过下载，只从现有视频提取帧
python setup_i2v_datasets.py --skip_download

# 清空现有帧重新开始
python setup_i2v_datasets.py --clear_existing

# 不生成提示词
python setup_i2v_datasets.py --skip_prompts
```

## 🔧 分步运行（如果需要更细粒度控制）

```bash
# 步骤1: 下载DL3DV
python download_dl3dv.py --dataset sample --num_videos 20

# 步骤2: 下载RealEstate10K
python download_realestate10k.py --num_videos 100 --workers 4

# 步骤3: 提取第一帧
python extract_first_frames.py --datasets all

# 步骤4: 生成相机运动提示词
python assign_cam_motion.py
```

## 📁 生成的目录结构

```
i2v/
├── datasets/
│   ├── dl3dv/              # DL3DV下载的视频
│   └── realestate10k/
│       ├── videos/         # RealEstate10K下载的视频
│       └── metadata/       # 相机姿态数据
│
├── first_frames/           # 提取的第一帧
│   ├── 0.jpg              # DL3DV视频1的第一帧
│   ├── 1.jpg              # DL3DV视频2的第一帧
│   ├── ...
│   ├── 10.jpg             # DL3DV视频11的第一帧
│   ├── 11.jpg             # RealEstate视频1的第一帧
│   └── ...
│
└── generated_prompts.json  # 相机运动提示词
```

## 📝 generated_prompts.json 格式

```json
{
  "0": {
    "image_prompt": "first_frames/0.jpg",
    "camera_motion": "push forward into the scene, then pan across the room",
    "text_prompt": "A realistic continuation of the reference indoor scene. Everything must remain completely static: no moving people, no shifting objects, and no dynamic elements. All furniture and geometry must stay identical. Only the camera is allowed to move. Render physically accurate multi-step camera motion. Camera motion: push forward into the scene, then pan across the room."
  },
  ...
}
```

## ⚠️ 注意事项

### DL3DV数据集
- 首次使用需要在Hugging Face上请求访问权限
- 访问 https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample
- 点击"Access repository"并接受条款
- 运行 `huggingface-cli login` 输入token

### RealEstate10K数据集
- 从YouTube下载，部分视频可能已删除或私有（正常现象）
- 建议使用4-8个并行worker
- 如果遇到速率限制，减少worker数量或等待后重试

## 🎯 下一步

数据准备完成后：

1. 检查 `first_frames/` 目录确认帧已正确提取
2. 查看 `generated_prompts.json` 确认提示词生成正确
3. 使用这些数据运行你的I2V生成管线（如CogVideo、Wan等）

## 🔗 相关资源

- [DL3DV官网](https://dl3dv-10k.github.io/DL3DV-10K/)
- [RealEstate10K官网](https://google.github.io/realestate10k/)
- [详细文档](file:///home/junjie/i2v/README_DATASETS.md)

## 🐛 遇到问题？

1. 查看 [README_DATASETS.md](file:///home/junjie/i2v/README_DATASETS.md) 的故障排除部分
2. 确保所有依赖已正确安装
3. 检查网络连接（需要访问Hugging Face和YouTube）
