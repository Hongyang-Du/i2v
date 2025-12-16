# I2V Dataset Setup: DL3DV + RealEstate10K

This directory contains scripts to download, process, and prepare DL3DV and RealEstate10K datasets for I2V (Image-to-Video) generation.

## 📋 Overview

The pipeline consists of 4 steps:
1. **Download DL3DV dataset** from Hugging Face
2. **Download RealEstate10K videos** from YouTube
3. **Extract first frames** from all videos with numerical ordering
4. **Generate camera motion prompts** for each frame

## 🚀 Quick Start

### One-Command Setup

```bash
# Install dependencies and run full pipeline
pip install opencv-python huggingface-hub tqdm requests yt-dlp
python setup_i2v_datasets.py
```

This will:
- Download DL3DV sample dataset (~11 videos)
- Download 100 RealEstate10K videos
- Extract first frames to `first_frames/`
- Generate prompts to `generated_prompts.json`

### Custom Configuration

```bash
# Download more videos
python setup_i2v_datasets.py --dl3dv_videos 50 --realestate_videos 200

# Only process DL3DV
python setup_i2v_datasets.py --datasets dl3dv

# Only process RealEstate10K
python setup_i2v_datasets.py --datasets realestate

# Skip download, only extract frames from existing videos
python setup_i2v_datasets.py --skip_download

# Clear existing frames and start fresh
python setup_i2v_datasets.py --clear_existing
```

## 📁 Directory Structure

```
i2v/
├── setup_i2v_datasets.py      # Master script (run this!)
├── download_dl3dv.py           # DL3DV downloader
├── download_realestate10k.py   # RealEstate10K downloader
├── extract_first_frames.py     # Frame extraction
├── assign_cam_motion.py        # Prompt generation
│
├── datasets/                   # Downloaded videos
│   ├── dl3dv/                  # DL3DV videos
│   └── realestate10k/          # RealEstate10K videos
│       └── videos/
│
├── first_frames/               # Extracted first frames (0.jpg, 1.jpg, ...)
└── generated_prompts.json      # Camera motion prompts for each frame
```

## 🔧 Individual Scripts

### 1. Download DL3DV Dataset

```bash
# Download sample dataset (11 videos)
python download_dl3dv.py --dataset sample

# Download specific number of videos
python download_dl3dv.py --dataset sample --num_videos 20
```

**Note**: For full DL3DV-10K dataset, you need to:
1. Request access at https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample
2. Run `huggingface-cli login`
3. Modify the script to use the full dataset repo

### 2. Download RealEstate10K Videos

```bash
# Download 100 videos from train split
python download_realestate10k.py --num_videos 100

# Download from test split
python download_realestate10k.py --num_videos 50 --split test

# Use more parallel workers
python download_realestate10k.py --num_videos 200 --workers 8

# Install yt-dlp if needed
python download_realestate10k.py --install_deps
```

### 3. Extract First Frames

```bash
# Extract from all datasets
python extract_first_frames.py --datasets all

# Extract only from DL3DV
python extract_first_frames.py --datasets dl3dv

# Limit videos per dataset
python extract_first_frames.py --max_videos_per_dataset 100

# Clear existing frames first
python extract_first_frames.py --clear_existing
```

### 4. Generate Camera Motion Prompts

```bash
# Generate prompts for all frames in first_frames/
python assign_cam_motion.py
```

This creates `generated_prompts.json` with entries like:

```json
{
  "0": {
    "image_prompt": "first_frames/0.jpg",
    "camera_motion": "push forward into the scene, then pan across the room",
    "text_prompt": "A realistic continuation of the reference indoor scene. Everything must remain completely static: no moving people, no shifting objects, and no dynamic elements. All furniture and geometry must stay identical. Only the camera is allowed to move. Render physically accurate multi-step camera motion. Camera motion: push forward into the scene, then pan across the room."
  },
  "1": {
    ...
  }
}
```

## 📦 Dependencies

```bash
pip install opencv-python huggingface-hub tqdm requests yt-dlp
```

Or check dependencies:

```bash
python setup_i2v_datasets.py --check_deps
```

## 🔍 Dataset Information

### DL3DV-10K
- **Source**: https://github.com/DL3DV-10K/Dataset
- **Hugging Face**: https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample
- **Size**: 10,510 videos, 51.2M frames
- **Resolution**: 4K, 960P, 480P versions available
- **Content**: Indoor/outdoor scenes with camera parameters

### RealEstate10K
- **Source**: https://google.github.io/realestate10k/
- **Size**: ~10K YouTube video clips
- **Split**: 90% train, 10% test
- **Content**: Indoor/outdoor real estate tours
- **License**: Creative Commons Attribution 4.0

## 🎯 Frame Numbering

Frames are saved with sequential numerical names:
- DL3DV frames: Start from 0 (or next available number)
- RealEstate10K frames: Continue from last DL3DV number

Example:
```
first_frames/
  0.jpg      # DL3DV video 1, frame 1
  1.jpg      # DL3DV video 2, frame 1
  ...
  10.jpg     # DL3DV video 11, frame 1
  11.jpg     # RealEstate10K video 1, frame 1
  12.jpg     # RealEstate10K video 2, frame 1
  ...
```

## 🚨 Troubleshooting

### DL3DV Download Issues

**Problem**: Access denied when downloading DL3DV

**Solution**:
1. Visit https://huggingface.co/datasets/DL3DV/DL3DV-10K-Sample
2. Click "Access repository"
3. Accept terms
4. Run `huggingface-cli login` and enter your token

### RealEstate10K Download Issues

**Problem**: YouTube videos fail to download

**Solution**:
- Some videos may be removed or private (normal)
- Update yt-dlp: `pip install -U yt-dlp`
- Reduce `--workers` to avoid rate limiting
- Use VPN if YouTube is blocked

**Problem**: "yt-dlp not found"

**Solution**:
```bash
pip install -U yt-dlp
# or
python download_realestate10k.py --install_deps
```

### Frame Extraction Issues

**Problem**: "No videos found"

**Solution**:
- Run download scripts first
- Check paths in script configuration
- Verify videos exist in `datasets/` directory

## 🎨 Camera Motion Prompts

The `assign_cam_motion.py` script generates natural language camera motions:

**Translations**: push forward, pull back, slide sideways, etc.
**Rotations**: pan across, tilt upward, roll gently, etc.
**Complex**: orbit around, arc around, circle around, etc.

Each prompt combines 2-3 motions for realistic multi-stage camera movement.

## 📊 Expected Results

After running the full pipeline:

```
first_frames/        # 111+ JPG images (11 DL3DV + 100 RealEstate10K)
generated_prompts.json   # 111+ entries with camera motions
```

## 🔗 References

- [DL3DV-10K Paper](https://dl3dv-10k.github.io/DL3DV-10K/)
- [RealEstate10K Paper](https://google.github.io/realestate10k/)
- [Hugging Face Hub Docs](https://huggingface.co/docs/huggingface_hub/)
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)

## 📝 License

- **DL3DV**: Check terms on Hugging Face
- **RealEstate10K**: CC BY 4.0
- **These scripts**: MIT License
