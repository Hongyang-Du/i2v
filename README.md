### ⚙️ 1. 环境准备
Python 版本在 3.10 到 3.12 之间。

```
conda create -n video3d-dpo python=3.10 -y
conda activate video3d-dpo
pip install -r requirements.txt
```
### 🌐 2. 数据集下载 (DL3DV)
先登录 Hugging Face 并获取数据集访问权限。

```
wget https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/scripts/download.py

python download.py --odir DL3DV-10K --subset 1K --resolution 480P --file_type images+poses --clean_cache
```

获取第一帧随机生成text prompt
```
python extract_first_frames.py
python assign_cam_motion.py
```
CogvideoX-I2V-5B 生成视频 (需要修改一些sh里的参数)
```
bash run_cog_gen.sh
```