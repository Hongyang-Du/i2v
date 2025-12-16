import torch
import json
from pathlib import Path
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image # <--- 引入 load_image
import random
import os
import sys
import logging

# 配置日志，确保信息输出
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_ID = "THUDM/CogVideoX-5b-I2V"
# ----------------------------

def inspect_cogvideo_config():
    # -----------------------------------
    # 1. 初始化和核心组件检查 (保持不变)
    # -----------------------------------
    logging.info(f"--- 1. 正在加载模型配置: {MODEL_ID} ---")
    pipeline = None
    
    try:
        pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        logging.error(f"[ERROR] 无法加载 Pipeline。错误: {e}")
        return

    pipeline.to("cpu") 
    
    logging.info(f"--- 2. 模型核心组件类型与名称 ---")
    # ... (DIT, VAE, Text Encoder, Image Encoder 检查逻辑保持不变)
    
    # Denoising Model (DIT)
    try:
        logging.info(f"Denoising Model (DIT): {type(pipeline.transformer).__name__} (属性名: .transformer)")
    except AttributeError:
        logging.info("Denoising Model (DIT): ERROR - '.transformer' 属性访问失败。")
        
    # VAE Encoder/Decoder
    try:
        logging.info(f"VAE Encoder/Decoder: {type(pipeline.vae).__name__} (属性名: .vae)")
    except AttributeError:
        logging.info("VAE Encoder/Decoder: ERROR - '.vae' 属性访问失败。")

    # Text Encoder
    try:
        logging.info(f"Text Encoder (T5): {type(pipeline.text_encoder).__name__} (属性名: .text_encoder)")
    except AttributeError:
        logging.info("Text Encoder (T5): ERROR - '.text_encoder' 属性访问失败。")
        
    # Image Encoder (I2V 模型的关键)
    try:
        if hasattr(pipeline, 'image_encoder') and pipeline.image_encoder is not None:
             logging.info(f"Image Encoder: {type(pipeline.image_encoder).__name__} (属性名: .image_encoder)")
        elif hasattr(pipeline, 'feature_extractor') and pipeline.feature_extractor is not None:
             logging.info(f"Image Encoder: {type(pipeline.feature_extractor).__name__} (属性名: .feature_extractor)")
        else:
             logging.info("Image Encoder: (未找到标准属性名)")
    except AttributeError:
        logging.info("Image Encoder: (属性检查失败)")
        
    logging.info("-" * 60)

    # ------------------------------------------------
    # ⭐ 5. 检查 VAE/图像预处理器配置 (确定像素尺寸 H, W)
    # ------------------------------------------------
    logging.info(f"--- 5. VAE/图像预处理器配置 ---")
    
    # 尝试访问 VAE 配置中的样本尺寸 (如果存在)
    try:
        if hasattr(pipeline.vae, 'config') and 'sample_size' in pipeline.vae.config:
            vae_size = pipeline.vae.config.sample_size
            logging.info(f"VAE Latent Sample Size (H', W'): {vae_size} x {vae_size}")
    except Exception:
        pass
        
    # 尝试从图像预处理器中获取输入尺寸
    if hasattr(pipeline, 'image_processor') and pipeline.image_processor is not None:
        processor = pipeline.image_processor
        logging.info(f"Image Processor Type: {type(processor).__name__}")
        
        try:
            # 尝试获取处理器的配置
            config = dict(processor.config)
            logging.info("\n[Image Processor Config (JSON)]: ")
            logging.info(json.dumps(config, indent=4))
            
            # 尝试从配置中提取分辨率信息 (通常是 height/width)
            target_h = config.get('size', {}).get('height') or config.get('height')
            target_w = config.get('size', {}).get('width') or config.get('width')
            if target_h and target_w:
                 logging.info(f"\n[推测的输入像素尺寸]: H={target_h}, W={target_w}")
                 
        except Exception:
            logging.warning("[WARN] 无法从 image_processor config 中解析尺寸。")

    # ------------------------------------------------
    # 模拟 VAE 编码步骤来确认像素张量形状
    # ------------------------------------------------
    logging.info(f"--- 6. 模拟 VAE 编码输入以确认形状 ---")
    
    # ⚠️ 关键步骤：创建一个伪造的输入张量，模拟加载的视频帧
    # VAE 期望 [B, C, F, H, W] 形状的张量 (通常是 float32 或 float16/bfloat16)
    
    # 我们使用一个保守的尺寸占位符 (例如 768x1280)，你可能需要修正
    H_PXL, W_PXL = 768, 1280 
    F = 81 # 帧数
    
    # 伪造一个视频输入张量 (Batch=1, Channels=3)
    dummy_video_tensor = torch.randn(1, 3, F, H_PXL, W_PXL, dtype=torch.bfloat16)
    
    try:
        logging.info(f"尝试使用 {dummy_video_tensor.shape} 张量运行 VAE Encoder...")
        
        with torch.no_grad():
            dummy_video_tensor = dummy_video_tensor.to(pipeline.vae.dtype) # 匹配 VAE 的 dtype
            
            # 运行 VAE 编码
            latent_dist = pipeline.vae.encode(dummy_video_tensor).latent_dist
            
            # 潜在变量的形状
            latent_sample_shape = latent_dist.sample().shape 
            
            logging.info(f"[VAE Latent Output Shape]: {latent_sample_shape}")
            logging.info(f"[推测的 VAE 输入 Pixel H, W]: {H_PXL}, {W_PXL} (基于假设)")
            
    except RuntimeError as e:
        # 如果尺寸不匹配，VAE 会抛出运行时错误
        if "size mismatch" in str(e) or "Input size" in str(e):
             logging.error(f"[ERROR] VAE 输入尺寸不匹配。请根据错误信息修正 H_PXL/W_PXL。")
             logging.error(f"  原始错误: {e}")
        else:
             logging.error(f"[ERROR] VAE 编码失败: {e}")
             
    except Exception as e:
        logging.error(f"[ERROR] VAE 编码检查失败: {e}")
        
    logging.info("-" * 60)

    # ... (SCHEDULER 和 DIT 检查逻辑保持不变)
    logging.info(f"--- 3. SCHEDULER (调度器) 配置 ---")
    scheduler = pipeline.scheduler
    logging.info(f"Scheduler Type: {type(scheduler).__name__}")
    if hasattr(scheduler, 'config'):
        logging.info("\n[Scheduler Config (JSON)]: ")
        logging.info(json.dumps(dict(scheduler.config), indent=4))
    if hasattr(scheduler, 'alphas_cumprod'):
        alphas_tensor = scheduler.alphas_cumprod.cpu()
        logging.info(f"\n[Scheduler Alphas Cumprod (ᾱt)]: ")
        logging.info(f"  Shape: {alphas_tensor.shape}")
        logging.info(f"  Dtype: {alphas_tensor.dtype}")
    logging.info("-" * 60)
    
    logging.info(f"--- 4. DIT 模型配置 (用于 LoRA) ---")
    try:
        if hasattr(pipeline.transformer, 'config'):
            dit_config = pipeline.transformer.config
            logging.info(f"  Hidden Size / Embed Dim: {dit_config.get('hidden_size')}")
            logging.info(f"  Cross-Attention Dim: {dit_config.get('cross_attention_dim')}")
            logging.info(f"  Timestep/Frame Count: {dit_config.get('num_frames')}") 
    except AttributeError:
        logging.error("DIT 配置访问失败。")
        
    logging.info("---------------------------------------------------\n")

    # 显式删除 pipeline 以释放内存
    if pipeline:
        del pipeline
    
    # 正常退出
    sys.exit(0) 


if __name__ == "__main__":
    inspect_cogvideo_config()