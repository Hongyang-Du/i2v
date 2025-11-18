import torch
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

dtype = torch.bfloat16
device = "cuda"

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

height = 480
width = 720
num_frames = 49
num_inference_steps = 50
guidance_scale = 5.0



prompt = "a man playing drum"
image = load_image(image="input.jpg")
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

output = pipe(
    prompt=prompt,
    image=image,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
).frames[0]
export_to_video(output, "5bit2v_output.mp4", fps=24)
