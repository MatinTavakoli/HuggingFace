from diffusers import AutoPipelineForText2Image, ControlNetModel
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    cache_dir="/scratch/ssd004/scratch/seyedmat/huggingface", # required for not exceeding Vector quota. Comment otherwise
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    enable_pag=True,
    pag_applied_layers=["mid"],
    torch_dtype=torch.float16,
    cache_dir="/scratch/ssd004/scratch/seyedmat/huggingface", # required for not exceeding Vector quota. Comment otherwise
)
pipeline.enable_model_cpu_offload()

canny_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/pag_control_input.png")

for pag_scale in [0.0, 3.0]:
    generator = torch.Generator(device="cpu").manual_seed(1)
    images = pipeline(
        prompt="",
        controlnet_conditioning_scale=1.0,
        image=canny_image,
        num_inference_steps=50,
        guidance_scale=0,
        generator=generator,
        pag_scale=pag_scale,
    ).images

    # Save the images
    for idx, img in enumerate(images):
        img.save(f"generated_image_controlnet_{pag_scale}_{idx}.png")