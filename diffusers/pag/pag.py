from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    enable_pag=True,
    pag_applied_layers=["mid"],
    torch_dtype=torch.float16,
    cache_dir="/scratch/ssd004/scratch/seyedmat/huggingface", # required for not exceeding Vector quota. Comment otherwise
)
pipeline.enable_model_cpu_offload()

prompt = "an insect robot preparing a delicious meal, anime style"

for pag_scale in [0.0, 3.0]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    images = pipeline(
        prompt=prompt,
        num_inference_steps=25,
        guidance_scale=7.0,
        generator=generator,
        pag_scale=pag_scale,
    ).images

    # Save the images
    for idx, img in enumerate(images):
        img.save(f"generated_image_pag_scale_{pag_scale}_{idx}.png")