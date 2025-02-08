import torch
from diffusers import AutoPipelineForText2Image, ControlNetModel
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the controlnet and pipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    cache_dir="/scratch/ssd004/scratch/seyedmat/huggingface", # required for not exceeding Vector quota. Comment otherwise
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    enable_pag=True,
    pag_applied_layers=["mid"],  # Default, can be modified in loop
    torch_dtype=torch.float16,
    cache_dir="/scratch/ssd004/scratch/seyedmat/huggingface", # required for not exceeding Vector quota. Comment otherwise
)
pipeline.enable_model_cpu_offload()

# Load the canny image for conditioning
canny_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/pag_control_input.png")

# Define your experiment parameters
seeds = [0, 42, 123]
pag_scales = [0.0, 3.0, 5.0]  # Different scales for PAG
guidance_scales = [5.0, 7.0, 10.0]  # Guidance scales for experimentation
controlnet_conditioning_scales = [0.5, 1.0, 1.5, 2.0]  # Added 1.5 here
pag_layer_combinations = [
    ["mid"],  # Experiment with layers that can be applied
]

# Initialize an empty list to store images for the grid
all_images = []

# Iterate through combinations of seed, pag_scale, guidance_scale, controlnet_conditioning_scale, and pag_layer_combinations
for seed in seeds:
    for pag_scale in pag_scales:
        for guidance_scale in guidance_scales:
            for controlnet_conditioning_scale in controlnet_conditioning_scales:
                for pag_layers in pag_layer_combinations:
                    # Set the pag_applied_layers dynamically based on the current experiment
                    pipeline.pag_applied_layers = pag_layers
                    
                    # Generate an image using the current combination of parameters
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    images = pipeline(
                        prompt="",
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        image=canny_image,
                        num_inference_steps=50,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        pag_scale=pag_scale,
                    ).images

                    # Store the generated image
                    for img in images:
                        all_images.append(img)

# Create a grid of images
num_images = len(all_images)
grid_size = int(np.ceil(np.sqrt(num_images)))  # Find grid size that fits all images
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))

# Remove empty subplots
for ax in axes.flat:
    ax.axis('off')

# Fill the grid with images
image_idx = 0
for i in range(grid_size):
    for j in range(grid_size):
        if image_idx < num_images:
            ax = axes[i, j]
            ax.imshow(all_images[image_idx])
            image_idx += 1

# Save the grid of images
plt.tight_layout()
plt.savefig("generated_image_controlnet_grid_with_conditioning_scale.png")
# plt.show()