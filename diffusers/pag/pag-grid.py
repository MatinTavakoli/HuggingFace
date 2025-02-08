import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Initialize the pipeline
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    enable_pag=True,
    pag_applied_layers=["mid"],  # Start with a default for now
    torch_dtype=torch.float16,
    cache_dir="/scratch/ssd004/scratch/seyedmat/huggingface", # required for not exceeding Vector quota. Comment otherwise
)
pipeline.enable_model_cpu_offload()

# Define your prompt and ranges for experiment
prompt = "an insect robot preparing a delicious meal, anime style"
seeds = [0, 42, 123]
pag_scales = [0.0, 3.0, 5.0]  # Different scales for PAG
guidance_scales = [5.0, 7.0, 10.0]  # Guidance scales for experimentation
pag_layer_combinations = [
    ["down.block_0"],
    ["down.block_1"],
    ["down.block_2"],
    ["down.block_3"],
    ["down.block_0.attentions_0"],
    ["down.block_1.attentions_0"],
    ["down.block_2.attentions_0"],
    ["down.block_3.attentions_0"],
    ["mid.block_0"],
    ["mid.block_1"],
    ["mid.block_2"],
    ["mid.block_0.attentions_0"],
    ["mid.block_1.attentions_0"],
    ["mid.block_2.attentions_0"],
    ["up.block_0"],
    ["up.block_1"],
    ["up.block_2"],
    ["up.block_3"],
    ["up.block_0.attentions_0"],
    ["up.block_1.attentions_0"],
    ["up.block_2.attentions_0"],
    ["up.block_3.attentions_0"],
]

# Initialize an empty list to store images and metadata
all_images = []
image_labels = []

# Iterate through combinations of seed, pag_scale, guidance_scale, and pag_layer_combinations
for seed in seeds:
    for pag_scale in pag_scales:
        for guidance_scale in guidance_scales:
            for pag_layers in pag_layer_combinations:
                # Set the pag_applied_layers dynamically based on the current experiment
                pipeline.pag_applied_layers = pag_layers
                
                # Generate an image using the current combination of parameters
                generator = torch.Generator(device="cpu").manual_seed(seed)
                images = pipeline(
                    prompt=prompt,
                    num_inference_steps=25,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    pag_scale=pag_scale,
                ).images

                # Store the generated image
                for img in images:
                    all_images.append(img)
                    image_labels.append(f"Seed: {seed}, PAG Scale: {pag_scale}, Guidance: {guidance_scale}, Layers: {pag_layers}")

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
            ax.set_title(image_labels[image_idx], fontsize=6)
            image_idx += 1

# Save the grid of images
plt.tight_layout()
plt.savefig("generated_pag_image_grid.png")
# plt.show()
