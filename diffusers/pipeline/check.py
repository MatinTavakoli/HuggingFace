from diffusers import DiffusionPipeline
import torch

# Example: Run a text-to-image pipeline
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")  # Use "cpu" if no GPU

image = pipeline("A photo of a cat in a astronaut helmet.").images[0]
image.save("astronaut_cat.png")