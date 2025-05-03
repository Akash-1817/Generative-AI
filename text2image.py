#!pip install diffusers transformers accelerate scipy safetensors

from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load pipeline from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
).to("cuda")

# Text prompt
prompt = "a japanese samurai fighting against a dragon"

# Generate image
image = pipe(prompt).images[0]

# Show image
plt.imshow(image)
plt.axis("off")
plt.title(prompt)
plt.show()

from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load pipeline from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16,
).to("cuda")

# Text prompt
prompt = "goku"

# Generate image
image = pipe(prompt).images[0]

# Show image
plt.imshow(image)
plt.axis("off")
plt.title(prompt)
plt.show()

!pip install gradio

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion 2.1 model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

# Define the generation function
def generate_image(prompt):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

# Build the Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt", placeholder="e.g., a castle in the clouds at sunset"),
    outputs=gr.Image(type="pil"),
    title="🧠 Text-to-Image Generator",
    description="Powered by stabilityai/stable-diffusion-2-1 on Hugging Face"
)

# Launch the app
demo.launch()


