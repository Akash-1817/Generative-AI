pip install diffusers transformers accelerate safetensors
pip install --upgrade Pillow requests

import torch
from diffusers import StableDiffusionXLPipeline, KDPM2AncestralDiscreteScheduler, AutoencoderKL
from PIL import ImageDraw, ImageFont, Image
import requests
import json
import time

# 🔑 OpenRouter API key
API_KEY = ""  # <-- Replace this

# 🚀 Load VAE and anime model
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "enhanceaiteam/AnimeSAI",
    vae=vae,
    torch_dtype=torch.float16
)
pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

# 📘 Story generation using OpenRouter
def generate_page(prompt, page_number):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a creative story writer. Write vivid and emotional stories with strong character development."},
            {"role": "user", "content": f"Write page {page_number} of a 10-page story (around 1000 words) based on: '{prompt}'."}
        ],
        "temperature": 0.9,
        "max_tokens": 1300
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
        result = response.json()
        if 'choices' not in result:
            print(f"❌ Error on page {page_number}:\n{json.dumps(result, indent=2)}")
            return f"[Error generating page {page_number}]"
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Exception: {e}")
        return f"[Exception generating page {page_number}]"

def generate_story(prompt, title="My Story"):
    story = []
    print(f"\n📖 Generating story: {title}")
    for i in range(1, 11):
        print(f"✍️ Generating page {i}...")
        page = generate_page(prompt, i)
        story.append(f"\n--- Page {i} ---\n{page}")
        time.sleep(1)

    filename = f"{title.replace(' ', '_').lower()}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(story))
    print(f"\n✅ Story saved as: {filename}")

def generate_image(prompt, title):
    print("\n🖼️ Generating cover image...")
    image = pipe(
        prompt,
        negative_prompt="low quality, blurry",
        width=1024,
        height=1024,
        guidance_scale=7,
        num_inference_steps=50,
        clip_skip=2
    ).images[0]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Center the title at bottom
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    W, H = image.size

    draw.text(
        ((W - text_width) / 2, H - text_height - 20),
        title,
        fill="white",
        font=font
    )

    filename = f"{title.replace(' ', '_').lower()}.png"
    image.save(filename)
    print(f"✅ Cover image saved as: {filename}")

# 🧠 MAIN FLOW
if __name__ == "__main__":
    story_prompt = input("📥 Enter your story idea (1 line): ")
    story_title = input("🎨 Enter a title for your story: ")
    generate_image(story_prompt, story_title)
    generate_story(story_prompt, story_title)

