#!pip install transformers gradio accelerate

import gradio as gr
from transformers import pipeline

# Load a text generation pipeline with a free model
story_gen = pipeline("text-generation", model="pranavpsv/gpt2-genre-story-generator", device_map="auto", torch_dtype="auto")

# Function to generate the story
def generate_story(prompt):
    formatted_prompt = f"Write a short story based on this idea:\n{prompt}\n\nStory:"
    result = story_gen(formatted_prompt, max_length=500, do_sample=True, temperature=0.9)[0]['generated_text']
    return result.split("Story:")[-1].strip()

# Gradio UI
gr.Interface(
    fn=generate_story,
    inputs=gr.Textbox(label="Enter your story idea", placeholder="e.g., A boy discovers a dragon egg"),
    outputs=gr.Textbox(label="Generated Story"),
    title="📖 Text-to-Story Generator",
    description="Give a prompt, get a creative story! Powered by Mistral-7B from Hugging Face."
).launch()
