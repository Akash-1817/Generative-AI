#!pip install gradio transformers torch

import gradio as gr
from transformers import pipeline

# Create a summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text):
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# Build Gradio interface
app = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, label="Input Text"),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarization App",
    description="Enter text to summarize it into a shorter version."
)

app.launch()

