#!pip install gradio spacy
import gradio as gr
import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

def pos_tagging(text):
    doc = nlp(text)
    # Create a list of tuples containing the word and its POS tag
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

# Build Gradio interface
app = gr.Interface(
    fn=pos_tagging,
    inputs=gr.Textbox(placeholder="Enter your text here...", lines=2),
    outputs="json",
    title="Part of Speech Tagging",
    description="This app tags parts of speech in your text."
)

app.launch()

