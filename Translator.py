from transformers import MarianMTModel, MarianTokenizer

# Function to download the model without login (for public models)
def download_model(model_name):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Model for English to Hindi translation
model_name_en_to_hi = 'Helsinki-NLP/opus-mt-en-hi'  # This is the public English to Hindi model
model_en_to_hi, tokenizer_en_to_hi = download_model(model_name_en_to_hi)

# Function to translate English to Hindi
def translate_en_to_hi(text):
    # Tokenize and translate
    translated = model_en_to_hi.generate(**tokenizer_en_to_hi(text, return_tensors="pt", padding=True))
    return tokenizer_en_to_hi.decode(translated[0], skip_special_tokens=True)

# Example Usage:
english_text = "say my name"

# Translating English to Hindi
print("English to Hindi:", translate_en_to_hi(english_text))

from gtts import gTTS
from IPython.display import Audio
import os

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    # Convert text to speech
    tts = gTTS(text=text, lang=lang, slow=False)

    # Save the audio file
    tts.save("output.mp3")

    # Play the audio file in Colab
    return Audio("output.mp3")

# Example Usage:

# Example 1: Text (Tunglish or English) to Speech
tunglish_text = "Naan thaan da leo"  # Romanized Tamil (Tunglish)
english_text = "i am the one who knocks"

# Convert Tunglish (Romanized Tamil) to Speech
print("Converting Tunglish to speech...")
tunglish_audio = text_to_speech(tunglish_text, lang='en')  # You can keep lang='ta' for Tamil, but let's use English here
tunglish_audio  # This will display the audio player in Colab


# Convert English to Speech
print("Converting English to speech...")
english_audio = text_to_speech(english_text, lang='en')
english_audio  # This will display the audio player in Colab
