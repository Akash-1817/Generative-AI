<a href="https://colab.research.google.com/github/gosaitos/GEN-AI/blob/main/Chatbot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#!pip install langchain openai openrouter dotenv
#!pip install gradio openai
import requests

API_KEY = ""

headers = {"Authorization": f"Bearer {API_KEY}"}
response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)

if response.status_code == 200:
    free_models = [model["id"] for model in response.json()["data"] if ":free" in model["id"]]

    if free_models:
        print("✅ Available Free Models:")
        for model in free_models:
            print("-", model)
    else:
        print("⚠️ No free models found.")
else:
    print("❌ Error:", response.json())

import os

# Set your OpenRouter API key
os.environ["OPENROUTER_API_KEY"] = ""

import openai
import gradio as gr
import os

# Ensure OpenRouter API key is set
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("Missing OpenRouter API Key! Set it using os.environ.")

# Set OpenAI-compatible API details
openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# Pick a valid OpenRouter model
model = "deepseek/deepseek-r1-distill-qwen-32b:free"

# Define chat history
messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

# Define function to interact with chatbot
def chat(user_input):
    global messages

    messages.append({"role": "user", "content": user_input})

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )

        ai_response = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": ai_response})
        return ai_response

    except Exception as e:
        return f"❌ Error: {e}"

# Create Gradio UI
iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Chatbot using OpenRouter API",
    description="Ask me anything!",
)

# Launch the chatbot
iface.launch()

import openai
import os

# Ensure OpenRouter API key is set
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("Missing OpenRouter API Key! Set it using os.environ.")

# Set OpenAI-compatible API details
openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# Define conversation history
messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

while True:
    user_input = input("User: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})

    try:
        # OpenAI-compatible API call
        response = openai.ChatCompletion.create(
            #model="meta-llama/llama-3-8b-instruct:free",
            model = "deepseek/deepseek-r1-distill-qwen-32b:free",# Change model if needed
            messages=messages
        )

        # Print full response for debugging
        #print("🔍 FULL RESPONSE:", response)

        # Extract AI response safely
        if "choices" in response and response["choices"]:
            ai_response = response["choices"][0]["message"]["content"]
            print("Chatbot:", ai_response)

            # Append AI response to chat history
            messages.append({"role": "assistant", "content": ai_response})
        else:
            print("⚠️ No 'choices' found in response. Check the API response format.")

    except Exception as e:
        print("❌ Error:", e)
