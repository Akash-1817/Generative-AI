#!pip install langchain openai wikipedia
#!pip install langchain-community

import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper

# Set OpenRouter API Key
os.environ["OPENAI_API_KEY"] = ""

# Use OpenRouter API
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mistral-7b-instruct",  # Choose a model from OpenRouter's available options
    temperature=0.7
)

wiki = WikipediaAPIWrapper()

def search_wikipedia(query):
    try:
        result = wiki.run(query)
        return result
    except Exception as e:
        return f"Error fetching Wikipedia content: {e}"

prompt = PromptTemplate(
    input_variables=["query", "wiki_info"],
    template="You are a Wikipedia expert. Based on the query '{query}', provide a response using this Wikipedia information:\n\n{wiki_info}"
)

chain = LLMChain(llm=llm, prompt=prompt)

def wikipedia_chatbot(query):
    wiki_info = search_wikipedia(query)
    response = chain.run(query=query, wiki_info=wiki_info)
    return response

query = "Anna university"
response = wikipedia_chatbot(query)
print(response)

import gradio as gr
import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.generativeai import configure

# ✅ Set your Google Gemini API Key
GOOGLE_API_KEY = ""
configure(api_key=GOOGLE_API_KEY)

# ✅ Function to create a Wikipedia-based retriever
def create_wikipedia_retriever(query):
    try:
        loader = WikipediaLoader(query=query, lang="en")
        docs = loader.load()

        if not docs:
            return None  # No Wikipedia data found

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        # Generate embeddings using Google AI
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

        # Store embeddings in FAISS
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        return vectorstore.as_retriever()
    except Exception as e:
        return None  # Handle errors

# ✅ Initialize Gemini AI LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# ✅ Function to get a Wikipedia-based response
def get_response(user_query):
    retriever = create_wikipedia_retriever(user_query)

    if not retriever:
        return "No relevant data found on Wikipedia. Try another query."

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.run(user_query)

    return response if response else "Could not generate a meaningful response."

# ✅ Gradio UI
def chatbot_ui(user_input):
    return get_response(user_input)

# ✅ Launch Gradio Interface
iface = gr.Interface(
    fn=chatbot_ui,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything from Wikipedia..."),
    outputs="text",
    title="Wikipedia Chatbot (Gemini AI)",
    description="A chatbot that retrieves Wikipedia information using Gemini AI and RAG.",
    theme="compact"
)

iface.launch(share=True)  # 🔗 Generates a public link

