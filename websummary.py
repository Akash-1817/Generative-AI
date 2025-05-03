pip install langchain-community
pip install chromadb

import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  # Switched to free embeddings
import os

# Define available sources
AVAILABLE_SOURCES = {
    "wikipedia": "https://en.wikipedia.org/wiki/",
    "arxiv": "https://arxiv.org/search/?query=",
    "bbc_news": "https://www.bbc.co.uk/search?q=",
    "nature": "https://www.nature.com/search?q=",
    "custom": ""  # For user-defined URLs
}

# Fetch content from Wikipedia
def fetch_wikipedia(query):
    url = AVAILABLE_SOURCES["wikipedia"] + query.replace(" ", "_")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p", limit=5)
        return "\n".join([p.get_text() for p in paragraphs]) or "No relevant content found."
    except requests.RequestException as e:
        return f"Error fetching Wikipedia content: {e}"

# Fetch research content from ArXiv or Nature
def fetch_research_papers(query, source):
    if source == "arxiv":
        url = AVAILABLE_SOURCES["arxiv"] + query.replace(" ", "+")
    elif source == "nature":
        url = AVAILABLE_SOURCES["nature"] + query.replace(" ", "+")
    else:
        return "Invalid research source."
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        if source == "arxiv":
            abstracts = soup.select("span.abstract-full", limit=5)
            return "\n".join([a.get_text().strip() for a in abstracts]) or "No relevant content found."
        else:  # Nature
            paragraphs = soup.find_all("p", limit=5)
            return "\n".join([p.get_text() for p in paragraphs]) or "No relevant content found."
    except requests.RequestException as e:
        return f"Error fetching content from {source}: {e}"

# Fetch content from a custom URL
def fetch_custom(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p", limit=5)
        return "\n".join([p.get_text() for p in paragraphs]) or "No relevant content found."
    except requests.RequestException as e:
        return f"Error fetching content from {url}: {e}"

# Initialize LLM with OpenRouter, adding required headers
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    model="mistralai/mixtral-8x7b-instruct",  # Replace with correct model from OpenRouter
    temperature=0.5,
    default_headers={
        "HTTP-Referer": "http://localhost",  # Required by OpenRouter
        "X-Title": "RAG Chatbot"  # Required by OpenRouter
    }
)

# Initialize ChromaDB with free Hugging Face embeddings (no OpenAI key needed)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="dynamic_data", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Based on the following information:\n{context}\nAnswer this question: {question}"
)

# Set up RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Store content in vector database
def store_content(query, content, source):
    if "Error" not in content:
        vectorstore.add_texts(
            texts=[content],
            metadatas=[{"query": query, "source": source}]
        )

# RAG Chatbot function
def rag_chatbot(query, source):
    if source == "wikipedia":
        content = fetch_wikipedia(query)
    elif source in ["arxiv", "nature"]:
        content = fetch_research_papers(query, source)
    elif source.startswith("http"):
        content = fetch_custom(source)
    else:
        return "Invalid source. Please select Wikipedia, ArXiv, Nature, or provide a valid URL."

    store_content(query, content, source)
    response = rag_chain({"query": query})
    return response["result"]

# Test the chatbot
query = "Madras institute of technology"
source = "wikipedia"
response = rag_chatbot(query, source)
print(response)
