#!pip install PyPDF2
#!pip install -U langchain-community
#!pip install faiss-cpu
#!pip install keybert

import re
import spacy
import nltk
import os
import requests
import numpy as np
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings

# Load NLP model
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")

# OpenRouter API Key (Replace with actual key)
OPENROUTER_API_KEY = ""

# HuggingFace Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# OpenRouter API Configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "your-website.com"
}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception as e:
        print("Error extracting text from PDF:", e)
    return text.strip() if text else "No text extracted"

def extract_key_points(resume_text):
    """Extracts key points (Name, Email, Phone, Skills, Experience, and Education) from the resume."""
    doc = nlp(resume_text)

    # Extract Name
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Not Found")

    # Extract Email
    email = re.findall(r"[\w.-]+@[\w.-]+", resume_text)
    email = email[0] if email else "Not Found"

    # Extract Phone Number
    phone = re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", resume_text)
    phone = phone[0] if phone else "Not Found"

    # Extract Skills (Dynamic Matching)
    common_skills = [
        "python", "java", "c++", "javascript", "react", "angular", "node.js", "django", "flask",
        "tensorflow", "pytorch", "machine learning", "deep learning", "nlp", "data science",
        "big data", "sql", "mongodb", "postgresql", "aws", "azure", "google cloud", "docker",
        "kubernetes", "devops", "linux", "cybersecurity", "ai", "computer vision", "data analysis"
    ]
    skills = [token.text for token in doc if token.text.lower() in common_skills]
    skills_str = ", ".join(set(skills)) if skills else "Not Found"

    # Extract Experience (Flexible Matching)
    experience_matches = re.findall(r"(\d+)\s*(?:year|years)\s*of experience", resume_text, re.IGNORECASE)
    experience_str = f"{experience_matches[0]} years" if experience_matches else "Not Found"

    # Extract Education (Flexible Matching)
    education_matches = re.findall(r"(Bachelor|Master|PhD)[^\n,]*\sin\s*([\w\s]+)", resume_text, re.IGNORECASE)
    education_str = f"{education_matches[0][0]} in {education_matches[0][1]}" if education_matches else "Not Found"

    return {
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Skills": skills_str,
        "Experience": experience_str,
        "Education": education_str
    }

def match_resume_with_job(resume_text, job_description):
    """Calculates the match score between resume and job description using embeddings."""
    # Convert text to embeddings
    resume_embedding = embedding_model.embed_documents([resume_text])[0]
    job_embedding = embedding_model.embed_documents([job_description])[0]

    # Convert to NumPy arrays
    resume_embedding = np.array(resume_embedding).reshape(1, -1)
    job_embedding = np.array(job_embedding).reshape(1, -1)

    # Compute cosine similarity
    similarity = np.dot(resume_embedding, job_embedding.T) / (np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding))

    match_score = similarity[0][0] * 100  # Convert to percentage
    return round(match_score, 2)

def suggest_improvements(resume_text, job_description):
    """Uses OpenRouter API to suggest resume improvements in key-value format."""
    max_resume_length = 800  # Adjust to avoid token limit errors
    max_job_length = 500

    resume_text = resume_text[:max_resume_length]
    job_description = job_description[:max_job_length]

    payload = {
        "model": "mistralai/mistral-7b-instruct",  # Use "openai/gpt-4" if you want GPT-4
        "messages": [
            {"role": "system", "content": "You are an AI that provides resume improvement suggestions in key-value format."},
            {"role": "user", "content": f"""
            Resume Key Points:
            {extract_key_points(resume_text)}

            Job Description:
            {job_description}

            Provide key-value pairs suggesting improvements in these categories:
            - Missing Skills
            - Resume Formatting
            - Additional Experience Needed
            - Certifications to Add
            - Keywords to Include
            """}
        ],
        "max_tokens": 300  # Ensures response is within API limits
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response_data = response.json()

    if "choices" in response_data:
        return response_data["choices"][0]["message"]["content"]
    return {"Error": "Unable to generate suggestions."}


# Example Usage
file_path = "/content/resume.pdf"  # Change this to actual file path
job_description = "Data Scientist role requiring Python, NLP, and Machine Learning."

resume_text = extract_text_from_pdf(file_path)
if resume_text != "No text extracted":
    key_points = extract_key_points(resume_text)
    match_score = match_resume_with_job(resume_text, job_description)
    suggestions = suggest_improvements(resume_text, job_description)

    print("\n📌 **Extracted Resume Key Points:**")
    for key, value in key_points.items():
        print(f"{key}: {value}")

    print("\n📌 **Resume Match Score:**", match_score, "%")

    print("\n📌 **Improvement Suggestions:**")
    print(suggestions)

else:
    print("No text found in the provided resume file.")

import requests

# Replace with your OpenRouter API key
API_KEY = ""

# OpenRouter API Endpoint
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "your-website.com"  # Replace with your domain if needed
}

# Request Payload
data = {
    "model": "openai/gpt-4",  # Choose your desired model
    "messages": [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
}

# Send Request
response = requests.post(API_URL, headers=headers, json=data)

# Check Response
if response.status_code == 200:
    result = response.json()
    print("Response:", result["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code, response.json())

