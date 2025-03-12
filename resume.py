# -*- coding: utf-8 -*-
"""AI Resume Screening & Ranking System"""

# Install necessary libraries
import streamlit as st
import PyPDF2
import nltk
import spacy
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# Download required NLTK & Spacy models
import nltk.data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    raise RuntimeError("NLTK 'punkt' tokenizer is missing. Ensure it is pre-installed.")


import os
import spacy
import subprocess

# Ensure Spacy model is installed
spacy_model = "en_core_web_sm"

nlp = spacy.load("en_core_web_sm")  # Assume model is pre-installed



# Load a powerful embedding model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Function to clean and preprocess text
def preprocess_text(text):
    """Cleans and preprocesses text for better similarity matching."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to calculate semantic similarity using Sentence Transformers
def calculate_similarity(resumes, job_desc):
    """Uses sentence embeddings to measure meaning-based similarity."""
    all_texts = resumes + [job_desc]  # Combine resumes & job description
    embeddings = model.encode(all_texts, convert_to_tensor=True)  # Convert text to embeddings
    similarity_scores = util.pytorch_cos_sim(embeddings[:-1], embeddings[-1])  # Compare with job description
    return similarity_scores.squeeze().tolist()  # Return similarity scores

# Streamlit UI
st.title("📄 AI Resume Screening & Ranking System")

uploaded_files = st.file_uploader("Upload Resumes (Multiple PDFs allowed)", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")

if st.button("Rank Resumes"):
    if uploaded_files and job_description:
        resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
        resumes_cleaned = [preprocess_text(resume) for resume in resumes_text]
        job_desc_cleaned = preprocess_text(job_description)

        similarity_scores = calculate_similarity(resumes_cleaned, job_desc_cleaned)
        ranked_indices = np.argsort(similarity_scores)[::-1].tolist()  # Convert to a Python list

        results = pd.DataFrame({
            'Resume': [uploaded_files[i].name for i in ranked_indices],  # Ensure index is an integer
            'Similarity Score': [similarity_scores[i] for i in ranked_indices]
        })

        st.write("### 📊 Ranked Resumes:")
        st.dataframe(results)

    else:
        st.warning("⚠ Please upload resumes and enter a job description.")
