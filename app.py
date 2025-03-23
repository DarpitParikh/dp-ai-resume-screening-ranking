import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import numpy as np
import re
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

# Load Spacy model
spacy_model = "en_core_web_sm"
try:
    nlp = spacy.load(spacy_model)
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
    nlp = spacy.load(spacy_model)

# Load Sentence Transformer Model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Function to Extract Text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Function to Preprocess Text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

# Function to Extract Keywords
def extract_keywords(text, top_n=10):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha and not token.is_stop]
    word_freq = pd.Series(words).value_counts()
    return word_freq.head(top_n).to_dict()

# Function to Calculate Similarity
def calculate_similarity(resumes, job_desc):
    all_texts = resumes + [job_desc]
    embeddings = model.encode(all_texts, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(embeddings[:-1], embeddings[-1])
    return similarity_scores.squeeze().tolist()

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
        ranked_indices = np.argsort(similarity_scores)[::-1].tolist()
        results = pd.DataFrame({
            'Resume': [uploaded_files[i].name for i in ranked_indices],
            'Similarity Score': [similarity_scores[i] for i in ranked_indices]
        })
        st.write("### 📊 Ranked Resumes:")
        st.dataframe(results)
        
        # Recruiter Dashboard
        st.write("## 📈 Recruiter Dashboard")
        st.write(f"- **Total Resumes Processed:** {len(uploaded_files)}")
        st.write(f"- **Average Similarity Score:** {np.mean(similarity_scores):.2f}")
        
        fig, ax = plt.subplots()
        sns.histplot(similarity_scores, bins=10, kde=True, ax=ax)
        ax.set_title("Similarity Score Distribution")
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        # AI Explainability View
        st.write("## 🤖 AI Explainability View")
        job_keywords = extract_keywords(job_desc_cleaned)
        st.write("### 🔑 Top Keywords from Job Description:")
        st.json(job_keywords)
        
        for i, idx in enumerate(ranked_indices):
            resume_keywords = extract_keywords(resumes_cleaned[idx])
            st.write(f"### 📄 {uploaded_files[idx].name}")
            st.write("**Top Keywords in Resume:**")
            st.json(resume_keywords)
            matching_keywords = set(job_keywords.keys()) & set(resume_keywords.keys())
            st.write(f"**Matching Keywords with Job Description:** {list(matching_keywords)}")
    else:
        st.warning("⚠ Please upload resumes and enter a job description.")
