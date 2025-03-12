import streamlit as st
import PyPDF2
import nltk
import spacy
import pandas as pd
import numpy as np
import re
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# ✅ Set a fixed NLTK data directory
NLTK_DATA_DIR = "/home/appuser/nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)

# ✅ Download necessary NLTK resources only if not found
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR)

# ✅ Ensure Spacy model is available
SPACY_MODEL = "en_core_web_sm"
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    st.error(f"⚠️ Missing SpaCy model: {SPACY_MODEL}. Run `python -m spacy download {SPACY_MODEL}`.")
    raise SystemExit

# ✅ Load an embedding model
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

# Function to calculate semantic similarity
def calculate_similarity(resumes, job_desc):
    """Uses sentence embeddings to measure meaning-based similarity."""
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

    else:
        st.warning("⚠ Please upload resumes and enter a job description.")
