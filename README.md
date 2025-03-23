📄 AI-Powered Resume Screening & Ranking System
🚀 Overview
This project is an AI-powered Resume Screening and Ranking System that helps recruiters automatically evaluate and rank resumes based on their relevance to a given job description. It leverages Natural Language Processing (NLP) and Machine Learning to compute similarity scores between resumes and job descriptions.

🎯 Features
📌 Resume Parsing – Extracts text from uploaded PDFs.

🔍 AI-Based Resume Ranking – Uses sentence-transformers to compare resumes with job descriptions.

📊 Recruiter Dashboard – Displays analytics like the number of resumes processed, average similarity score, and score distribution.

🧐 AI Explainability View – Highlights key matching keywords to improve transparency in resume ranking.

🌐 Streamlit Web Interface – User-friendly and interactive UI for recruiters.

🏗️ Tech Stack
Programming Language: Python 3.x

Framework: Streamlit

NLP Libraries: Spacy, NLTK

Machine Learning: Sentence Transformers (paraphrase-MiniLM-L6-v2)

PDF Processing: PyPDF2

Data Visualization: Matplotlib, Seaborn

📦 Installation
1️⃣ Clone the Repository

git clone https://github.com/YOUR_USERNAME/Resume-Ranking-AI.git
cd Resume-Ranking-AI
2️⃣ Install Dependencies

pip install -r requirements.txt
3️⃣ Run the Application

streamlit run app.py
📌 Usage Guide
Upload multiple PDF resumes.

Enter the job description in the text box.

Click on "Rank Resumes" to get ranked results.

View analytics in the Recruiter Dashboard.

Check the Explainability View to understand AI decisions.

📊 Recruiter Dashboard
Total Resumes Processed

Average Similarity Score

Similarity Score Distribution (Histogram)

🤖 AI Explainability
Extracted Keywords from the job description.

Extracted Keywords from each resume.

Matching Keywords between the job description and resume.

🛠️ Deployment (on Hugging Face Spaces)
Create a Hugging Face Space (Streamlit template).

Upload the project files.

Ensure requirements.txt includes:

sentence-transformers
torch
transformers
streamlit
PyPDF2
spacy
matplotlib
seaborn
Restart the Space and run!

📜 License
This project is licensed under the MIT License.

🙌 Contributing
Feel free to fork this repository and submit a pull request with improvements.

📞 Contact
For queries, reach out via:
LinkedIn: [Darpit Parikh](https://www.linkedin.com/in/darpit-parikh-376734313)
