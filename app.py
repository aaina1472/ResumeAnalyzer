import streamlit as st
import PyPDF2
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# Safe check: download only if missing
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# ---------- Text Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# ---------- PDF Reader ----------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# ---------- Streamlit App ----------
st.title("📄 Resume vs Job Description Analyzer")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_text_input = st.text_area("Paste the Job Description you're applying for", height=250)

if uploaded_file and job_text_input:
    with st.spinner("Analyzing Resume..."):
        # Extract and clean text
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(job_text_input)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

        # Similarity Score
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
        st.success(f"✅ Resume–JD Match Score: {round(score, 2)}%")

        # Missing Keywords
        resume_words = set(cleaned_resume.split())
        jd_words = set(cleaned_jd.split())
        missing = jd_words - resume_words

        if missing:
            st.warning("🔍 Missing Keywords from Resume:")
            st.write(", ".join(sorted(list(missing))[:10]))
        else:
            st.info("🎯 Your resume contains all major keywords from the JD!")

    # Show Resume & JD Preview (optional)
    with st.expander("📄 See Extracted Resume Text"):
        st.write(resume_text)

    with st.expander("📌 Job Description Used"):
        st.write(job_text_input)

