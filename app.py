import streamlit as st
import PyPDF2
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go  #For circular gauge

# Handle stopwords safely
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
    words = [w for w in words if w not in stop_words]
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

# 🔘 Button to trigger analysis
if st.button("🔍 Analyze"):
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
            rounded_score = round(score, 2)

            # ✅ Match score output with a meter
            st.success(f"✅ Resume–JD Match Score: {rounded_score}%")

            # Progress Bar (acts like a wheel meter)
            st.progress(int(rounded_score))  # value between 0 to 100
            st.metric("🎯 Match Percentage", f"{rounded_score}%", delta=None)


            # 🧭 Circular Gauge (Plotly)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rounded_score,
                title = {'text': "Resume Match Gauge"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"},
                    ],
                }
            ))
            st.plotly_chart(fig)

            # 🔍 Missing Keywords
            resume_words = set(cleaned_resume.split())
            jd_words = set(cleaned_jd.split())
            missing = jd_words - resume_words

            if missing:
                st.warning("🔍 Missing Keywords from Resume:")
                st.write(", ".join(sorted(list(missing))[:10]))
            else:
                st.info("🎯 Your resume contains all major keywords from the JD!")

        # Expanders to show full text (optional)
        with st.expander("📄 See Extracted Resume Text"):
            st.write(resume_text)

        with st.expander("📌 Job Description Used"):
            st.write(job_text_input)
    else:
        st.error("Please upload your resume and paste a job description first.")


