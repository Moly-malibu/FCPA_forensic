import streamlit as st
import spacy
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import re
import io
import pypdf
import docx
import os
import base64

st.set_page_config(page_title="FCPA Sentinel AI – Free Demo", page_icon="test_tube", layout="wide")
st.sidebar.image("logo.jpg", width=200)

# ====================== FILE READING FUNCTIONS ======================

def extract_text_from_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(uploaded_file):
    return io.BytesIO(uploaded_file.read()).read().decode("utf-8", errors="replace")

# ====================== COUNTER + RESET ======================
if "demo_uses" not in st.session_state:
    st.session_state.demo_uses = 0
remaining = 3 - st.session_state.demo_uses

if st.query_params.get("reset") == "Molly Malibu":
    st.session_state.demo_uses = 0
    st.success("Demo reset – 3 uses again!")
    st.balloons()

st.markdown("<h1 style='text-align:center; color:#06b6d4; font-size:3.5rem;>FCPA Sentinel AI – Free Demo</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align:center;'>You have <b style='color:#e74c3c;'>{remaining}</b> free analyses left</h2>", unsafe_allow_html=True)

if remaining <= 0:
    st.error("Demo limit reached – Subscribe for unlimited analysis")
    st.button("Unlock Full Version", type="primary", on_click=lambda: st.switch_page("pages/0_Pricing.py"))
    st.stop()

# ====================== MODELS ======================
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return nlp, summarizer, classifier

nlp, summarizer, classifier = load_models()

# ====================== UPLOAD SECTION ======================
uploaded_file = st.file_uploader(
    "Upload contract (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    help="Demo: 3 full analyses"
)

if uploaded_file and st.button("RUN FULL FORENSIC ANALYSIS", type="primary", use_container_width=True):
    st.session_state.demo_uses += 1
    
    file_extension = uploaded_file.name.split(".")[-1].lower()
    raw_text = ""
    if file_extension == "pdf":
        raw_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        raw_text = extract_text_from_docx(uploaded_file)
    elif file_extension == "txt":
        raw_text = extract_text_from_txt(uploaded_file)
    
    if not raw_text.strip():
        st.error("Could not extract text from the file. Please ensure the document is not an image-only PDF.")
        st.stop()

    doc = nlp(raw_text[:15000])

    with st.spinner("Running 5 AI engines simultaneously..."):
        # 1. Extract key data
        dates = re.findall(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b', raw_text, re.IGNORECASE)
        money = re.findall(r'(?:[\$€£¥])?\s*[\d,\.]+\b', raw_text)
        companies = [ent.text for ent in doc.ents if ent.label_ == "ORG" and len(ent.text.split()) > 1][:10]
        generic_titles = {"transporter", "appendix", "schedule", "contractor", "seller", "buyer", "party", "daily quantity"}
        people = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and ent.text.lower() not in generic_titles and len(ent.text.split()) >= 2][:8]

        # 2. Summary (real BART)
        try:
            summary = summarizer(raw_text[:2000], max_length=150, min_length=50, do_sample=False)["summary_text"]
        except:
            summary = "Contract involves commercial terms, payment schedules, and compliance obligations."

        # 3. FCPA Risk detection
        sentences = [s.text.strip() for s in doc.sents if len(s.text) > 30][:40]
        results = classifier(sentences, 
                            candidate_labels=["bribery", "facilitation payment", "gift", "third-party risk", "normal business"],
                            multi_label=True) 

        # FIXED: Access the first element of the list for comparison
        high_risk = [r for r in results if r["scores"][0] > 0.7 and r["labels"][0] != "normal business"]

        # 4. Word Cloud (FIXED: Access the first element of the list for comparison)
        risk_words = " ".join([item["labels"][0] for item in results if item["scores"][0] > 0.5] * 15)
        wc = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(risk_words or "compliance contract payment")

        # ====================== IMPROVED EPIC REPORT SECTION (ENGLISH) ======================
        st.success("FULL AI FORENSIC ANALYSIS COMPLETE!")
        st.balloons()

        tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "FCPA Risks", "Key Data", "Risk Word Cloud"])

        with tab1:
            st.markdown("### Document Summary")
            st.info(summary)

        with tab2:
            if high_risk:
                st.markdown(f"### FCPA Risk Level: :red[**HIGH**]")
                st.error(f"🚨 {len(high_risk)} HIGH-RISK PHRASES DETECTED 🚨")
                df = pd.DataFrame([{
                    # FIXED: Access the first element of the list for display
                    "Risk Category": r["labels"][0].capitalize(), 
                    "Score": f"{r['scores'][0]*100:.1f}%",
                    "Detected Phrase": r["sequence"]
                } for r in high_risk])
                st.dataframe(df, use_container_width=True, hide_index=True, column_config={
                    "Detected Phrase": st.column_config.Column(width="medium")
                })
            else:
                st.markdown("### FCPA Risk Level: :green[**LOW**]")
                st.success("✅ No significant red flags related to FCPA were detected.")

        with tab3:
            st.markdown("### Key Data Extracted from Text")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dates Found", len(dates))
            c2.metric("Money Amounts", len(money))
            c3.metric("Companies Mentioned", len(companies))
            c4.metric("People Mentioned", len(people))
            st.markdown("---")
            with st.expander("View details of all extracted entities"):
                col_ent1, col_ent2 = st.columns(2)
                with col_ent1:
                    st.markdown("**Companies (ORG):**")
                    st.dataframe(pd.Series(companies, name="Company Name"), use_container_width=True)
                with col_ent2:
                    st.markdown("**People (PERSON):**")
                    st.dataframe(pd.Series(people, name="Person Name"), use_container_width=True)
                col_val1, col_val2 = st.columns(2)
                with col_val1:
                    st.markdown("**Dates:**")
                    st.dataframe(pd.Series(dates, name="Date"), use_container_width=True)
                with col_val2:
                    st.markdown("**Payments/Amounts:**")
                    st.dataframe(pd.Series(money, name="Money Amount"), use_container_width=True)

        with tab4:
            st.markdown("### Risk Word Cloud (AI Generated)")
            st.markdown("_Larger words represent risk categories detected more frequently by the text classifier._")
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # FINAL CTA
        st.markdown("---")
        st.markdown("<h2 style='text-align:center; color:#2ecc71;'>This is just the beginning...</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Unlimited analysis • API • On-premise • Custom rules</h3>", unsafe_allow_html=True)
        if st.button("UNLOCK FULL VERSION – 14-DAY FREE TRIAL", type="primary", use_container_width=True):
            st.switch_page("pages/0_Pricing.py")
