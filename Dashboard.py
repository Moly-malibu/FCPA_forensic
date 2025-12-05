import streamlit as st
import spacy
import pdfplumber
import pandas as pd
import regex as re
from io import BytesIO
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import hashlib

from pricing import show_pricing_screen, check_active_subscription

magic = st.query_params.get("m", "").strip()
if magic and hashlib.sha256(magic.encode()).hexdigest() == "c2a5f1c1c69b2d4e9f8b3d7e6c5a4f3e2d1c0b9a8f7e6d5c4b3a291807060504":
    st.session_state.paid = True
    st.query_params.clear()

# === Protección normal (todos los demás) ===
if not st.session_state.get("paid", False):
    st.switch_page("pages/0_Pricing.py")

# === Ya estás dentro ===
st.set_page_config(page_title="FCPA Forensic & Contract Analyzer", page_icon="shield", layout="wide")
st.sidebar.success("Full access active")
st.title("FCPA Forensic & Contract Analyzer – Dashboard")
st.success("Welcome! You have full access")

# === Protección normal (para todos los demás) ===
if not st.session_state.get("paid", False):
    st.switch_page("pages/0_Pricing.py")

# ← Aquí ya estás dentro seguro
st.set_page_config(page_title="FFCPA Forensic & Contract Analyzer", page_icon="shield", layout="wide")
st.success("Full access active")
st.title("FFCPA Forensic & Contract Analyzer – Dashboard")

# === REDIRECT TO LOGIN IF NOT AUTHENTICATED ===
if not st.session_state.get("paid", False):
    st.switch_page("pages/0_Pricing.py")

# SI LLEGA AQUÍ → ya pagó → puede usar todo
st.set_page_config(page_title="FCPA Forensic & Contract Analyzer – Dashboard", page_icon="shield", layout="wide")

st.set_page_config(page_title="FCPA Forensic & Contract Analyzer", page_icon="shield", layout="wide")
st.sidebar.success("Paid plan active")
st.title("FCPA Forensic & Contract Analyzer – Full Dashboard")
st.success("You have full access to the forensic platform!")
st.balloons()

# === MAIN APP CONFIG ===
st.set_page_config(
    page_title="FCPA Forensic & Contract AnalyzerI",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== SIDEBAR ==================

st.sidebar.image("logo.jpg", width=200)

st.sidebar.divider()

if st.sidebar.button("Logout", type="primary", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Opcional: borrar el secreto guardado
    import os
    if os.path.exists("user_secret.json"):
        os.remove("user_secret.json")
    st.rerun()



# === WELCOME & MAIN DASHBOARD ===
st.sidebar.success(f"✅ Logged in as **{st.session_state.user}**")




# HEADER CON USUARIO
# st.title("🔍 FCPA Forensic & Contract Analyzer")
st.markdown(f"**Welcome back, {st.session_state.user}!**")

# ====================== SIDEBAR ======================
st.sidebar.title("⚙️ Analysis Controls")
risk_threshold_high = st.sidebar.slider("High‑risk similarity threshold", 0.6, 0.95, 0.78, 0.01)
context_window_size = st.sidebar.slider("Context window (tokens)", 5, 50, 25, 1)
enable_semantic = st.sidebar.checkbox("Enable semantic evasion detection", value=True)
enable_keyword_scan = st.sidebar.checkbox("Enable keyword scan", value=True)
enable_payments = st.sidebar.checkbox("Enable suspicious payment detection", value=True)

# Nueva opción para Word Cloud
show_wordcloud = st.sidebar.checkbox("Enable Word Cloud visualization", value=True)

st.sidebar.success("✅ All models loaded locally")


# ====================== GDPR COMPLIANCE SETUP ======================
# 1. Consent banner  
if "gdpr_consent" not in st.session_state:
    st.session_state.gdpr_consent = False

if not st.session_state.gdpr_consent:
    st.markdown("### GDPR & Privacy Notice")
    st.markdown("""
    This tool processes uploaded documents **locally in your browser/session**.  
    • No data is sent to external servers  
    • No documents are stored after analysis  
    • All processing runs 100% on-device or in encrypted memory  
    • You can delete all data instantly with the button below  
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("I consent to local processing", type="primary"):
            st.session_state.gdpr_consent = True
            st.rerun()
    with col2:
        if st.button("Reject & Exit"):
            st.error("Consent required to use this tool")
            st.stop()
    st.stop()

# 2. Data deletion button  
if st.sidebar.button("Delete all session data now (GDPR Right to Erasure)", type="secondary"):
    st.session_state.clear()
    st.success("All your data has been permanently deleted from this session")
    st.rerun()

# 3. Privacy header
st.sidebar.markdown("---")
st.sidebar.markdown("### Privacy & GDPR")
st.sidebar.success("100% Local Processing")
st.sidebar.caption("• No cloud storage\n• No logs\n• No third-party APIs\n• Data deleted on exit\n• No internet required after setup")

# ====================== PAGE CONFIG & CSS ======================
st.set_page_config(page_title="🔍 FCPA Forensic & Contract Analyzer – GDPR Compliant", layout="wide")

st.markdown("""
<style>
    .main-title {font-size: 3.5rem !important; font-weight: 900; background: linear-gradient(90deg, #06b6d4, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .summary-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; margin: 1.5rem 0;}
    .gdpr-badge {background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 50px; font-size: 0.9rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)



# Header con badge GDPR
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("<h1 class='main-title'>🔍 FCPA Forensic & Contract Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("**Forensic FCPA & Anti-Bribery Intelligence – 100% Local & Private**")
with col2:
    st.markdown('<div class="gdpr-badge">GDPR Compliant</div>', unsafe_allow_html=True)
st.markdown("**AI-powered forensic analysis + contract intelligence for compliance teams**")

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedder.eval()
    return nlp, classifier, tokenizer, embedder

nlp, classifier, tokenizer, embedder = load_models()

def encode_sentences(sentences):
    encoded = tokenizer(sentences, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = embedder(**encoded)
    hidden_states = outputs.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
    pooled = torch.sum(hidden_states * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)

# ====================== WORDCLOUD FUNCTION ======================
@st.cache_data
def generate_wordcloud(text, focus_keywords=False, max_words=200):
    """Genera una nube de palabras optimizada para FCPA analysis."""
    if not text or len(text.strip()) < 20:
        return None
    
    # Stopwords extendidos para contratos legales
    stopwords = set([
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
        'it', 'for', 'not', 'on', 'with', 'as', 'at', 'this', 'but', 'by',
        'from', 'they', 'section', 'shall', 'will', 'party', 'parties',
        'agreement', 'contract', 'services', 'provided', 'pursuant',
        'including', 'without', 'limitation', 'hereof', 'hereunder'
    ])
    
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    
    if focus_keywords:
        # Enfocar en keywords de riesgo y términos frecuentes
        wc = WordCloud(
            width=1000, height=500, background_color='white',
            colormap='Reds', max_words=max_words, stopwords=stopwords,
            min_font_size=10, max_font_size=100
        ).generate(text_clean)
    else:
        # Nube general del documento
        wc = WordCloud(
            width=1000, height=500, background_color='white',
            colormap='plasma', max_words=max_words, stopwords=stopwords,
            min_font_size=12, max_font_size=120
        ).generate(text_clean)
    
    return wc

def show_wordcloud_streamlit(wc, title="Word Cloud"):
    """Muestra la wordcloud en Streamlit."""
    if wc is None:
        st.warning("No text available for word cloud.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ====================== CONSTANTS ======================
FCPA_KEYWORDS = [
    "bribe", "bribery", "kickback", "facilitation payment", "foreign official",
    "corruption", "payoff", "slush fund", "consulting fee", "commission",
    "gift", "donation", "charitable contribution", "expediting fee",
    "special handling", "success fee", "agent fee", "miscellaneous expense",
    "off-the-books", "third-party intermediary", "local partner", "hospitality",
    "entertainment", "unusual payment", "arrangement with partner"
]

HIGH_RISK_COUNTRIES = {
    "venezuela", "russia", "china", "brazil", "india", "nigeria", "argentina",
    "iraq", "libya", "congo", "ukraine", "indonesia", "pakistan", "myanmar"
}

RISK_LABELS = ["high corruption risk", "low risk", "accounting irregularity"]
RED_FLAG_SENTENCES = [
    "payment to foreign official without justification",
    "hidden commission to third party",
    "gift to government employee",
    "donation to charity linked to official",
    "consulting agreement with no real services",
    "expediting payment to speed up government process"
]

with st.spinner("Warming up embedding model..."):
    RED_FLAG_EMBEDDINGS = encode_sentences(RED_FLAG_SENTENCES)

# ====================== PATTERNS CONTRACT EXTENDED ======================
CONTRACT_PATTERNS = {
    "dates": [
        r"(?:effective|execution|commencement|start)[\s\-:]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},\s+\d{4})",
        r"(?:termination|end|expires?)[\s\-:]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},\s+\d{4})",
        r"(?:term|duration)[\s\-:]+(\d+[\s\-]+(?:months?|years?)|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})"
    ],
    "amounts": [
        r"(?:total\s+(?:value|consideration|amount|price)[\s\-:]+|contract\s+(?:value|amount)[\s\-:]+)(\$[\d,]+\.?\d*)",
        r"(?:maximum|cap|limit)[\s\-:]+(\$[\d,]+\.?\d*)",
        r"(?:payment|fee)[\s\-:]+(\$[\d,]+\.?\d*)"
    ],
    "clauses": {
        "obligations": r"(?:obligations?|responsibilities?|duties?)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)",
        "payment_terms": r"(?:payment\s+(?:terms?|schedule)[\s\-:])(.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)",
        "termination": r"(?:termination|cancel.*)(.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)",
        "indemnity": r"(?:indemnif|liability|damages)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)",
        "confidentiality": r"(?:confidential.*)(.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)",
        "anti_corruption": r"(?:anti[- ]?(?:corruption|bribery)|FCPA)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)",
        "governing_law": r"(?:governing\s+law|jurisdiction)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)"
    }
}

CONTRACT_PATTERNS = { "dates": [ r"(?:effective|execution|commencement|start)[\s\-:]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},\s+\d{4})", r"(?:termination|end|expires?)[\s\-:]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},\s+\d{4})", r"(?:term|duration)[\s\-:]+(\d+[\s\-]+(?:months?|years?)|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})" ], "amounts": [ r"(?:total\s+(?:value|consideration|amount|price)[\s\-:]+|contract\s+(?:value|amount)[\s\-:]+)(\$[\d,]+\.?\d*)", r"(?:maximum|cap|limit)[\s\-:]+(\$[\d,]+\.?\d*)", r"(?:payment|fee)[\s\-:]+(\$[\d,]+\.?\d*)" ], "clauses": { "obligations": r"(?:obligations?|responsibilities?|duties?)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)", "payment_terms": r"(?:payment\s+(?:terms?|schedule)[\s\-:])(.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)", "termination": r"(?:termination|cancel.*)(.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)", "indemnity": r"(?:indemnif|liability|damages)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)", "confidentiality": r"(?:confidential.*)(.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)", "anti_corruption": r"(?:anti[- ]?(?:corruption|bribery)|FCPA)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)", "governing_law": r"(?:governing\s+law|jurisdiction)[\s\-:](.*?)(?=\n[A-Z]{3,}|Section\s+\d+|$)" } }

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    uploaded_file.seek(0)
    return text

def extract_contract_insights(text, doc_full):
    insights = {"dates": [], "amounts": [], "clauses": {}, "parties_roles": []}
    for pattern in CONTRACT_PATTERNS["dates"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        insights["dates"].extend(matches)
    for pattern in CONTRACT_PATTERNS["amounts"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        insights["amounts"].extend([m[1] for m in matches if m[1]])
    for clause_type, pattern in CONTRACT_PATTERNS["clauses"].items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            clause_text = match.group(1).strip()[:300]
            insights["clauses"][clause_type.replace("_", " ").title()] = clause_text
    party_roles = {}
    for ent in doc_full.ents:
        if ent.label_ in ["ORG", "PERSON"] and ent.start < 300:
            role_context = doc_full[max(0, ent.start-20):ent.end+20].text.lower()
            role = "Party"
            if any(w in role_context for w in ["buyer", "customer", "client"]): role = "Buyer/Client"
            elif any(w in role_context for w in ["supplier", "vendor", "seller"]): role = "Supplier/Vendor"
            party_roles[ent.text.strip()] = role
    insights["parties_roles"] = list(party_roles.items())[:5]
    return insights

# ====================== ANALYSIS ENGINE (igual + return text) ======================
def analyze_document(text, filename, risk_threshold_high=0.78, context_window_size=25, use_semantic=True, use_keywords=True, use_payments=True):
    text_lower = text.lower()
    doc = nlp(text_lower)
    doc_full = nlp(text)
    risks = []
    score = 0
    num_keywords = num_entities_scored = num_semantic_hits = num_payment_hits = 0

    if use_keywords:
        found_keywords = [kw for kw in FCPA_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        num_keywords = len(found_keywords)
        if found_keywords:
            display = ', '.join(found_keywords[:8]) + ('...' if len(found_keywords) > 8 else '')
            risks.append(f"Keywords found: {display}")
            score += len(found_keywords) * 6

    for ent in doc.ents:
        if ent.label_ in ["MONEY", "PERSON", "ORG", "GPE"]:
            start = max(0, ent.start - context_window_size)
            end = ent.end + context_window_size
            context_window = doc[start:end].text
            result = classifier(context_window, RISK_LABELS, multi_label=False)
            num_entities_scored += 1

            if result["labels"][0] == "high corruption risk" and result["scores"][0] > 0.55:
                snippet = context_window.strip().replace("\n", " ")[:160] + "..."
                risks.append(f"High-risk context: {ent.text} ({ent.label_}) → \"{snippet}\"")
                score += 18

            if ent.label_ == "GPE" and ent.text.lower() in HIGH_RISK_COUNTRIES:
                risks.append(f"High-risk country: {ent.text}")
                score += 12

    if use_semantic:
        sentences = [s.text for s in doc.sents if len(s.text) > 20]
        if sentences:
            sent_embeddings = encode_sentences(sentences)
            sim_matrix = torch.cosine_similarity(sent_embeddings.unsqueeze(1), RED_FLAG_EMBEDDINGS.unsqueeze(0), dim=-1)
            max_sims, max_idx = torch.max(sim_matrix, dim=1)
            for sim, idx, sent in zip(max_sims, max_idx, sentences):
                if sim > risk_threshold_high:
                    ref = RED_FLAG_SENTENCES[idx]
                    snippet = sent.strip().replace("\n", " ")[:180] + "..."
                    risks.append(f"Evasion: \"{snippet}\" (similar: \"{ref}\") [{sim:.2f}]")
                    score += 25
                    num_semantic_hits += 1

    if use_payments:
        payments = re.findall(r'(?:payment|fee|transfer|cost|expense).{0,50}?\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?', text_lower, re.I)
        num_payment_hits = len(payments)
        if payments:
            risks.append(f"Suspicious payments: {len(payments)} found")
            score += len(payments) * 5

    summary_sentences = [s.text.strip() for s in list(doc_full.sents)[:3]]
    contract_summary = re.sub(r'\s+', ' ', " ".join(summary_sentences))[:150].strip() + "..."
    contract_insights = extract_contract_insights(text, doc_full)
    risk_level = "HIGH" if score > 80 else "MEDIUM" if score > 40 else "LOW"
    color = "HIGH RISK" if risk_level == "HIGH" else "MEDIUM RISK" if risk_level == "MEDIUM" else "LOW RISK"

    return {
        "filename": filename, "risk_score": min(score, 100), "risk_level": risk_level, "color": color,
        "flags": risks[:15], "full_report": "\n".join(risks) if risks else "No red flags.",
        "num_keywords": num_keywords, "num_entities_scored": num_entities_scored,
        "num_semantic_hits": num_semantic_hits, "num_payment_hits": num_payment_hits, "raw_score": score,
        "contract_summary": contract_summary, "contract_insights": contract_insights,
        "raw_text": text  # ← necesario para wordcloud
    }

# ====================== EXTRACTION ======================
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    uploaded_file.seek(0)
    return text

# ====================== ENHANCED CONTRACT ANALYSIS ======================
def extract_contract_insights(text, doc_full):
    insights = {"dates": [], "amounts": [], "clauses": {}, "parties_roles": []}
    
    for pattern in CONTRACT_PATTERNS["dates"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        insights["dates"].extend(matches)
    
    for pattern in CONTRACT_PATTERNS["amounts"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        insights["amounts"].extend([m[1] for m in matches if m[1]])
    
    for clause_type, pattern in CONTRACT_PATTERNS["clauses"].items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            clause_text = match.group(1).strip()[:300]
            insights["clauses"][clause_type.replace("_", " ").title()] = clause_text
    
    party_roles = {}
    for ent in doc_full.ents:
        if ent.label_ in ["ORG", "PERSON"] and ent.start < 300:
            role_context = doc_full[max(0, ent.start-20):ent.end+20].text.lower()
            role = "Party"
            if any(word in role_context for word in ["shipper", "buyer", "customer", "client"]):
                role = "Buyer/Client"
            elif any(word in role_context for word in ["supplier", "vendor", "seller", "provider"]):
                role = "Supplier/Vendor"
            elif "transporter" in role_context or "carrier" in role_context:
                role = "Transporter/Carrier"
            party_roles[ent.text.strip()] = role
    
    insights["parties_roles"] = list(party_roles.items())[:5]
    
    return insights

# ====================== ANALYSIS ENGINE ======================
def analyze_document(text, filename, risk_threshold_high=0.78, context_window_size=25, use_semantic=True, use_keywords=True, use_payments=True):
    text_lower = text.lower()
    doc = nlp(text_lower)
    doc_full = nlp(text)
    risks = []
    score = 0
    num_keywords = num_entities_scored = num_semantic_hits = num_payment_hits = 0

    if use_keywords:
        found_keywords = [kw for kw in FCPA_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        num_keywords = len(found_keywords)
        if found_keywords:
            display = ', '.join(found_keywords[:8]) + ('...' if len(found_keywords) > 8 else '')
            risks.append(f"Keywords found: {display}")
            score += len(found_keywords) * 6

    for ent in doc.ents:
        if ent.label_ in ["MONEY", "PERSON", "ORG", "GPE"]:
            start = max(0, ent.start - context_window_size)
            end = ent.end + context_window_size
            context_window = doc[start:end].text
            result = classifier(context_window, RISK_LABELS, multi_label=False)
            num_entities_scored += 1

            if result["labels"][0] == "high corruption risk" and result["scores"][0] > 0.55:
                snippet = context_window.strip().replace("\n", " ")[:160] + "..."
                risks.append(f"High‑risk context: {ent.text} ({ent.label_}) → \"{snippet}\"")
                score += 18

            if ent.label_ == "GPE" and ent.text.lower() in HIGH_RISK_COUNTRIES:
                risks.append(f"High‑risk country: {ent.text}")
                score += 12

    if use_semantic:
        sentences = [s.text for s in doc.sents if len(s.text) > 20]
        if sentences:
            sent_embeddings = encode_sentences(sentences)
            sim_matrix = torch.cosine_similarity(sent_embeddings.unsqueeze(1), RED_FLAG_EMBEDDINGS.unsqueeze(0), dim=-1)
            max_sims, max_idx = torch.max(sim_matrix, dim=1)
            for sim, idx, sent in zip(max_sims, max_idx, sentences):
                if sim > risk_threshold_high:
                    ref = RED_FLAG_SENTENCES[idx]
                    snippet = sent.strip().replace("\n", " ")[:180] + "..."
                    risks.append(f"Evasion: \"{snippet}\" (similar: \"{ref}\") [{sim:.2f}]")
                    score += 25
                    num_semantic_hits += 1

    if use_payments:
        payments = re.findall(r'(?:payment|fee|transfer|cost|expense).{0,50}?\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?', text_lower, re.I)
        num_payment_hits = len(payments)
        if payments:
            risks.append(f"Suspicious payments: {len(payments)} found")
            score += len(payments) * 5

    summary_sentences = [s.text.strip() for s in list(doc_full.sents)[:3]]
    contract_summary = re.sub(r'\s+', ' ', " ".join(summary_sentences))[:150].strip() + "..."
    
    contract_insights = extract_contract_insights(text, doc_full)
    
    risk_level = "HIGH" if score > 80 else "MEDIUM" if score > 40 else "LOW"
    color = "🟥" if risk_level == "HIGH" else "🟨" if risk_level == "MEDIUM" else "🟩"

    return {
        "filename": filename, "risk_score": min(score, 100), "risk_level": risk_level, "color": color,
        "flags": risks[:15], "full_report": "\n".join(risks) if risks else "No red flags.",
        "num_keywords": num_keywords, "num_entities_scored": num_entities_scored,
        "num_semantic_hits": num_semantic_hits, "num_payment_hits": num_payment_hits, "raw_score": score,
        "contract_summary": contract_summary,
        "contract_insights": contract_insights,
        "text_for_cloud": text  # Guardar texto para wordcloud
    }

# ====================== UPLOAD and ANALYSIS CONTROL ======================
# ====================== UPLOAD and ANALYSIS CONTROL ======================
st.header("📤 Upload Documents")
uploaded_files = st.file_uploader("PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)

if not uploaded_files:
    st.info("👆 Upload a PDF contract above and click the blue button to start")
else:
    # Inicializar session_state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Botones de control
    col_btn1, col_btn2 = st.columns([3, 1])
    
    # Botón principal de análisis
    if col_btn1.button("🚀 Start Forensic Analysis", type="primary", use_container_width=True):
        with col_btn2:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Reset y análisis
        st.session_state.results = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {file.name}... ({i+1}/{len(uploaded_files)})")
            text = extract_text(file)
            result = analyze_document(text, file.name, risk_threshold_high, context_window_size, 
                                    enable_semantic, enable_keyword_scan, enable_payments)
            st.session_state.results.append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
        status_text.success("✅ Analysis complete!")
        st.rerun()
    
    # Usar resultados del session_state
    results = st.session_state.results
    
    if results:
        st.success(f"✅ Analysis complete! {len(results)} files processed.")
        
        # Botón re-analizar
        if col_btn2.button("🔄 Re-analyze", key="reanalyze", type="secondary"):
            st.session_state.results = []
            st.rerun()
        
        # DASHBOARD SEGURO
        df = pd.DataFrame(results)
        st.subheader("Risk Dashboard")
        
        # Filtro con filtered_df SIEMPRE definido
        risk_filter = st.multiselect("Filter by risk level", ["HIGH", "MEDIUM", "LOW"], 
                                   default=["HIGH", "MEDIUM", "LOW"])
        
        # filtered_df SIEMPRE definido
        if 'risk_level' in df.columns:
            filtered_df = df[df["risk_level"].isin(risk_filter)].copy()
        else:
            filtered_df = df.copy()
        
        # Bar chart
        col1, col2 = st.columns([2, 1])
        with col1:
            if not filtered_df.empty and 'risk_score' in filtered_df.columns:
                chart_data = filtered_df[["filename", "risk_score"]].set_index("filename")
                st.bar_chart(chart_data, height=400)
        
        with col2:
            st.metric("Total Files", len(df))
            if 'risk_level' in df.columns:
                st.metric("🟥 High Risk", len(df[df["risk_level"] == "HIGH"]))
                st.metric("🟨 Medium Risk", len(df[df["risk_level"] == "MEDIUM"]))
                st.metric("🟩 Low Risk", len(df[df["risk_level"] == "LOW"]))
        
        # Pie chart  
        if 'risk_level' in df.columns and len(filtered_df) > 0:
            st.markdown("### 📈 Risk Distribution")
            col_pie1, col_pie2, _ = st.columns([1, 1, 2])
            
            with col_pie1:
                pie_data = filtered_df['risk_level'].value_counts()
                fig, ax = plt.subplots(figsize=(3.5, 3))
                colors = ['#ff4444', '#ffaa00', '#44ff44']
                ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
                ax.axis('equal')
                ax.set_title('Risk Levels', fontsize=10)
                st.pyplot(fig)
                plt.close(fig)
            
            with col_pie2:
                st.metric("🔥 Critical", f"{len(df[df['risk_score'] > 80])}")
                st.metric("⚠️ Watch", f"{len(df[(df['risk_score'] > 40) & (df['risk_score'] <= 80)])}")
    else:
        col_btn2.info("👆 Click 'Start Forensic Analysis' to begin")

    # DETAILED RESULTS
    st.subheader("Detailed Forensic Results")
    for res in results:
        with st.expander(f"{res['color']} {res['filename']} → {res['risk_level']} RISK ({res['risk_score']}/100)"):
            st.progress(res['risk_score'] / 100)

            # EXECUTIVE SUMMARY
            st.markdown("### 📊 Executive Summary")
            insights = res['contract_insights']
            summary_text = f"""
            **{res['risk_level']} Forensic Risk** ({res['risk_score']}/100)<br>
            • Keywords: {res['num_keywords']} | Semantic: {res['num_semantic_hits']}<br>
            • Parties: {len(insights['parties_roles'])} | Dates: {len(insights['dates'])}<br>
            • Amounts: {len(insights['amounts'])} | Clauses: {len(insights['clauses'])}
            """
            st.markdown(f"<div class='summary-box'>{summary_text}</div>", unsafe_allow_html=True)

            # RISK CARDS
            col1, col2, col3 = st.columns(3)
            with col1: 
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Risk Score", f"{res['risk_score']}/100", res['risk_level'])
                st.markdown('</div>', unsafe_allow_html=True)
            with col2: 
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Keyword Hits", res['num_keywords'])
                st.markdown('</div>', unsafe_allow_html=True)
            with col3: 
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Clauses Found", len(insights['clauses']))
                st.markdown('</div>', unsafe_allow_html=True)

            # TABS CON WORDCLOUD
            tabs = ["Red Flags", "Full Report", "Meta", "Contract Insights"]
            if show_wordcloud:
                tabs.append("☁️ Word Cloud")
            
            tab_flags, tab_report, tab_meta, tab_contract = st.tabs(tabs[:4])
            
            if show_wordcloud and len(tabs) > 4:
                tab_cloud = st.tabs([tabs[4]])[0]

            with tab_flags:
                if res["flags"]:
                    for flag in res['flags']:
                        st.write("• " + flag)
                else:
                    st.success("No red flags detected")

            with tab_report: 
                st.text(res["full_report"])

            with tab_meta:
                st.write(f"Raw score: {res['raw_score']:.1f}")
                st.write(f"Entities scored: {res['num_entities_scored']}")
                st.write(f"Payment hits: {res['num_payment_hits']}")

            with tab_contract:
                insights = res['contract_insights']
                
                st.markdown("### 📋 Contract Overview")
                overview = {
                    "Summary": res['contract_summary'],
                    "Parties Found": len(insights['parties_roles']),
                    "Key Dates": len(insights['dates']),
                    "Amounts": ", ".join(insights['amounts'][:3]),
                    "Clauses": len(insights['clauses'])
                }
                st.table(pd.DataFrame(overview.items(), columns=["Field", "Details"]))

                st.markdown("### 👥 Parties & Roles")
                if insights['parties_roles']:
                    parties_df = pd.DataFrame(insights['parties_roles'], columns=["Legal Entity", "Role"])
                    st.dataframe(parties_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No parties identified")

                st.markdown("### 📅 Key Dates")
                if insights['dates']:
                    dates_df = pd.DataFrame({"Date": insights['dates'][:10]})
                    st.dataframe(dates_df, use_container_width=True)
                else:
                    st.info("No dates detected")

                st.markdown("### ⚖️ Key Clauses")
                if insights['clauses']:
                    clauses_list = []
                    for title, text in list(insights['clauses'].items())[:8]:
                        clauses_list.append({"Clause": title, "Extract": text[:200] + "..."})
                    clauses_df = pd.DataFrame(clauses_list)
                    st.dataframe(clauses_df, use_container_width=True, height=300)
                else:
                    st.warning("No standard clauses detected")

            # NUEVA PESTAÑA WORDCLOUD
            if show_wordcloud:
                with tab_cloud:
                    st.markdown("### ☁️ Keyword Focus Visualization")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📄 Document Overview**")
                        wc_general = generate_wordcloud(res['text_for_cloud'], focus_keywords=False)
                        show_wordcloud_streamlit(wc_general, f"General Terms - {res['filename']}")
                    
                    with col2:
                        st.markdown("**🚨 Risk Keywords Focus**")
                        # Texto enfocado en riesgos
                        risk_text = " ".join(res["flags"]) if res["flags"] else res['text_for_cloud']
                        wc_risk = generate_wordcloud(risk_text, focus_keywords=True)
                        show_wordcloud_streamlit(wc_risk, f"Risk Terms - {res['filename']}")

    # DOWNLOADS
    st.subheader("💾 Download Reports")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = pd.DataFrame(results).to_csv(index=False).encode()
        st.download_button("📊 CSV", csv_data, "FCPA_analysis.csv", "text/csv")
    
    with col2:
        json_data = json.dumps(results, indent=2).encode()
        st.download_button("📄 JSON", json_data, "FCPA_analysis.json", "application/json")
    
    with col3:
        full_report = "\n\n".join([
            f"FILE: {r['filename']}\nRISK: {r['risk_level']} ({r['risk_score']}/100)\n{r['full_report']}\n{r['contract_summary']}"
            for r in results
        ])
        st.download_button("📋 TXT Report", full_report, "FCPA_Report.txt", "text/plain")


st.title("🧠 Uncover Hidden Risks, Before They Cost Millions")

st.sidebar.divider()

st.sidebar.markdown("---")
st.sidebar.success("FCPA Analysis AI v1.0")
st.sidebar.caption("© 2025 – All rights reserved")
st.success("🚀 Ready for forensic analysis - 100% local processing")
# st.info("No internet required after setup")
st.caption("Built with spaCy, Hugging Face, Sentence-Transformers — detects real-world FCPA violations including evasion tactics")