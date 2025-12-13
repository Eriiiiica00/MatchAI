import json
import io
import os
import re
import string
import streamlit as st
import torch
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)
from sentence_transformers import SentenceTransformer
from torch.nn.functional import softmax, cosine_similarity

import PyPDF2
import docx2txt

# ============================================================
# 0. BASIC SETUP & GLOBAL STYLE
# ============================================================

st.set_page_config(
    page_title="MatchAI: Candidate Suitability Screening",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg-main: #f5f5f7;
        --card-bg: #ffffff;
        --border-subtle: #e5e5ea;
        --accent-soft: #f0f0f5;
        --accent-strong: #d1d1d6;
        --text-main: #1c1c1e;
        --text-muted: #6e6e73;
    }

    html, body, [class^="css"]  {
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Text",
                     "Helvetica Neue", Arial, sans-serif;
    }

    .main { background-color: var(--bg-main); }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1100px;
    }

    .round-card {
        padding: 1.25rem 1.5rem;
        border-radius: 20px;
        background-color: var(--card-bg);
        border: 1px solid var(--border-subtle);
        box-shadow: 0 16px 35px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    .pill-input textarea,
    .pill-input .stTextArea textarea,
    .pill-input .stTextInput input {
        border-radius: 16px !important;
        border: 1px solid var(--border-subtle) !important;
        background-color: #ffffff !important;
        min-height: 220px !important;
        padding: 0.9rem 1rem !important;
        box-shadow: none !important;
    }

    .pill-upload .stFileUploader {
        border-radius: 16px !important;
        border: 1px dashed var(--border-subtle) !important;
        background-color: #fafafa !important;
        padding: 0.9rem 1rem !important;
    }

    .section-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.4rem;
        color: var(--text-main);
    }

    .soft-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .muted {
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    .score-number {
        font-size: 2.1rem;
        font-weight: 600;
        color: #111111;
        margin: 0;
    }

    .metric-box {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        background: #fafafa;
        border: 1px solid var(--accent-soft);
        font-size: 0.9rem;
    }

    .io-box {
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid var(--border-subtle);
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 999px !important;
        padding: 0.45rem 1.4rem !important;
        border: 1px solid var(--accent-strong) !important;
        background: linear-gradient(135deg, #ffffff, #f2f2f7) !important;
        color: #111111 !important;
        font-weight: 500 !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.06) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffffff, #e5e5ea) !important;
        border-color: #c7c7cc !important;
    }

    /* Small circular icon button */
    .icon-btn .stButton > button {
        width: 38px !important;
        height: 38px !important;
        padding: 0 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# CPU-friendly defaults
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(1)
except Exception:
    pass

# ============================================================
# 0.1 SESSION STATE DEFAULTS
# ============================================================

if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0

if "is_evaluating" not in st.session_state:
    st.session_state["is_evaluating"] = False

if "pending_action" not in st.session_state:
    st.session_state["pending_action"] = None  # "single" | "batch"

if "upload_note" not in st.session_state:
    st.session_state["upload_note"] = ""

if "active_candidate_idx" not in st.session_state:
    st.session_state["active_candidate_idx"] = 0

# ============================================================
# 1. LOAD CONFIG & MODELS (Classifier + Summarizer + Embeddings)
# ============================================================

@st.cache_resource
def load_matchai_config(path: str | None = None):
    if path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "matchai_config.json")

    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        return cfg
    except Exception as e:
        st.error(
            "Could not load matchai_config.json.\n"
            f"Using fallback config for demo. Error: {e}"
        )
        return {
            "fine_tuned_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
            "summarization_model": "sshleifer/distilbart-cnn-12-6",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "weights": {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2},
            "label_id2name": {"0": "Not Fit", "1": "Good Fit"},
        }

@st.cache_resource
def load_models_and_pipelines(cfg: dict):
    hf_token = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN", None)

    clf_id = cfg["fine_tuned_model_id"]
    try:
        clf_tokenizer = AutoTokenizer.from_pretrained(clf_id, token=hf_token)
        clf_model = AutoModelForSequenceClassification.from_pretrained(clf_id, token=hf_token)
        clf_model.to(device)
        clf_model.eval()

        summarizer = hf_pipeline(
            "summarization",
            model=cfg["summarization_model"],
            device=0 if torch.cuda.is_available() else -1,
            token=hf_token,
        )

        sim_model = SentenceTransformer(cfg["embedding_model"], token=hf_token)

    except Exception as e:
        st.error("❌ MatchAI couldn’t download one of the Hugging Face models.")
        st.write(
            "This usually means Streamlit Cloud could not reach Hugging Face, "
            "your model ID is invalid, or you hit a rate limit.\n\n"
            "Try:\n"
            "- Add `HF_TOKEN` in Streamlit Secrets\n"
            "- Double-check `matchai_config.json`\n"
            "- Reboot the app / try again later"
        )
        st.code(str(e))
        st.stop()

    raw_map = cfg.get("label_id2name", {"0": "No Fit", "1": "Potential Fit", "2": "Good Fit"})
    label_id2name = {int(k): v for k, v in raw_map.items()}

    return clf_tokenizer, clf_model, summarizer, sim_model, label_id2name


config = load_matchai_config()
clf_tokenizer, clf_model, summarizer, sim_model, label_id2name = load_models_and_pipelines(config)
DEFAULT_WEIGHTS = config.get("weights", {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2})

# ============================================================
# 2. HELPERS (TEXT, KEYWORDS, EVIDENCE, QUESTIONS, SCORING)
# ============================================================

STOPWORDS = {
    "the","and","for","with","from","that","this","you","your","are","our","will","have",
    "role","work","team","teams","using","use","used","able","must","should","within",
    "include","including","responsibilities","requirements","years","year","experience",
    "skills","skill","ability","strong","good","excellent","preferred","plus"
}

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _strip_bullet_prefix(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^[\u2022•\-\–\—\*\·\+]+\s*", "", s)  # bullets
    s = re.sub(r"^\(\s*\)\s*", "", s)
    return s.strip()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return _clean_spaces(text)
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with io.BytesIO(file_bytes) as f:
            text = docx2txt.process(f)
        return _clean_spaces(text or "")
    except Exception as e:
        st.warning(f"Could not read Word file: {e}")
        return ""

def summarize_text(text: str, max_len: int = 140) -> str:
    if not text or not isinstance(text, str):
        return ""
    truncated = text[:2200]
    try:
        out = summarizer(
            truncated,
            max_length=max_len,
            min_length=40,
            do_sample=False,
        )[0]["summary_text"]
        return _clean_spaces(out)
    except Exception:
        return _clean_spaces(truncated[:350])

def compute_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    emb1 = sim_model.encode(text1, convert_to_tensor=True)
    emb2 = sim_model.encode(text2, convert_to_tensor=True)
    sim = cosine_similarity(emb1, emb2, dim=0).item()
    return float(sim)

def predict_fit_label(jd_text: str, res_text: str):
    combined = res_text + " [SEP] " + jd_text
    inputs = clf_tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = clf_model(**inputs)
        probs = softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    return {
        "label_id": pred_id,
        "label_name": label_id2name.get(pred_id, f"Class {pred_id}"),
        "probs": probs.tolist(),
    }

def _tokens(text: str) -> list[str]:
    t = (text or "").lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    parts = [p.strip() for p in t.split() if p.strip()]
    keep = [p for p in parts if len(p) >= 4 and p not in STOPWORDS and not p.isdigit()]
    return keep

def _top_keywords_from_text(text: str, k: int = 22) -> list[str]:
    toks = _tokens(text)
    freq: dict[str, int] = {}
    for w in toks:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:k]]

def keyword_match_score(jd_keywords: list[str], resume_text_or_summary: str) -> float:
    if not jd_keywords:
        return 0.0
    resume_words = set(_tokens(resume_text_or_summary))
    hits = sum(1 for kw in jd_keywords if kw in resume_words)
    return hits / len(jd_keywords)

def process_job_description(jd_text: str):
    summary = summarize_text(jd_text, max_len=140)
    kw = _top_keywords_from_text(jd_text + " " + summary, k=22)
    return {"raw": jd_text, "summary": summary, "keywords": kw}

def process_resume(res_text: str):
    summary = summarize_text(res_text, max_len=140)
    return {"raw": res_text, "summary": summary}

# ---------- Contact/header filtering (exclude from evidence) ----------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+\d{1,3}\s*)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}")
LINKEDIN_RE = re.compile(r"(https?://(www\.)?linkedin\.com/[^\s]+)", re.I)

def _is_header_or_contact_line(line: str) -> bool:
    l = (line or "").strip()
    if not l:
        return True
    if EMAIL_RE.search(l) or LINKEDIN_RE.search(l):
        return True
    # Phone often noisy — only treat as contact if also contains '+' or 'tel' or 'phone'
    if ("+" in l or "tel" in l.lower() or "phone" in l.lower()) and PHONE_RE.search(l):
        return True
    # very short all-caps headings
    letters = re.sub(r"[^A-Za-z]", "", l)
    if len(letters) >= 8:
        cap_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
        if cap_ratio > 0.85 and len(l.split()) <= 10:
            return True
    # obvious section headers
    if l.lower() in {"education", "experience", "skills", "summary", "projects", "certifications"}:
        return True
    return False

def _resume_lines(resume_raw: str) -> list[str]:
    # Prefer line splits for resumes
    lines = [x.strip() for x in re.split(r"[\n\r]+", resume_raw or "") if x.strip()]
    # Fallback: split by bullets if text was flattened
    if len(lines) < 8:
        lines = [x.strip() for x in re.split(r"[•\u2022]", resume_raw or "") if x.strip()]
    # clean
    cleaned = []
    for ln in lines:
        ln = _clean_spaces(ln)
        ln = _strip_bullet_prefix(ln)
        if ln:
            cleaned.append(ln)
    return cleaned

def extract_evidence_bullets(jd_keywords: list[str], resume_raw: str, max_bullets: int = 5):
    """
    Returns (evidence_lines, matched_keywords_set)
    Evidence is ONLY from resume body (header/contact excluded).
    """
    if not resume_raw or not jd_keywords:
        return [], set()

    jd_set = set(jd_keywords)
    lines = _resume_lines(resume_raw)

    scored = []
    matched_all = set()

    for ln in lines:
        if _is_header_or_contact_line(ln):
            continue
        if len(ln) < 35:
            continue

        toks = set(_tokens(ln))
        overlap = [kw for kw in jd_set if kw in toks]
        score = len(overlap)

        if score > 0:
            matched_all.update(overlap)
            # prefer more specific / quantified lines
            quant_bonus = 1 if re.search(r"\b(\d+(\.\d+)?\s*(m|bn|mm|%|mw|kw|usd|hkd|krw|ntd))\b", ln.lower()) else 0
            scored.append((score + quant_bonus, score, ln))

    scored.sort(key=lambda x: (-x[0], -x[1], len(x[2])))
    evidence = []
    seen = set()
    for _, _, ln in scored:
        cut = ln[:220] + ("…" if len(ln) > 220 else "")
        key = cut.lower()
        if key in seen:
            continue
        seen.add(key)
        evidence.append(cut)
        if len(evidence) >= max_bullets:
            break

    return evidence, matched_all

def build_interview_kit(jd_keywords: list[str], evidence_lines: list[str], max_q: int = 5):
    """
    Only generate questions if there is concrete evidence to probe.
    Questions are anchored to the evidence lines (not generic).
    """
    if not evidence_lines:
        return []

    qs = []
    jd_set = set(jd_keywords)

    # For each evidence line, make a probing question
    for ln in evidence_lines:
        toks = set(_tokens(ln))
        overlap = [kw for kw in jd_set if kw in toks]
        anchor = overlap[0] if overlap else "this"
        qs.append(
            f'You mentioned “{ln}”. Can you walk me through your specific role, scope, and measurable outcomes (tools, stakeholders, impact), especially around **{anchor}**?'
        )
        if len(qs) >= max_q:
            break

    # Add 1 synthesis question if room
    if len(qs) < max_q:
        # pick 2-3 anchors from all overlaps
        anchors = []
        for ln in evidence_lines:
            toks = set(_tokens(ln))
            anchors.extend([k for k in jd_keywords if k in toks])
        anchors = list(dict.fromkeys(anchors))[:3]
        if anchors:
            qs.append(
                f"If you joined tomorrow, what would your first 30 days plan look like to deliver on **{', '.join(anchors)}**?"
            )

    return qs[:max_q]

def map_priority(score: float) -> str:
    if score >= 0.8:
        return "Interview priority: HIGH"
    elif score >= 0.6:
        return "Interview priority: MEDIUM"
    else:
        return "Interview priority: LOW"

def evaluate_candidate(jd_text: str, res_text: str, weights: dict):
    jd = process_job_description(jd_text)
    res = process_resume(res_text)

    sim_raw = compute_similarity(jd["summary"], res["summary"])
    sim_norm = (sim_raw + 1) / 2 if sim_raw < 1 else min(sim_raw, 1.0)

    kw_score = keyword_match_score(jd["keywords"], res["raw"] + " " + res["summary"])

    fit = predict_fit_label(jd_text, res_text)
    if len(fit["probs"]) >= 3:
        prob_good_fit = fit["probs"][2]
    else:
        prob_good_fit = fit["probs"][fit["label_id"]]

    final_score = (
        weights["classifier"] * prob_good_fit
        + weights["similarity"] * sim_norm
        + weights["keywords"] * kw_score
    )

    # Evidence + interview kit
    evidence, matched_kw = extract_evidence_bullets(jd["keywords"], res["raw"], max_bullets=5)
    questions = build_interview_kit(jd["keywords"], evidence, max_q=5)

    return {
        "jd": jd,
        "resume": res,
        "similarity_raw": float(sim_raw),
        "similarity": float(sim_norm),
        "keyword_score": float(kw_score),
        "fit": fit,
        "prob_good_fit": float(prob_good_fit),
        "final_score": float(final_score),
        "evidence": evidence,
        "questions": questions,
        "matched_keywords": sorted(list(matched_kw))[:12],
    }

def evaluate_batch(jd_text: str, resumes_list: list, weights: dict):
    results = []
    for item in resumes_list:
        name = item.get("name", "Unknown")
        text = item.get("text", "")
        if not text:
            continue
        r = evaluate_candidate(jd_text, text, weights)
        r["file_name"] = name
        results.append(r)

    results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # differentiator sentence per candidate
    # compute group averages
    if results_sorted:
        avg_sim = float(np.mean([x["similarity"] for x in results_sorted]))
        avg_kw = float(np.mean([x["keyword_score"] for x in results_sorted]))
        avg_prob = float(np.mean([x["prob_good_fit"] for x in results_sorted]))
    else:
        avg_sim = avg_kw = avg_prob = 0.0

    for idx, r in enumerate(results_sorted, start=1):
        # dominant driver (relative to avg)
        sim_delta = r["similarity"] - avg_sim
        kw_delta = r["keyword_score"] - avg_kw
        prob_delta = r["prob_good_fit"] - avg_prob

        # choose the strongest positive delta; if none, choose the least negative
        deltas = {
            "semantic similarity": sim_delta,
            "keyword coverage": kw_delta,
            "model confidence": prob_delta,
        }
        driver = max(deltas.items(), key=lambda kv: kv[1])[0]

        # craft sentence
        if driver == "semantic similarity":
            sentence = f"Stands out with {driver} versus other candidates."
        elif driver == "keyword coverage":
            sentence = f"Stands out with stronger {driver} of JD terms."
        else:
            sentence = f"Stands out with higher {driver} from the classifier."

        r["differentiator"] = sentence
        r["rank"] = idx

    return results_sorted

# ---------- Upload reading ----------

def _read_uploaded_file(uploaded) -> str:
    if uploaded is None:
        return ""
    data = uploaded.read()
    if uploaded.type == "application/pdf":
        return extract_text_from_pdf(data)
    if uploaded.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]:
        return extract_text_from_docx(data)
    try:
        return _clean_spaces(data.decode("utf-8", errors="ignore"))
    except Exception:
        return ""

# ============================================================
# NEW EVALUATION (safe reset)
# ============================================================

def new_evaluation():
    st.session_state["jd_text"] = ""
    st.session_state["resume_text"] = ""
    st.session_state["upload_note"] = ""
    st.session_state["active_candidate_idx"] = 0

    st.session_state.pop("last_single_result", None)
    st.session_state.pop("last_batch_results", None)

    st.session_state["is_evaluating"] = False
    st.session_state["pending_action"] = None

    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1
    st.rerun()

# ============================================================
# 3. HEADER (New evaluation icon on right)
# ============================================================

h1, h2 = st.columns([0.9, 0.1])
with h1:
    st.title("🔍 MatchAI: Candidate Suitability Screening")
    st.markdown(
        "<span class='muted'>Screen and prioritise candidates against a specific job description.</span>",
        unsafe_allow_html=True,
    )
with h2:
    st.markdown("<div class='icon-btn'>", unsafe_allow_html=True)
    st.button("?", help="Start a new evaluation (clears inputs and uploads).", on_click=new_evaluation)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 4. INPUTS CARD
# ============================================================

nonce = st.session_state["uploader_nonce"]
single_upload_key = f"single_upload_{nonce}"
batch_upload_key = f"batch_upload_{nonce}"

with st.container():
    st.markdown("<div class='round-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-label'>INPUTS</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Job description</div>", unsafe_allow_html=True)
    st.markdown("<div class='io-box pill-input'>", unsafe_allow_html=True)
    st.text_area(
        "",
        key="jd_text",
        height=230,
        placeholder="Paste the job description here...",
        disabled=st.session_state["is_evaluating"],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Candidate resumes</div>", unsafe_allow_html=True)
    c_mode, c_hint = st.columns([0.45, 0.55])
    with c_mode:
        mode = st.radio(
            "Mode",
            ["Single candidate", "Batch (multiple resumes)"],
            horizontal=True,
            label_visibility="collapsed",
            key="mode",
            disabled=st.session_state["is_evaluating"],
        )
    with c_hint:
        if mode == "Single candidate":
            st.markdown(
                "<div class='muted' style='margin-top:0.3rem;'>Upload a file or paste the resume text.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='muted' style='margin-top:0.3rem;'>Upload multiple resumes (max 30 per batch). PDFs, Word, or text.</div>",
                unsafe_allow_html=True,
            )

    if mode == "Single candidate":
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='pill-upload'>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload resume (PDF / Word / TXT, optional)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=single_upload_key,
            disabled=st.session_state["is_evaluating"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Populate resume_text BEFORE rendering the widget (prevents StreamlitAPIException)
        if uploaded_file is not None:
            extracted = _read_uploaded_file(uploaded_file)
            if extracted:
                st.session_state["upload_note"] = f"Text extracted from: {uploaded_file.name[:60]}"
                if not st.session_state.get("resume_text", "").strip():
                    st.session_state["resume_text"] = extracted

        st.markdown("<div class='pill-input'>", unsafe_allow_html=True)
        st.text_area(
            "",
            key="resume_text",
            height=220,
            placeholder="Paste resume text here...",
            disabled=st.session_state["is_evaluating"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

        note = st.session_state.get("upload_note", "")
        if note:
            st.info(note)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        MAX_RESUMES = 30
        st.markdown("<div class='pill-upload'>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload multiple resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=batch_upload_key,
            disabled=st.session_state["is_evaluating"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_files:
            st.caption(f"Maximum {MAX_RESUMES} resumes per batch.")
            if len(uploaded_files) > MAX_RESUMES:
                st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 5. EVALUATION & RESULTS CARD
# ============================================================

with st.container():
    st.markdown("<div class='round-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-label'>EVALUATION</div>", unsafe_allow_html=True)

    top_row = st.columns([0.5, 0.5])
    with top_row[0]:
        st.markdown("<div class='section-title'>MatchAI score</div>", unsafe_allow_html=True)
    with top_row[1]:
        with st.expander("❔  How is the score calculated?", expanded=False):
            st.write(
                "The final suitability score combines three signals:\n"
                "- **P(Good Fit)** from the classifier\n"
                "- **Semantic similarity** between JD and resume summaries\n"
                "- **Keyword coverage** of JD terms in the resume\n\n"
                "Default formula:\n"
                "**Final Score = 0.5 × P(Good Fit) + 0.3 × Similarity + 0.2 × Keyword Coverage**.\n\n"
                "Weights can be adjusted below to reflect different HR priorities."
            )

    # Weights
    with st.expander("Adjust scoring weights (optional)"):
        st.caption("Weights are normalised automatically. Reset to return to defaults.")

        if "w_clf" not in st.session_state:
            st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
        if "w_sim" not in st.session_state:
            st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
        if "w_kw" not in st.session_state:
            st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]

        c1, c2, c3 = st.columns(3)
        with c1:
            w_clf = st.slider("Weight: P(Good Fit)", 0.0, 1.0, float(st.session_state.w_clf), 0.05, key="slider_w_clf")
        with c2:
            w_sim = st.slider("Weight: Similarity", 0.0, 1.0, float(st.session_state.w_sim), 0.05, key="slider_w_sim")
        with c3:
            w_kw = st.slider("Weight: Keywords", 0.0, 1.0, float(st.session_state.w_kw), 0.05, key="slider_w_kw")

        st.session_state.w_clf = w_clf
        st.session_state.w_sim = w_sim
        st.session_state.w_kw = w_kw

        total = w_clf + w_sim + w_kw
        if total == 0:
            weights = {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2}
        else:
            weights = {
                "classifier": w_clf / total,
                "similarity": w_sim / total,
                "keywords": w_kw / total,
            }

        st.caption(
            f"Normalised weights →  P(Good Fit): {weights['classifier']:.2f},  "
            f"Similarity: {weights['similarity']:.2f},  Keywords: {weights['keywords']:.2f}"
        )

        if st.button("Reset to default weights"):
            st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
            st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
            st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]
            st.rerun()

    # Ensure weights exists even if expander never opened
    if "weights" not in locals():
        tot = DEFAULT_WEIGHTS["classifier"] + DEFAULT_WEIGHTS["similarity"] + DEFAULT_WEIGHTS["keywords"]
        weights = {
            "classifier": DEFAULT_WEIGHTS["classifier"] / tot,
            "similarity": DEFAULT_WEIGHTS["similarity"] / tot,
            "keywords": DEFAULT_WEIGHTS["keywords"] / tot,
        }

    st.markdown("---")

    # Non-janky loading: disable buttons + change label
    score_single_label = "Evaluating…" if (st.session_state["is_evaluating"] and st.session_state["pending_action"] == "single") else "🔍 Score single candidate"
    score_batch_label = "Evaluating…" if (st.session_state["is_evaluating"] and st.session_state["pending_action"] == "batch") else "🔍 Score candidates (batch)"

    button_row = st.columns([0.35, 0.35, 0.3])
    with button_row[0]:
        clicked_single = st.button(score_single_label, disabled=st.session_state["is_evaluating"])
    with button_row[1]:
        clicked_batch = st.button(score_batch_label, disabled=st.session_state["is_evaluating"])

    # Queue actions (sets state first, then reruns cleanly)
    if clicked_single and not st.session_state["is_evaluating"]:
        st.session_state["is_evaluating"] = True
        st.session_state["pending_action"] = "single"
        st.rerun()

    if clicked_batch and not st.session_state["is_evaluating"]:
        st.session_state["is_evaluating"] = True
        st.session_state["pending_action"] = "batch"
        st.rerun()

    # Perform evaluation in a stable place (no layout collapse)
    if st.session_state["is_evaluating"] and st.session_state["pending_action"] == "single":
        jd_val = st.session_state.get("jd_text", "")
        resume_val = st.session_state.get("resume_text", "")

        if not jd_val.strip():
            st.session_state["is_evaluating"] = False
            st.session_state["pending_action"] = None
            st.error("Please provide a job description in the Inputs section.")
        elif not resume_val.strip():
            st.session_state["is_evaluating"] = False
            st.session_state["pending_action"] = None
            st.error("Please provide resume text or upload a file in the Inputs section.")
        else:
            with st.spinner("Analyzing candidate profile…"):
                result = evaluate_candidate(jd_val, resume_val, weights)
                st.session_state["last_single_result"] = result

            st.session_state["is_evaluating"] = False
            st.session_state["pending_action"] = None
            st.rerun()

    if st.session_state["is_evaluating"] and st.session_state["pending_action"] == "batch":
        jd_val = st.session_state.get("jd_text", "")
        batch_files = st.session_state.get(batch_upload_key, None)

        if not jd_val.strip():
            st.session_state["is_evaluating"] = False
            st.session_state["pending_action"] = None
            st.error("Please provide a job description in the Inputs section.")
        elif not batch_files:
            st.session_state["is_evaluating"] = False
            st.session_state["pending_action"] = None
            st.error("Please upload at least one resume file in the Inputs section.")
        else:
            MAX_RESUMES = 30
            if len(batch_files) > MAX_RESUMES:
                st.session_state["is_evaluating"] = False
                st.session_state["pending_action"] = None
                st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")
            else:
                resumes_list = []
                for f in batch_files:
                    text = _read_uploaded_file(f)
                    if text.strip():
                        resumes_list.append({"name": f.name, "text": text})

                if not resumes_list:
                    st.session_state["is_evaluating"] = False
                    st.session_state["pending_action"] = None
                    st.error("No readable text found in uploaded resumes.")
                else:
                    with st.spinner("Ranking candidates…"):
                        batch_results = evaluate_batch(jd_val, resumes_list, weights)
                        st.session_state["last_batch_results"] = batch_results
                        st.session_state["active_candidate_idx"] = 0

                    st.session_state["is_evaluating"] = False
                    st.session_state["pending_action"] = None
                    st.rerun()

    # ---------------- SINGLE RESULT RENDER ----------------
    if "last_single_result" in st.session_state:
        result = st.session_state["last_single_result"]
        score = result["final_score"]
        score_pct = score * 100
        label = result["fit"]["label_name"]
        prob_good = result["prob_good_fit"]
        priority = map_priority(score)

        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Result – Single candidate</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='soft-label'>Final suitability score</p>"
            f"<p class='score-number'>{score_pct:.1f}%</p>",
            unsafe_allow_html=True,
        )
        st.write(f"**{priority}**")
        st.write(f"**Fit label:** {label}")
        st.write(f"**P(Good Fit):** {prob_good:.2f}")

        # (3) Metrics moved up right after priority
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Semantic similarity**")
            st.write(f"{result['similarity']*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Keyword coverage**")
            st.write(f"{result['keyword_score']*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Model confidence**")
            st.write(f"{prob_good*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Evidence bullets + Interview kit logic
        st.markdown("<div class='section-title'>Evidence bullets</div>", unsafe_allow_html=True)
        if not result["evidence"]:
            st.write("No clear JD-aligned evidence was found in the resume body. Relevance should be validated via targeted interview discussion and follow-up questions.")
            st.markdown("<div class='section-title' style='margin-top:1rem;'>Interview kit</div>", unsafe_allow_html=True)
            st.write("Given the lack of concrete JD-aligned evidence in the resume body, it’s recommended to do a deeper interview dive to verify practical fit, transferable skills, and role-specific exposure.")
        else:
            for e in result["evidence"][:5]:
                st.write(f"- “{e}”")

            st.markdown("<div class='section-title' style='margin-top:1rem;'>Interview kit</div>", unsafe_allow_html=True)
            # (5) only generate questions when something concrete to probe
            if result["questions"]:
                for q in result["questions"][:5]:
                    st.write(f"- {q}")
            else:
                st.write("Evidence exists, but it is not specific enough to generate targeted questions. Use the interview to probe scope, ownership, and measurable impact for the evidence bullets above.")

    # ---------------- BATCH RESULT RENDER ----------------
    if "last_batch_results" in st.session_state:
        batch_results = st.session_state["last_batch_results"]

        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Result – Batch candidate ranking</div>", unsafe_allow_html=True)

        rows = []
        for r in batch_results:
            priority = map_priority(r["final_score"]).replace("Interview priority: ", "")
            rows.append(
                {
                    "Rank": r.get("rank"),
                    "File": r.get("file_name", "Candidate"),
                    # (7-2) priority first, then final score, fit label, similarity, keyword
                    "Priority": priority,
                    "Final score (%)": round(r["final_score"] * 100, 1),
                    "Fit label": r["fit"]["label_name"],
                    # (7-1) percentages
                    "Similarity (%)": round(r["similarity"] * 100, 1),
                    "Keyword (%)": round(r["keyword_score"] * 100, 1),
                    # (7-3) differentiator
                    "Why this candidate": r.get("differentiator", ""),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Candidate selector
        st.markdown("<div class='section-title'>Candidate details</div>", unsafe_allow_html=True)
        options = [f"{i+1}. {r.get('file_name','Candidate')}" for i, r in enumerate(batch_results)]
        current_i = int(st.session_state.get("active_candidate_idx", 0))
        selected_label = st.selectbox("Select a candidate", options=options, index=min(current_i, len(options) - 1))
        selected_i = int(str(selected_label).split(".")[0]) - 1
        st.session_state["active_candidate_idx"] = selected_i
        sel = batch_results[selected_i]

        # (8) Same layout as single candidate
        score = sel["final_score"]
        score_pct = score * 100
        label = sel["fit"]["label_name"]
        prob_good = sel["prob_good_fit"]
        priority = map_priority(score)

        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Result – Selected candidate</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='soft-label'>Final suitability score</p>"
            f"<p class='score-number'>{score_pct:.1f}%</p>",
            unsafe_allow_html=True,
        )
        st.write(f"**{priority}**")
        st.write(f"**Fit label:** {label}")
        st.write(f"**P(Good Fit):** {prob_good:.2f}")
        if sel.get("differentiator"):
            st.write(f"**Differentiator:** {sel['differentiator']}")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Semantic similarity**")
            st.write(f"{sel['similarity']*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Keyword coverage**")
            st.write(f"{sel['keyword_score']*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Model confidence**")
            st.write(f"{prob_good*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Evidence + Interview focus (short)
        st.markdown("<div class='section-title'>Evidence bullets (short)</div>", unsafe_allow_html=True)
        if not sel["evidence"]:
            st.write("No clear JD-aligned evidence was found in the resume body. Relevance should be validated via interview deep dive.")
            st.markdown("<div class='section-title' style='margin-top:1rem;'>Interview focus (short)</div>", unsafe_allow_html=True)
            st.write("Given limited concrete evidence, use the interview to validate transferable skills, relevant exposure, and role-specific hands-on depth.")
        else:
            for e in sel["evidence"][:3]:
                st.write(f"- “{e}”")

            st.markdown("<div class='section-title' style='margin-top:1rem;'>Interview focus (short)</div>", unsafe_allow_html=True)
            if sel["questions"]:
                for q in sel["questions"][:3]:
                    st.write(f"- {q}")
            else:
                st.write("Evidence exists, but it is not specific enough to generate targeted questions. Probe scope, ownership, and measurable impact for the evidence bullets above.")

    st.markdown("</div>", unsafe_allow_html=True)
