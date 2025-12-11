import json
import io
import os
import streamlit as st
import torch
import numpy as np

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

# Neutral Apple-ish theme: white / light grey only, full-width cards
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

    .main {
        background-color: var(--bg-main);
    }

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

    /* Try to harmonise file uploader button */
    .stFileUploader > label div {
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. LOAD CONFIG & MODELS
# ============================================================

@st.cache_resource
def load_matchai_config(path: str | None = None):
    """
    Load matchai_config.json from the same folder as streamlit_app.py
    so it works both locally and on Streamlit Cloud.
    """
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
        # Fallback config (2-class SST model – just for pipeline testing)
        return {
            "fine_tuned_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
            "summarization_model": "sshleifer/distilbart-cnn-12-6",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "weights": {
                "classifier": 0.5,
                "similarity": 0.3,
                "keywords": 0.2,
            },
            "label_id2name": {
                "0": "Not Fit",
                "1": "Good Fit",
            },
        }


@st.cache_resource
def load_models_and_pipelines(cfg: dict):
    """
    Load classifier, summariser, and embedding model.
    NER was removed to reduce memory usage.
    """
    hf_token = st.secrets.get("HF_TOKEN", None) or os.getenv("HF_TOKEN", None)

    clf_id = cfg["fine_tuned_model_id"]
    clf_tokenizer = AutoTokenizer.from_pretrained(clf_id, token=hf_token)
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        clf_id, token=hf_token
    )
    clf_model.to(device)
    clf_model.eval()

    summarizer = hf_pipeline(
        "summarization",
        model=cfg["summarization_model"],
        device=0 if torch.cuda.is_available() else -1,
        token=hf_token,
    )

    sim_model = SentenceTransformer(cfg["embedding_model"], token=hf_token)

    raw_map = cfg.get(
        "label_id2name",
        {"0": "No Fit", "1": "Potential Fit", "2": "Good Fit"},
    )
    label_id2name = {int(k): v for k, v in raw_map.items()}

    return clf_tokenizer, clf_model, summarizer, sim_model, label_id2name


config = load_matchai_config()
clf_tokenizer, clf_model, summarizer, sim_model, label_id2name = (
    load_models_and_pipelines(config)
)

DEFAULT_WEIGHTS = config.get(
    "weights", {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2}
)

# ============================================================
# 2. HELPER FUNCTIONS (TEXT, MODELS, SCORING)
# ============================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with io.BytesIO(file_bytes) as f:
            text = docx2txt.process(f)
        return (text or "").strip()
    except Exception as e:
        st.warning(f"Could not read Word file: {e}")
        return ""


def summarize_text(text: str, max_len: int = 150) -> str:
    if not text or not isinstance(text, str):
        return ""
    truncated = text[:2000]
    try:
        out = summarizer(
            truncated,
            max_length=max_len,
            min_length=40,
            do_sample=False,
        )[0]["summary_text"]
        return out
    except Exception as e:
        st.warning(f"Summarisation issue: {e}")
        return truncated[:300]


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


def process_job_description(jd_text: str):
    summary = summarize_text(jd_text)
    keywords = list({w.lower() for w in summary.split() if len(w) > 4})
    return {"raw": jd_text, "summary": summary, "keywords": keywords}


def process_resume(res_text: str):
    summary = summarize_text(res_text)
    # NER removed – keep the structure minimal
    return {"raw": res_text, "summary": summary}


def keyword_match_score(jd_keywords, resume_summary: str) -> float:
    if not jd_keywords:
        return 0.0
    resume_words = set(w.lower() for w in resume_summary.split())
    hits = sum(1 for kw in jd_keywords if kw in resume_words)
    return hits / len(jd_keywords)


def generate_candidate_highlights(result_dict: dict) -> str:
    """
    Generate a natural, useful 2–3 sentence highlight
    based on label, score, similarity, keywords and resume summary.
    """
    label = result_dict["fit"]["label_name"]
    score_pct = result_dict["final_score"] * 100
    sim = result_dict["similarity"]
    kw = result_dict["keyword_score"]
    jd_summary = result_dict["jd"]["summary"]
    res_summary = result_dict["resume"]["summary"]
    res_lower = res_summary.lower()

    parts = []

    # 1) Overall match level
    if "Good" in label and score_pct >= 80:
        parts.append(
            f"Overall, this profile looks like a strong match for the role "
            f"(score {score_pct:.1f}%, label {label})."
        )
    elif "Good" in label or "Potential" in label:
        parts.append(
            f"This profile shows a reasonable match to the role "
            f"(score {score_pct:.1f}%, label {label})."
        )
    else:
        parts.append(
            f"At first glance this profile appears to be a weaker match "
            f"(score {score_pct:.1f}%, label {label}), but still worth a quick review."
        )

    # 2) Similarity & keywords
    if sim >= 0.8 and kw >= 0.6:
        parts.append(
            "The CV aligns closely with the job description, both in overall content "
            "and in coverage of key requirements."
        )
    elif sim >= 0.7:
        parts.append(
            "There is solid thematic overlap between the CV and the role, "
            "with several relevant responsibilities reflected."
        )
    elif sim >= 0.55:
        parts.append(
            "The CV captures some important aspects of the role but may miss a few "
            "of the core requirements."
        )
    else:
        parts.append(
            "Alignment with the job description appears limited, suggesting that the role "
            "may only be a partial fit."
        )

    # 3) Simple heuristic on role / seniority / domain from resume summary
    signals = []
    if any(x in res_lower for x in ["manager", "lead", "head", "director"]):
        signals.append("experience in a leadership or manager-level capacity")
    elif "senior" in res_lower:
        signals.append("senior-level responsibilities")

    if any(x in res_lower for x in ["data", "analytics", "python", "sql", "machine learning"]):
        signals.append("a strong data/analytics or technical component")
    if any(x in res_lower for x in ["finance", "bank", "investment", "audit"]):
        signals.append("exposure to finance or banking")
    if any(x in res_lower for x in ["marketing", "brand", "campaign"]):
        signals.append("experience in marketing or brand-related work")
    if any(x in res_lower for x in ["hr", "recruitment", "talent acquisition"]):
        signals.append("background in HR or recruitment")

    if signals:
        # Turn signal list into human sentence
        if len(signals) == 1:
            parts.append(f"The CV indicates {signals[0]}.")
        else:
            last = signals[-1]
            initial = ", ".join(signals[:-1])
            parts.append(f"The CV indicates {initial}, and {last}.")

    # 4) Fallback if summary is very short
    if not res_summary.strip():
        parts.append(
            "The resume text is limited, so this assessment should be treated as indicative only."
        )

    return " ".join(parts)


def evaluate_candidate(jd_text: str, res_text: str, weights: dict):
    jd = process_job_description(jd_text)
    res = process_resume(res_text)

    sim_raw = compute_similarity(jd["summary"], res["summary"])
    sim_norm = (sim_raw + 1) / 2 if sim_raw < 1 else min(sim_raw, 1.0)

    kw_score = keyword_match_score(jd["keywords"], res["summary"])

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

    result = {
        "jd": jd,
        "resume": res,
        "similarity_raw": sim_raw,
        "similarity": sim_norm,
        "keyword_score": kw_score,
        "fit": fit,
        "prob_good_fit": float(prob_good_fit),
        "final_score": float(final_score),
    }
    result["highlight"] = generate_candidate_highlights(result)
    return result


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
    return results_sorted


def map_priority(score: float) -> str:
    if score >= 0.8:
        return "Interview priority: HIGH"
    elif score >= 0.6:
        return "Interview priority: MEDIUM"
    else:
        return "Interview priority: LOW"


def clear_all_inputs():
    for key in ["jd_text", "resume_text", "uploaded_files"]:
        if key in st.session_state:
            del st.session_state[key]

# ============================================================
# 3. HEADER
# ============================================================

st.title("🔍 MatchAI: Candidate Suitability Screening")
st.markdown(
    "<span class='muted'>Screen and prioritise candidates against a specific job description.</span>",
    unsafe_allow_html=True,
)

# ============================================================
# 4. INPUTS CARD (FULL-WIDTH ROW)
# ============================================================

with st.container():
    st.markdown("<div class='round-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-label'>INPUTS</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Job description</div>", unsafe_allow_html=True)
    st.markdown("<div class='io-box pill-input'>", unsafe_allow_html=True)
    jd_text = st.text_area(
        "",
        key="jd_text",
        height=230,
        placeholder="Paste the job description here...",
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
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='pill-input'>", unsafe_allow_html=True)
        resume_text = st.text_area(
            "",
            key="resume_text",
            height=220,
            placeholder="Paste resume text here...",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                extracted = extract_text_from_pdf(uploaded_file.read())
            elif uploaded_file.type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ]:
                extracted = extract_text_from_docx(uploaded_file.read())
            else:
                extracted = uploaded_file.read().decode("utf-8", errors="ignore")

            if extracted and not resume_text.strip():
                st.session_state["resume_text"] = extracted
                resume_text = extracted
                st.info(f"Text extracted from: {uploaded_file.name[:40]}")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        MAX_RESUMES = 30
        st.markdown("<div class='pill-upload'>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload multiple resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="uploaded_files",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_files and len(uploaded_files) > MAX_RESUMES:
            st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Clear button – aligned to left within the card
    clear_col, _ = st.columns([0.2, 0.8])
    with clear_col:
        if st.button("Clear inputs"):
            clear_all_inputs()
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 5. EVALUATION & RESULTS CARD (FULL-WIDTH ROW)
# ============================================================

with st.container():
    st.markdown("<div class='round-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-label'>EVALUATION</div>", unsafe_allow_html=True)

    top_row = st.columns([0.5, 0.5])
    with top_row[0]:
        st.markdown("<div class='section-title'>MatchAI score</div>", unsafe_allow_html=True)
    with top_row[1]:
        with st.expander("ℹ️  How is the score calculated?", expanded=False):
            st.write(
                "The final suitability score combines three signals:\n"
                "- **P(Good Fit)** from the classifier\n"
                "- **Semantic similarity** between JD and resume summaries\n"
                "- **Keyword coverage** of JD terms in the resume\n\n"
                "Default formula:\n"
                "**Final Score = 0.5 × P(Good Fit) + 0.3 × Similarity + 0.2 × Keyword Coverage**.\n\n"
                "Weights can be adjusted below to reflect different HR priorities."
            )

    # Weights expander – close to score explanation
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
            w_clf = st.slider(
                "Weight: P(Good Fit)",
                0.0,
                1.0,
                float(st.session_state.w_clf),
                0.05,
            )
        with c2:
            w_sim = st.slider(
                "Weight: Similarity",
                0.0,
                1.0,
                float(st.session_state.w_sim),
                0.05,
            )
        with c3:
            w_kw = st.slider(
                "Weight: Keywords",
                0.0,
                1.0,
                float(st.session_state.w_kw),
                0.05,
            )

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
            st.experimental_rerun()

    # Safety: if expander not opened yet, ensure weights exist
    if "weights" not in locals():
        weights = DEFAULT_WEIGHTS

    st.markdown("---")

    # Score buttons – near top of evaluation card
    button_row = st.columns([0.35, 0.35, 0.3])
    with button_row[0]:
        single_btn = st.button("🔍 Score single candidate")
    with button_row[1]:
        batch_btn = st.button("🔍 Score candidates (batch)")

    # ---------------- SINGLE MODE RESULT ----------------
    if single_btn:
        jd_val = st.session_state.get("jd_text", "")
        resume_val = st.session_state.get("resume_text", "")

        if not jd_val.strip():
            st.error("Please provide a job description in the Inputs section.")
        elif not resume_val.strip():
            st.error("Please provide resume text or upload a file in the Inputs section.")
        else:
            with st.spinner("Evaluating candidate..."):
                result = evaluate_candidate(jd_val, resume_val, weights)

            score = result["final_score"]
            score_pct = score * 100
            label = result["fit"]["label_name"]
            prob_good = result["prob_good_fit"]

            st.markdown("<div class='io-box'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Result – Single candidate</div>", unsafe_allow_html=True)
            st.markdown(
                f"<p class='soft-label'>Final suitability score</p>"
                f"<p class='score-number'>{score_pct:.1f}%</p>",
                unsafe_allow_html=True,
            )
            st.write(f"**Predicted fit label:** {label}")
            st.write(f"**P(Good Fit):** {prob_good:.2f}")
            st.write(f"**{map_priority(score)}**")
            st.write(f"**Highlight:** {result['highlight']}")
            st.markdown("</div>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.write("**Semantic similarity**")
                st.write(f"{result['similarity']:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with m2:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.write("**Keyword coverage**")
                st.write(f"{result['keyword_score']:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with m3:
                st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
                st.write("**Model fit confidence**")
                st.write(f"{prob_good:.3f}")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Model-driven details"):
                st.write("**JD summary**")
                st.write(result["jd"]["summary"])
                st.write("**Resume summary**")
                st.write(result["resume"]["summary"])

    # ---------------- BATCH MODE RESULT ----------------
    if batch_btn:
        jd_val = st.session_state.get("jd_text", "")
        uploaded_files_val = st.session_state.get("uploaded_files", None)

        if not jd_val.strip():
            st.error("Please provide a job description in the Inputs section.")
        elif not uploaded_files_val:
            st.error("Please upload at least one resume file in the Inputs section.")
        else:
            resumes_list = []
            for f in uploaded_files_val:
                if f.type == "application/pdf":
                    text = extract_text_from_pdf(f.read())
                elif f.type in [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/msword",
                ]:
                    text = extract_text_from_docx(f.read())
                else:
                    text = f.read().decode("utf-8", errors="ignore")

                if text.strip():
                    resumes_list.append({"name": f.name, "text": text})

            if not resumes_list:
                st.error("No readable text found in uploaded resumes.")
            else:
                with st.spinner("Evaluating all candidates..."):
                    batch_results = evaluate_batch(jd_val, resumes_list, weights)

                st.markdown("<div class='io-box'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Result – Batch candidate ranking</div>", unsafe_allow_html=True)

                rows = []
                for rank, r in enumerate(batch_results, start=1):
                    rows.append(
                        {
                            "Rank": rank,
                            "File": r.get("file_name", f"Candidate {rank}"),
                            "Fit label": r["fit"]["label_name"],
                            "Final score (%)": round(r["final_score"] * 100, 1),
                            "P(Good Fit)": round(r["prob_good_fit"], 3),
                            "Similarity": round(r["similarity"], 3),
                            "Keyword score": round(r["keyword_score"], 3),
                        }
                    )

                st.dataframe(rows, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                top = batch_results[0]
                st.markdown("<div class='section-title'>Top candidate highlight</div>", unsafe_allow_html=True)
                st.write(f"**File:** {top.get('file_name', 'N/A')}")
                st.write(f"**Final score:** {top['final_score']*100:.1f}%")
                st.write(f"**Fit label:** {top['fit']['label_name']}")
                st.write(f"**Highlight:** {top['highlight']}")

                with st.expander("View all candidate highlights"):
                    for r in batch_results:
                        st.markdown(f"**{r.get('file_name', 'Candidate')}**")
                        st.write(
                            f"- Final score: {r['final_score']*100:.1f}% "
                            f"(label: {r['fit']['label_name']})"
                        )
                        st.write(f"- Highlight: {r['highlight']}")
                        st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# END
# ============================================================
