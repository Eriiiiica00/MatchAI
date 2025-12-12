import json
import io
import os
import re
import string
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Keep CPU-friendly defaults on Streamlit Cloud
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(1)
except Exception:
    pass

# ============================================================
# 1. LOAD CONFIG & MODELS (NO NER – to save memory)
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
            # lighter summarizer than 12-6
            "summarization_model": "sshleifer/distilbart-cnn-6-6",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "weights": {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2},
            "label_id2name": {"0": "Not Fit", "1": "Good Fit"},
        }

@st.cache_resource
def load_models_and_pipelines(cfg: dict):
    """
    HF_TOKEN support to avoid 429 on Streamlit Cloud.
    """
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
# 2. HELPER FUNCTIONS (TEXT, MODELS, SCORING)
# ============================================================

STOPWORDS = {
    "the","and","for","with","from","that","this","you","your","are","our","will","have",
    "role","work","team","teams","using","use","used","able","must","should","within",
    "include","including","responsibilities","requirements","years","year","experience",
    "skills","skill","ability","strong","good","excellent","preferred","plus"
}

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

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

def summarize_text(text: str, max_len: int = 150) -> str:
    """
    Keep summarizer usage bounded to reduce latency/memory.
    """
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
        # Fail-safe: return a trimmed excerpt
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
    # keep informative tokens
    keep = [
        p for p in parts
        if len(p) >= 4 and p not in STOPWORDS and not p.isdigit()
    ]
    return keep

def _top_keywords_from_text(text: str, k: int = 18) -> list[str]:
    toks = _tokens(text)
    freq: dict[str, int] = {}
    for w in toks:
        freq[w] = freq.get(w, 0) + 1
    # sort by freq desc, then alphabetically
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:k]]

def process_job_description(jd_text: str):
    summary = summarize_text(jd_text, max_len=140)
    # keywords from BOTH raw and summary improves coverage
    kw = _top_keywords_from_text(jd_text + " " + summary, k=22)
    return {"raw": jd_text, "summary": summary, "keywords": kw}

def process_resume(res_text: str):
    summary = summarize_text(res_text, max_len=140)
    return {"raw": res_text, "summary": summary}

def keyword_match_score(jd_keywords, resume_text_or_summary: str) -> float:
    if not jd_keywords:
        return 0.0
    resume_words = set(_tokens(resume_text_or_summary))
    hits = sum(1 for kw in jd_keywords if kw in resume_words)
    return hits / len(jd_keywords)

def _extract_years(resume_raw: str) -> int | None:
    # e.g., "5 years", "7+ yrs", "10+ years"
    patterns = re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs)\b", (resume_raw or "").lower())
    if not patterns:
        return None
    nums = []
    for p in patterns:
        try:
            nums.append(int(p))
        except Exception:
            pass
    return max(nums) if nums else None

def _extract_email(resume_raw: str) -> str | None:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_raw or "")
    return m.group(0) if m else None

def _extract_linkedin(resume_raw: str) -> str | None:
    m = re.search(r"(https?://(www\.)?linkedin\.com/[^\s]+)", resume_raw or "", flags=re.I)
    return m.group(1) if m else None

def _best_resume_snippets(resume_raw: str, jd_keywords: list[str], top_n: int = 2) -> list[str]:
    """
    Pick real lines/sentences from resume that match the JD keywords most.
    This makes highlights feel genuinely grounded in the resume.
    """
    if not resume_raw or not jd_keywords:
        return []

    # split into lines first (works better for resumes)
    lines = [l.strip() for l in re.split(r"[\n\r]+", resume_raw) if l.strip()]
    # also consider sentence splits for long paragraphs
    if len(lines) < 10:
        lines = re.split(r"(?<=[.!?])\s+", resume_raw)

    jd_set = set(jd_keywords)
    scored: list[tuple[int, str]] = []
    for line in lines:
        line_clean = _clean_spaces(line)
        if len(line_clean) < 25:
            continue
        toks = set(_tokens(line_clean))
        score = sum(1 for kw in jd_set if kw in toks)
        if score > 0:
            scored.append((score, line_clean))

    scored.sort(key=lambda x: (-x[0], len(x[1])))
    snippets = []
    seen = set()
    for _, s in scored:
        s2 = s[:180] + ("…" if len(s) > 180 else "")
        if s2.lower() in seen:
            continue
        seen.add(s2.lower())
        snippets.append(s2)
        if len(snippets) >= top_n:
            break
    return snippets

def generate_candidate_highlights(result_dict: dict) -> str:
    """
    More insightful + natural highlight:
    - Verdict + why
    - Matched vs missing keywords
    - Real resume snippets most relevant to the JD
    - Quick extracted signals (years/email/linkedin) when present
    """
    label = result_dict["fit"]["label_name"]
    score_pct = result_dict["final_score"] * 100
    sim = result_dict["similarity"]
    kw = result_dict["keyword_score"]

    jd_keywords = result_dict["jd"]["keywords"]
    resume_raw = result_dict["resume"]["raw"]
    resume_summary = result_dict["resume"]["summary"]

    # matched / missing
    resume_token_set = set(_tokens(resume_raw + " " + resume_summary))
    matched = [k for k in jd_keywords if k in resume_token_set]
    missing = [k for k in jd_keywords if k not in resume_token_set]

    matched_top = matched[:6]
    missing_top = missing[:6]

    years = _extract_years(resume_raw)
    email = _extract_email(resume_raw)
    linkedin = _extract_linkedin(resume_raw)

    snippets = _best_resume_snippets(resume_raw, jd_keywords, top_n=2)

    # verdict tone
    if score_pct >= 80:
        verdict = "Strong match"
    elif score_pct >= 60:
        verdict = "Promising match"
    else:
        verdict = "Needs review"

    reasons = []
    # Keep reasons human-readable
    if sim >= 0.8:
        reasons.append("resume content aligns closely with the JD")
    elif sim >= 0.65:
        reasons.append("there is moderate alignment with the JD")

    if kw >= 0.6:
        reasons.append("many key requirements are explicitly mentioned")
    elif kw >= 0.4:
        reasons.append("some key requirements are mentioned")

    # Compose highlight
    parts = []
    parts.append(f"**{verdict} ({score_pct:.1f}% • {label})**")
    if reasons:
        parts.append("Why: " + "; ".join(reasons) + ".")

    if matched_top:
        parts.append("Matched keywords: " + ", ".join(matched_top) + ".")
    if missing_top and score_pct < 85:
        parts.append("Missing / not obvious: " + ", ".join(missing_top) + ".")

    # snippets grounded in resume
    if snippets:
        if len(snippets) == 1:
            parts.append("Resume evidence: “" + snippets[0] + "”")
        else:
            parts.append("Resume evidence:")
            parts.append(f"- “{snippets[0]}”")
            parts.append(f"- “{snippets[1]}”")

    # quick signals (optional)
    extras = []
    if years is not None and years >= 2:
        extras.append(f"{years}+ years mentioned")
    if linkedin:
        extras.append("LinkedIn link found")
    elif email:
        extras.append("Email found")

    if extras:
        parts.append("_Quick signals_: " + " • ".join(extras) + ".")

    return "\n".join(parts)

def evaluate_candidate(jd_text: str, res_text: str, weights: dict):
    jd = process_job_description(jd_text)
    res = process_resume(res_text)

    sim_raw = compute_similarity(jd["summary"], res["summary"])
    sim_norm = (sim_raw + 1) / 2 if sim_raw < 1 else min(sim_raw, 1.0)

    # keyword coverage should use raw+summary for better detection
    kw_score = keyword_match_score(jd["keywords"], res["raw"] + " " + res["summary"])

    fit = predict_fit_label(jd_text, res_text)
    if len(fit["probs"]) >= 3:
        prob_good_fit = fit["probs"][2]
    else:
        # 2-class fallback: treat predicted class prob as "confidence"
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
    return sorted(results, key=lambda x: x["final_score"], reverse=True)

def map_priority(score: float) -> str:
    if score >= 0.8:
        return "Interview priority: HIGH"
    elif score >= 0.6:
        return "Interview priority: MEDIUM"
    else:
        return "Interview priority: LOW"

# ---------- Upload handlers + clear ----------

def _read_uploaded_file(uploaded) -> str:
    if uploaded is None:
        return ""
    # NOTE: uploaded is an UploadedFile which behaves like a stream
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

def on_single_upload_change():
    """
    Called when file uploader changes.
    Safe place to update st.session_state for resume_text.
    """
    up = st.session_state.get("single_upload", None)
    if up is None:
        return

    extracted = _read_uploaded_file(up)
    # Fill text only if empty (don’t overwrite what user typed)
    if extracted and not st.session_state.get("resume_text", "").strip():
        st.session_state["resume_text"] = extracted
        st.session_state["upload_note"] = f"Text extracted from: {getattr(up, 'name', '')[:40]}"
    elif extracted:
        st.session_state["upload_note"] = f"Text extracted from: {getattr(up, 'name', '')[:40]}"

def clear_all_inputs():
    """
    Clears JD + resume + uploads + outputs.
    For uploads: change uploader keys by bumping a nonce.
    """
    st.session_state["jd_text"] = ""
    st.session_state["resume_text"] = ""
    st.session_state["upload_note"] = ""
    st.session_state["active_candidate_idx"] = 0

    # Clear results (so it doesn't look "uncleared")
    st.session_state.pop("last_single_result", None)
    st.session_state.pop("last_batch_results", None)

    # Reset weights UI values (optional)
    st.session_state["w_clf"] = DEFAULT_WEIGHTS["classifier"]
    st.session_state["w_sim"] = DEFAULT_WEIGHTS["similarity"]
    st.session_state["w_kw"] = DEFAULT_WEIGHTS["keywords"]

    # Most important: bump uploader nonce so file_uploader widgets reset
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1

    st.rerun()

# ============================================================
# 3. HEADER
# ============================================================

st.title("🔍 MatchAI: Candidate Suitability Screening")
st.markdown(
    "<span class='muted'>Screen and prioritise candidates against a specific job description.</span>",
    unsafe_allow_html=True,
)

# Ensure nonce exists
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0

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

        # Key includes nonce so Clear inputs truly resets uploader
        uploaded_file = st.file_uploader(
            "Upload resume (PDF / Word / TXT, optional)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"single_upload_{st.session_state['uploader_nonce']}",
            on_change=on_single_upload_change,
        )
        # NOTE: because we used dynamic key, read the real object from session_state:
        # In on_change, we used "single_upload" -> so we also set an alias here:
        st.session_state["single_upload"] = uploaded_file

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='pill-input'>", unsafe_allow_html=True)
        resume_text = st.text_area(
            "",
            key="resume_text",
            height=220,
            placeholder="Paste resume text here...",
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
            key=f"batch_upload_{st.session_state['uploader_nonce']}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_files:
            st.caption(f"Maximum {MAX_RESUMES} resumes per batch.")
            if len(uploaded_files) > MAX_RESUMES:
                st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Clear button
    clear_col, _ = st.columns([0.2, 0.8])
    with clear_col:
        if st.button("Clear inputs"):
            clear_all_inputs()

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

    # Weights expander
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
                key="slider_w_clf",
            )
        with c2:
            w_sim = st.slider(
                "Weight: Similarity",
                0.0,
                1.0,
                float(st.session_state.w_sim),
                0.05,
                key="slider_w_sim",
            )
        with c3:
            w_kw = st.slider(
                "Weight: Keywords",
                0.0,
                1.0,
                float(st.session_state.w_kw),
                0.05,
                key="slider_w_kw",
            )

        # store back
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

    # Ensure weights exist even if expander never opened
    if "weights" not in locals():
        # normalise defaults just in case
        tot = DEFAULT_WEIGHTS["classifier"] + DEFAULT_WEIGHTS["similarity"] + DEFAULT_WEIGHTS["keywords"]
        weights = {
            "classifier": DEFAULT_WEIGHTS["classifier"] / tot,
            "similarity": DEFAULT_WEIGHTS["similarity"] / tot,
            "keywords": DEFAULT_WEIGHTS["keywords"] / tot,
        }

    st.markdown("---")

    # Score buttons near top of evaluation card
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
            st.session_state["last_single_result"] = result

    # Render cached single result (so it stays visible)
    if "last_single_result" in st.session_state:
        result = st.session_state["last_single_result"]

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
        st.markdown(result["highlight"])
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
            st.write("**JD keywords (top)**")
            st.write(result["jd"]["keywords"][:20])

    # ---------------- BATCH MODE RESULT ----------------
    if batch_btn:
        jd_val = st.session_state.get("jd_text", "")
        # dynamic key means uploader value is in the local variable `uploaded_files` already
        # but if user didn't scroll back up, it might be None; best effort:
        batch_files = uploaded_files if mode != "Single candidate" else None

        if not jd_val.strip():
            st.error("Please provide a job description in the Inputs section.")
        elif not batch_files:
            st.error("Please upload at least one resume file in the Inputs section.")
        else:
            MAX_RESUMES = 30
            if len(batch_files) > MAX_RESUMES:
                st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")
            else:
                resumes_list = []
                for f in batch_files:
                    text = _read_uploaded_file(f)
                    if text.strip():
                        resumes_list.append({"name": f.name, "text": text})

                if not resumes_list:
                    st.error("No readable text found in uploaded resumes.")
                else:
                    with st.spinner("Evaluating all candidates..."):
                        batch_results = evaluate_batch(jd_val, resumes_list, weights)
                    st.session_state["last_batch_results"] = batch_results
                    st.session_state["active_candidate_idx"] = 0

    # Render cached batch results
    if "last_batch_results" in st.session_state:
        batch_results = st.session_state["last_batch_results"]

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

        # show top + allow browsing other highlights
        st.markdown("<div class='section-title'>Candidate highlights</div>", unsafe_allow_html=True)

        options = [f"{i+1}. {r.get('file_name','Candidate')}" for i, r in enumerate(batch_results)]
        idx = st.session_state.get("active_candidate_idx", 0)
        idx = st.selectbox("Select a candidate", options=options, index=min(idx, len(options)-1))
        # idx is a label string now; parse
        selected_i = int(str(idx).split(".")[0]) - 1
        st.session_state["active_candidate_idx"] = selected_i

        sel = batch_results[selected_i]
        st.write(f"**File:** {sel.get('file_name', 'N/A')}")
        st.write(f"**Final score:** {sel['final_score']*100:.1f}%")
        st.write(f"**Fit label:** {sel['fit']['label_name']}")
        st.markdown(sel["highlight"])

        with st.expander("View all candidate highlights"):
            for r in batch_results:
                st.markdown(f"**{r.get('file_name', 'Candidate')}**")
                st.write(
                    f"- Final score: {r['final_score']*100:.1f}% "
                    f"(label: {r['fit']['label_name']})"
                )
                st.markdown(r["highlight"])
                st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# END
# ============================================================
