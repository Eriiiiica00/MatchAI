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
    </style>
    """,
    unsafe_allow_html=True,
)

# CPU-friendly defaults on Streamlit Cloud
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(1)
except Exception:
    pass

# nonce exists early (used to reset uploaders)
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0


# ============================================================
# 1. LOAD CONFIG & MODELS (Classifier + Summarizer + Embeddings)
#    (NER removed to save memory & improve output relevance)
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
# 2. HELPER FUNCTIONS
# ============================================================

STOPWORDS = {
    "the","and","for","with","from","that","this","you","your","are","our","will","have",
    "role","work","team","teams","using","use","used","able","must","should","within",
    "include","including","responsibilities","requirements","years","year","experience",
    "skills","skill","ability","strong","good","excellent","preferred","plus","etc"
}

ACTION_VERBS = {
    "led","lead","owned","manage","managed","built","build","create","created","design","designed",
    "implement","implemented","improve","improved","increase","increased","reduce","reduced",
    "deliver","delivered","launch","launched","optimize","optimized","automate","automated",
    "analyze","analyzed","drive","drove","scale","scaled","partner","partnered","negotiate","negotiated"
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

def summarize_text(text: str, max_len: int = 140) -> str:
    """
    Keep summarizer usage bounded.
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

def process_job_description(jd_text: str):
    summary = summarize_text(jd_text, max_len=140)
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
    patterns = re.findall(r"(\d{1,2})\s*\+?\s*(?:years|yrs)\b", (resume_raw or "").lower())
    nums = []
    for p in patterns:
        try:
            nums.append(int(p))
        except Exception:
            pass
    return max(nums) if nums else None

def _looks_like_metric(line: str) -> bool:
    # %, $, numbers, KPI-ish patterns
    return bool(re.search(r"(\d+%|\$\s?\d+|\b\d+\b|\bKPIs?\b|\bOKRs?\b|\bROI\b|\bDSO\b|\bNPS\b)", line, flags=re.I))

def _action_verb_bonus(line: str) -> int:
    toks = set(_tokens(line))
    return 1 if any(v in toks for v in ACTION_VERBS) else 0

def extract_evidence_bullets(resume_raw: str, jd_keywords: list[str], top_n: int = 5) -> list[str]:
    """
    Pick grounded resume lines as evidence.
    Scoring: keyword hits + metric bonus + action verb bonus.
    """
    if not resume_raw or not jd_keywords:
        return []

    # split lines (resumes are line-based)
    lines = [l.strip() for l in re.split(r"[\n\r]+", resume_raw) if l.strip()]
    # fallback: sentence split if too few lines
    if len(lines) < 8:
        lines = re.split(r"(?<=[.!?])\s+", resume_raw)

    jd_set = set(jd_keywords)
    scored: list[tuple[float, str]] = []

    for line in lines:
        line_clean = _clean_spaces(line)
        if len(line_clean) < 25:
            continue

        toks = set(_tokens(line_clean))
        hit = sum(1 for kw in jd_set if kw in toks)

        if hit == 0:
            continue

        metric_bonus = 1 if _looks_like_metric(line_clean) else 0
        verb_bonus = _action_verb_bonus(line_clean)
        length_penalty = 0.2 if len(line_clean) > 220 else 0.0

        score = hit + metric_bonus + verb_bonus - length_penalty
        scored.append((score, line_clean))

    scored.sort(key=lambda x: (-x[0], len(x[1])))

    bullets = []
    seen = set()
    for _, s in scored:
        s2 = s[:190] + ("…" if len(s) > 190 else "")
        key = s2.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"“{s2}”")
        if len(bullets) >= top_n:
            break

    return bullets

def build_interview_kit(
    jd_keywords: list[str],
    matched_keywords: list[str],
    missing_keywords: list[str],
    evidence_bullets: list[str],
    score_pct: float
) -> list[str]:
    """
    Deterministic, recruiter-grade interview kit:
    - Evidence drill-down
    - Depth on matched keywords
    - Gap probes on missing keywords
    - Risk checks
    """
    kit: list[str] = []

    # 1) Evidence drill-down (best)
    for b in evidence_bullets[:2]:
        kit.append(f"Deep dive: In the resume you mention {b} — what was your role, what constraints did you face, and how did you measure impact?")

    # 2) Matched keyword depth
    for kw in matched_keywords[:2]:
        kit.append(f"Depth check ({kw}): Walk me through the most complex {kw}-related work you’ve done. What trade-offs did you make and why?")

    # 3) Gap probes
    for kw in missing_keywords[:2]:
        kit.append(f"Gap probe ({kw}): This role expects {kw}. What’s your closest comparable experience, and what would you need to ramp up quickly?")

    # 4) Risk checks (tailored by score)
    if score_pct < 60:
        kit.append("Clarity check: Pick one project you led end-to-end. Explain the goal, your specific contribution, the timeline, and the outcome (with numbers if possible).")
    kit.append("Stakeholder check: Tell me about a time you had conflicting stakeholders. How did you align them and what was the result?")

    # Ensure not too long
    return kit[:8]

def evaluate_candidate(jd_text: str, res_text: str, weights: dict):
    jd = process_job_description(jd_text)
    res = process_resume(res_text)

    # similarity on summaries (bounded length)
    sim_raw = compute_similarity(jd["summary"], res["summary"])
    sim_norm = (sim_raw + 1) / 2 if sim_raw < 1 else min(sim_raw, 1.0)

    # keyword coverage against raw+summary
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

    # evidence + interview kit (grounded)
    resume_token_set = set(_tokens(res["raw"] + " " + res["summary"]))
    matched = [k for k in jd["keywords"] if k in resume_token_set]
    missing = [k for k in jd["keywords"] if k not in resume_token_set]
    evidence = extract_evidence_bullets(res["raw"], jd["keywords"], top_n=5)
    score_pct = float(final_score) * 100.0
    interview_kit = build_interview_kit(jd["keywords"], matched, missing, evidence, score_pct)

    years = _extract_years(res["raw"])

    return {
        "jd": jd,
        "resume": res,
        "similarity_raw": sim_raw,
        "similarity": sim_norm,
        "keyword_score": kw_score,
        "fit": fit,
        "prob_good_fit": float(prob_good_fit),
        "final_score": float(final_score),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "evidence_bullets": evidence,
        "interview_kit": interview_kit,
        "years_signal": years,
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
    return sorted(results, key=lambda x: x["final_score"], reverse=True)

def map_priority(score: float) -> str:
    if score >= 0.8:
        return "Interview priority: HIGH"
    elif score >= 0.6:
        return "Interview priority: MEDIUM"
    else:
        return "Interview priority: LOW"

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
# ✅ NEW EVALUATION (Safe reset)
# ============================================================

def new_evaluation():
    """
    Safe reset:
    - clears text inputs + notes
    - clears cached results
    - resets candidate selection
    - bumps uploader_nonce so upload widgets reset
    - reruns
    """
    st.session_state["jd_text"] = ""
    st.session_state["resume_text"] = ""
    st.session_state["upload_note"] = ""
    st.session_state["active_candidate_idx"] = 0

    st.session_state.pop("last_single_result", None)
    st.session_state.pop("last_batch_results", None)

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

    uploaded_files = None  # define for safety

    if mode == "Single candidate":
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='pill-upload'>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload resume (PDF / Word / TXT, optional)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"single_upload_{st.session_state['uploader_nonce']}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Populate resume_text only if empty
        if uploaded_file is not None:
            extracted = _read_uploaded_file(uploaded_file)
            if extracted and not st.session_state.get("resume_text", "").strip():
                # Safe enough in practice because text_area below reads session_state.
                st.session_state["resume_text"] = extracted
            if extracted:
                st.session_state["upload_note"] = f"Text extracted from: {uploaded_file.name[:40]}"

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

        batch_key = f"batch_upload_{st.session_state['uploader_nonce']}"
        uploaded_files = st.file_uploader(
            "Upload multiple resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=batch_key,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_files:
            st.caption(f"Maximum {MAX_RESUMES} resumes per batch.")
            if len(uploaded_files) > MAX_RESUMES:
                st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")
        st.markdown("</div>", unsafe_allow_html=True)

    # New evaluation button
    btn_col, _ = st.columns([0.25, 0.75])
    with btn_col:
        st.button("New evaluation", on_click=new_evaluation)

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

    # weights expander (normalised to 100% automatically)
    with st.expander("Adjust scoring weights (optional)"):
        st.caption("Weights are normalised automatically.")

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
            weights = {"classifier": w_clf / total, "similarity": w_sim / total, "keywords": w_kw / total}

        st.caption(
            f"Normalised weights →  P(Good Fit): {weights['classifier']:.2f},  "
            f"Similarity: {weights['similarity']:.2f},  Keywords: {weights['keywords']:.2f}"
        )

        if st.button("Reset to default weights"):
            st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
            st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
            st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]
            st.rerun()

    if "weights" not in locals():
        tot = DEFAULT_WEIGHTS["classifier"] + DEFAULT_WEIGHTS["similarity"] + DEFAULT_WEIGHTS["keywords"]
        weights = {
            "classifier": DEFAULT_WEIGHTS["classifier"] / tot,
            "similarity": DEFAULT_WEIGHTS["similarity"] / tot,
            "keywords": DEFAULT_WEIGHTS["keywords"] / tot,
        }

    st.markdown("---")

    # score buttons
    button_row = st.columns([0.35, 0.35, 0.3])
    with button_row[0]:
        single_btn = st.button("🔍 Score single candidate")
    with button_row[1]:
        batch_btn = st.button("🔍 Score candidates (batch)")

    # ---------------- SINGLE ----------------
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

        # Evidence bullets
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Evidence bullets</div>", unsafe_allow_html=True)
        evidence = result.get("evidence_bullets", [])
        if evidence:
            for b in evidence:
                st.markdown(f"- {b}")
        else:
            st.markdown("<span class='muted'>No strong JD-linked evidence lines were detected in the resume.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Interview kit
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Interview kit</div>", unsafe_allow_html=True)
        kit = result.get("interview_kit", [])
        for q in kit:
            st.markdown(f"- {q}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- BATCH ----------------
    if batch_btn:
        jd_val = st.session_state.get("jd_text", "")

        if not jd_val.strip():
            st.error("Please provide a job description in the Inputs section.")
        elif st.session_state.get("mode") != "Batch (multiple resumes)":
            st.error("Please switch to Batch mode in the Inputs section to score multiple resumes.")
        elif not uploaded_files:
            st.error("Please upload at least one resume file in the Inputs section.")
        else:
            MAX_RESUMES = 30
            if len(uploaded_files) > MAX_RESUMES:
                st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")
            else:
                resumes_list = []
                for f in uploaded_files:
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
                    "Priority": map_priority(r["final_score"]).replace("Interview priority: ", ""),
                    "Similarity": round(r["similarity"], 3),
                    "Keyword score": round(r["keyword_score"], 3),
                }
            )
        st.dataframe(rows, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Candidate selector
        st.markdown("<div class='section-title'>Candidate selector</div>", unsafe_allow_html=True)
        options = [f"{i+1}. {r.get('file_name','Candidate')}" for i, r in enumerate(batch_results)]
        current_i = int(st.session_state.get("active_candidate_idx", 0))
        label = st.selectbox("Select a candidate", options=options, index=min(current_i, len(options)-1))
        selected_i = int(str(label).split(".")[0]) - 1
        st.session_state["active_candidate_idx"] = selected_i

        sel = batch_results[selected_i]

        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Selected candidate</div>", unsafe_allow_html=True)
        st.write(f"**File:** {sel.get('file_name', 'N/A')}")
        st.write(f"**Final score:** {sel['final_score']*100:.1f}%")
        st.write(f"**Fit label:** {sel['fit']['label_name']}")
        st.write(f"**{map_priority(sel['final_score'])}**")
        st.markdown("</div>", unsafe_allow_html=True)

        # Evidence bullets (short)
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Evidence bullets (short)</div>", unsafe_allow_html=True)
        ev = sel.get("evidence_bullets", [])[:3]
        if ev:
            for b in ev:
                st.markdown(f"- {b}")
        else:
            st.markdown("<span class='muted'>No strong JD-linked evidence lines were detected in this resume.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Interview focus (short)
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Interview focus (short)</div>", unsafe_allow_html=True)
        focus = sel.get("interview_kit", [])[:4]
        for q in focus:
            st.markdown(f"- {q}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# END
# ============================================================
