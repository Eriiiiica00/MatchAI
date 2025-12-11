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
# 0. BASIC SETUP & GLOBAL STATE
# ============================================================

st.set_page_config(
    page_title="MatchAI: Candidate Suitability Screening",
    page_icon=None,  # avoid cropped emoji favicon
    layout="wide",
)

# Minimal Apple-ish neutral styling
st.markdown(
    """
    <style>
    body, .main {
        background-color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }
    .app-card {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 20px 24px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.03);
        border: 1px solid #e5e5ea;
    }
    .metric-box {
        padding: 10px 14px;
        border-radius: 12px;
        background-color: #f2f2f7;
        border: 1px solid #e5e5ea;
        font-size: 0.9rem;
    }
    .section-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
        color: #1c1c1e;
    }
    /* Remove heavy borders from expanders */
    div[data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    div[data-testid="stExpander"] details {
        background-color: transparent !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 500;
        color: #3a3a3c;
    }
    /* Make text areas and results consistent full-width rounded boxes */
    textarea, .stTextArea textarea {
        border-radius: 18px !important;
        border: 1px solid #e5e5ea !important;
    }
    /* Buttons: neutral grey, rounded */
    .stButton > button {
        border-radius: 999px;
        border: 1px solid #d1d1d6;
        background-color: #f2f2f7;
        color: #1c1c1e;
        padding: 0.45rem 1.4rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #e5e5ea;
        border-color: #c7c7cc;
    }
    /* File uploader: soften look */
    section[data-testid="stFileUploader"] {
        padding: 0.5rem 0.75rem;
        border-radius: 14px;
        border: 1px dashed #e5e5ea;
        background-color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialise session_state containers
for key, default in [
    ("single_result", None),
    ("batch_results", None),
    ("mode", "Single candidate"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# 1. LOAD CONFIG & MODELS
# ============================================================

@st.cache_resource
def load_matchai_config(path: str | None = None):
    """
    Load matchai_config.json from the same folder as streamlit_app.py.
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
            f"Could not load matchai_config.json at: {path}\n"
            f"Using fallback config. Error: {e}"
        )
        # Fallback config (2-class testing model)
        return {
            "fine_tuned_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
            "summarization_model": "sshleifer/distilbart-cnn-12-6",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "ner_model": "dslim/bert-base-NER",
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
    # Classifier
    clf_id = cfg["fine_tuned_model_id"]
    clf_tokenizer = AutoTokenizer.from_pretrained(clf_id)
    clf_model = AutoModelForSequenceClassification.from_pretrained(clf_id)
    clf_model.to(device)
    clf_model.eval()

    # Summarizer
    summarizer = hf_pipeline(
        "summarization",
        model=cfg["summarization_model"],
        device=0 if torch.cuda.is_available() else -1,
    )

    # Embedding model
    sim_model = SentenceTransformer(cfg["embedding_model"])

    # NER pipeline
    ner_pipe = hf_pipeline(
        "ner", model=cfg["ner_model"], grouped_entities=True
    )

    # Label mapping
    raw_map = cfg.get("label_id2name", {"0": "Not Fit", "1": "Good Fit"})
    label_id2name = {int(k): v for k, v in raw_map.items()}

    return clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name

config = load_matchai_config()
clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name = load_models_and_pipelines(config)
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

def extract_entities(text: str):
    if not text or not isinstance(text, str):
        return {"ORG": [], "PER": [], "LOC": []}
    ents = ner_pipe(text[:2000])
    result = {"ORG": [], "PER": [], "LOC": []}
    for e in ents:
        label = e.get("entity_group")
        word = e.get("word", "").strip()
        # Basic noise filter: ignore very short / obviously partial tokens
        if len(word) < 3:
            continue
        if label in result and word:
            result[label].append(word)
    return result

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
    keywords = list({w.lower().strip(",.!?;:") for w in summary.split() if len(w) > 4})
    return {"raw": jd_text, "summary": summary, "keywords": keywords}

def process_resume(res_text: str):
    summary = summarize_text(res_text)
    entities = extract_entities(res_text)
    return {"raw": res_text, "summary": summary, "entities": entities}

def keyword_match_score(jd_keywords, resume_summary: str) -> float:
    if not jd_keywords:
        return 0.0
    resume_words = set(
        w.lower().strip(",.!?;:") for w in resume_summary.split()
    )
    hits = sum(1 for kw in jd_keywords if kw in resume_words)
    return hits / len(jd_keywords)

def generate_candidate_highlights(result_dict: dict) -> str:
    """
    Produce a short, natural summary that gives HR an immediate feel for the candidate.
    Uses: final score, P(Good Fit), similarity, keyword overlap, and ORG entities.
    """
    label = result_dict["fit"]["label_name"]
    score = result_dict["final_score"]
    score_pct = score * 100
    sim = result_dict["similarity"]
    kw = result_dict["keyword_score"]
    ents = result_dict["resume"]["entities"]
    orgs = list(dict.fromkeys(ents.get("ORG", [])))

    jd_keywords = result_dict["jd"]["keywords"]
    resume_summary_words = set(
        w.lower().strip(",.!?;:") for w in result_dict["resume"]["summary"].split()
    )
    matched_keywords = [
        kw for kw in jd_keywords if kw in resume_summary_words
    ][:5]

    parts = []

    # Overall match
    if label.lower().startswith("good"):
        parts.append(
            f"Strong overall match ({score_pct:.0f}% suitability)."
        )
    elif "potential" in label.lower():
        parts.append(
            f"Potential fit, with room to grow into the role ({score_pct:.0f}% suitability)."
        )
    else:
        parts.append(
            f"Limited match based on current resume ({score_pct:.0f}% suitability)."
        )

    # Similarity
    if sim >= 0.8:
        parts.append(
            f"Resume content closely mirrors the job description (similarity {sim:.2f})."
        )
    elif sim >= 0.6:
        parts.append(
            f"Some overlap with the role requirements (similarity {sim:.2f})."
        )

    # Keyword coverage
    if matched_keywords and kw >= 0.5:
        parts.append(
            "Good coverage of key requirements such as "
            + ", ".join(matched_keywords)
            + "."
        )
    elif matched_keywords:
        parts.append(
            "Covers a subset of important requirements (e.g. "
            + ", ".join(matched_keywords)
            + ")."
        )

    # Organisations
    if orgs:
        cleaned_orgs = []
        for o in orgs:
            o_clean = o.strip(",.;: ").title()
            if 2 < len(o_clean) <= 40:
                cleaned_orgs.append(o_clean)
        cleaned_orgs = list(dict.fromkeys(cleaned_orgs))[:3]
        if cleaned_orgs:
            parts.append(
                "Experience with organisations such as "
                + ", ".join(cleaned_orgs)
                + "."
            )

    if not parts:
        return "No standout automated signals; the profile may require manual review."

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
    """
    resumes_list: list of dicts [{"name": "...", "text": "..."}, ...]
    """
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

# ============================================================
# 3. HEADER & GLOBAL CONTROLS
# ============================================================

st.markdown(
    "<h2 style='margin-bottom:0.2rem;'>🔍 MatchAI: Candidate Suitability Screening</h2>",
    unsafe_allow_html=True,
)
st.write("Screen and prioritise candidates against a specific job description.")

# Clear button (resets session_state & reruns)
clear_col, _ = st.columns([0.2, 0.8])
with clear_col:
    if st.button("Clear inputs & outputs"):
        for key in [
            "jd_text",
            "resume_text",
            "single_result",
            "batch_results",
            "w_clf",
            "w_sim",
            "w_kw",
            "current_weights",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

st.markdown("")

# ============================================================
# 4. INPUT CARD (1 COLUMN)
# ============================================================

with st.container():
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)

    st.markdown("#### Inputs")

    jd_text = st.text_area(
        "Job description",
        key="jd_text",
        height=200,
        placeholder="Paste the job description here (responsibilities, required skills, qualifications)...",
    )

    st.radio(
        "Evaluation mode",
        ["Single candidate", "Batch (multiple resumes)"],
        key="mode",
        horizontal=True,
    )

    if st.session_state.mode == "Single candidate":
        st.caption("Paste resume text or upload a file (PDF / Word / TXT).")

        uploaded_file = st.file_uploader(
            "Upload resume (optional)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
        )

        resume_text = st.text_area(
            "Candidate resume",
            key="resume_text",
            height=200,
            placeholder="Paste resume text here or upload a file above...",
        )

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

            if extracted:
                if not st.session_state.resume_text.strip():
                    st.session_state.resume_text = extracted
                st.info(f"Text extracted from file: {uploaded_file.name[:40]}")

    else:
        st.caption("Upload multiple resumes (PDF / Word / TXT). Maximum 30 resumes per batch.")
        MAX_RESUMES = 30
        uploaded_files = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files and len(uploaded_files) > MAX_RESUMES:
            st.error(f"Too many resumes uploaded. Maximum allowed is {MAX_RESUMES}.")
            st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 5. SCORING EXPLANATION + WEIGHT CONTROLS
# ============================================================

# Ensure default weight state
if "w_clf" not in st.session_state:
    st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
if "w_sim" not in st.session_state:
    st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
if "w_kw" not in st.session_state:
    st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]
if "current_weights" not in st.session_state:
    st.session_state.current_weights = DEFAULT_WEIGHTS

st.markdown("")
with st.container():
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("#### Scoring")

    top_row = st.columns([0.35, 0.3, 0.35])
    with top_row[0]:
        # Score buttons will be used later (just placeholders here)
        pass
    with top_row[1]:
        pass
    with top_row[2]:
        # Info popover for score formula (no big formula printed)
        pop = st.popover("ℹ️ How is the score calculated?")
        with pop:
            st.write(
                "The final suitability score combines three signals:\n\n"
                "- **P(Good Fit)** from the fine-tuned classifier\n"
                "- **Semantic similarity** between JD and resume summaries\n"
                "- **Keyword coverage** of JD requirements in the resume\n\n"
                "Default weights: 0.5 × P(Good Fit) + 0.3 × Similarity + 0.2 × Keyword coverage.\n"
                "You can adjust these below for different HR priorities."
            )

    # Weight controls
    with st.expander("Adjust scoring weights (optional)"):
        st.caption("Weights are automatically normalised to sum to 1. Reset to go back to defaults.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.w_clf = st.slider(
                "Weight: P(Good Fit)",
                0.0,
                1.0,
                float(st.session_state.w_clf),
                0.05,
            )
        with c2:
            st.session_state.w_sim = st.slider(
                "Weight: Similarity",
                0.0,
                1.0,
                float(st.session_state.w_sim),
                0.05,
            )
        with c3:
            st.session_state.w_kw = st.slider(
                "Weight: Keywords",
                0.0,
                1.0,
                float(st.session_state.w_kw),
                0.05,
            )

        total_raw = (
            st.session_state.w_clf
            + st.session_state.w_sim
            + st.session_state.w_kw
        )
        if total_raw == 0:
            weights = DEFAULT_WEIGHTS.copy()
        else:
            weights = {
                "classifier": st.session_state.w_clf / total_raw,
                "similarity": st.session_state.w_sim / total_raw,
                "keywords": st.session_state.w_kw / total_raw,
            }

        st.session_state.current_weights = weights

        st.caption(
            f"Normalised weights → P(Good Fit): {weights['classifier']:.2f}, "
            f"Similarity: {weights['similarity']:.2f}, "
            f"Keywords: {weights['keywords']:.2f}"
        )

        if st.button("Reset weights to default"):
            st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
            st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
            st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]
            st.session_state.current_weights = DEFAULT_WEIGHTS.copy()
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

weights = st.session_state.current_weights

# ============================================================
# 6. RUN EVALUATION
# ============================================================

st.markdown("")
with st.container():
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)

    # Primary action buttons (now clearly visible)
    col_buttons = st.columns([0.35, 0.65])
    with col_buttons[0]:
        if st.session_state.mode == "Single candidate":
            score_clicked = st.button("🔍 Score single candidate", type="primary")
        else:
            score_clicked = st.button("🔍 Score multiple resumes", type="primary")
    with col_buttons[1]:
        st.caption("Run the evaluation after providing a job description and resume(s).")

    if score_clicked:
        if not jd_text or not jd_text.strip():
            st.error("Please provide a job description.")
        else:
            if st.session_state.mode == "Single candidate":
                resume_text = st.session_state.get("resume_text", "")
                if not resume_text.strip():
                    st.error("Please provide resume text or upload a file.")
                else:
                    with st.spinner("Evaluating candidate..."):
                        res = evaluate_candidate(jd_text, resume_text, weights)
                    st.session_state.single_result = res
                    st.session_state.batch_results = None
            else:
                if not uploaded_files:
                    st.error("Please upload at least one resume file.")
                else:
                    resumes_list = []
                    for f in uploaded_files:
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
                            batch_res = evaluate_batch(jd_text, resumes_list, weights)
                        st.session_state.batch_results = batch_res
                        st.session_state.single_result = None

    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    if st.session_state.single_result is not None:
        result = st.session_state.single_result
        score = result["final_score"]
        score_pct = score * 100
        label = result["fit"]["label_name"]
        prob_good = result["prob_good_fit"]

        st.markdown("### Result – Single candidate")
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-title'>Final suitability score</div>"
            f"<p style='font-size: 1.9rem; margin: 0 0 0.3rem 0;'><b>{score_pct:.1f}%</b></p>"
            f"<p style='margin:0;'>Predicted label: <b>{label}</b></p>"
            f"<p style='margin:0;'>P(Good Fit): <b>{prob_good:.2f}</b> · {map_priority(score)}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("**Profile highlight**")
        st.write(result["highlight"])

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Semantic similarity**")
            st.write(f"{result['similarity']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Keyword coverage**")
            st.write(f"{result['keyword_score']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Model fit confidence**")
            st.write(f"{prob_good:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("View model-level details"):
            st.write("**JD summary:**")
            st.write(result["jd"]["summary"])
            st.write("**Resume summary:**")
            st.write(result["resume"]["summary"])
            st.write("**Extracted organisations (top 5):**")
            st.write(result["resume"]["entities"].get("ORG", [])[:5])

    if st.session_state.batch_results is not None:
        batch_results = st.session_state.batch_results

        st.markdown("### Result – Batch candidate ranking")

        table_rows = []
        for rank, r in enumerate(batch_results, start=1):
            table_rows.append(
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

        st.dataframe(table_rows, use_container_width=True)

        # Allow selecting which candidate to inspect
        names = [r.get("file_name", f"Candidate {i+1}") for i, r in enumerate(batch_results)]
        idx = st.selectbox(
            "Select a candidate to view highlights and details",
            options=list(range(len(names))),
            format_func=lambda i: f"{i+1}. {names[i]}",
        )
        top = batch_results[idx]

        st.markdown("#### Candidate details")
        st.write(f"**File:** {top.get('file_name', 'N/A')}")
        st.write(f"**Final score:** {top['final_score']*100:.1f}%")
        st.write(f"**Fit label:** {top['fit']['label_name']}")
        st.write(f"**Highlight:** {top['highlight']}")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Semantic similarity**")
            st.write(f"{top['similarity']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Keyword coverage**")
            st.write(f"{top['keyword_score']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write("**Model fit confidence**")
            st.write(f"{top['prob_good_fit']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("View model-level details for selected candidate"):
            st.write("**JD summary:**")
            st.write(top["jd"]["summary"])
            st.write("**Resume summary:**")
            st.write(top["resume"]["summary"])
            st.write("**Extracted organisations (top 5):**")
            st.write(top["resume"]["entities"].get("ORG", [])[:5])

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# END
# ============================================================
