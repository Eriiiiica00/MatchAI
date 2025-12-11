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
# 0. BASIC SETUP
# ============================================================

st.set_page_config(
    page_title="MatchAI: Candidate Suitability Screening",
    page_icon="🔍",
    layout="wide",
)

# Apple-ish, but cleaner (no weird twin bars, consistent font)
st.markdown(
    """
    <style>
    :root {
        --matchai-bg: #f5f5f7;
        --matchai-card: #ffffff;
        --matchai-border: #d2d2d7;
        --matchai-accent: #007aff;
        --matchai-accent-hover: #0a84ff;
        --matchai-text-main: #1d1d1f;
        --matchai-text-muted: #6e6e73;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Helvetica Neue", Arial, sans-serif !important;
        color: var(--matchai-text-main) !important;
    }

    .main {
        background-color: var(--matchai-bg);
    }

    /* Generic cards */
    .score-card {
        padding: 1rem 1.5rem;
        border-radius: 0.85rem;
        background-color: var(--matchai-card);
        border: 1px solid var(--matchai-border);
        box-shadow: 0 8px 18px rgba(0,0,0,0.04);
    }

    .metric-box {
        padding: 0.75rem 1rem;
        border-radius: 0.65rem;
        background-color: #f0f1f5;
        border: 1px solid var(--matchai-border);
        font-size: 0.9rem;
    }

    .section-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
        color: var(--matchai-text-main);
    }

    p, label, span, div, h1, h2, h3, h4, h5 {
        color: var(--matchai-text-main) !important;
    }

    /* Primary buttons */
    div.stButton > button {
        background-color: var(--matchai-accent);
        color: #ffffff !important;
        border-radius: 999px;
        border: none;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
    }
    div.stButton > button:hover {
        background-color: var(--matchai-accent-hover);
        color: #ffffff !important;
    }

    /* Text inputs / textareas: remove big pill look / glow */
    textarea, .stTextArea textarea, .stTextInput input {
        border-radius: 8px !important;
        border: 1px solid var(--matchai-border) !important;
        box-shadow: none !important;
        background-color: #ffffff !important;
    }

    /* Radio / sliders accent */
    .st-emotion-cache-1n543e5 input:checked + div,
    .st-emotion-cache-16idsys input:checked + div {
        background-color: var(--matchai-accent) !important;
    }
    .stSlider > div > div > div > div {
        background: var(--matchai-accent) !important;
    }

    /* Dataframe header */
    thead tr th {
        background-color: #eaeaef !important;
        color: var(--matchai-text-main) !important;
    }

    *:focus {
        outline: none !important;
        box-shadow: none !important;
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
        st.error(f"Could not load matchai_config.json at: {path}\nUsing fallback config. Error: {e}")
        return {
            "fine_tuned_model_id": "your-username/matchai-fit-classifier",
            "summarization_model": "sshleifer/distilbart-cnn-12-6",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "ner_model": "dslim/bert-base-NER",
            "weights": {
                "classifier": 0.5,
                "similarity": 0.3,
                "keywords": 0.2,
            },
            "label_id2name": {
                "0": "No Fit",
                "1": "Potential Fit",
                "2": "Good Fit",
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
    raw_map = cfg.get("label_id2name", {"0": "No Fit", "1": "Potential Fit", "2": "Good Fit"})
    label_id2name = {int(k): v for k, v in raw_map.items()}

    return clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name

config = load_matchai_config()
clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name = load_models_and_pipelines(config)

DEFAULT_WEIGHTS = config.get(
    "weights", {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2}
)

# ============================================================
# 2. HELPER FUNCTIONS
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
    ents = ner_pipe(text[:1000])
    result = {"ORG": [], "PER": [], "LOC": []}
    for e in ents:
        label = e.get("entity_group")
        word = e.get("word", "").strip()
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
    keywords = list({w.lower() for w in summary.split() if len(w) > 4})
    return {"raw": jd_text, "summary": summary, "keywords": keywords}

def process_resume(res_text: str):
    summary = summarize_text(res_text)
    entities = extract_entities(res_text)
    return {"raw": res_text, "summary": summary, "entities": entities}

def keyword_match_score(jd_keywords, resume_summary: str) -> float:
    if not jd_keywords:
        return 0.0
    resume_words = set(w.lower() for w in resume_summary.split())
    hits = sum(1 for kw in jd_keywords if kw in resume_words)
    return hits / len(jd_keywords)

def generate_candidate_highlights(result_dict: dict) -> str:
    label = result_dict["fit"]["label_name"]
    score = result_dict["final_score"]
    score_pct = score * 100
    sim = result_dict["similarity"]
    kw = result_dict["keyword_score"]
    ents = result_dict["resume"]["entities"]
    orgs = ents.get("ORG", [])
    pers = ents.get("PER", [])

    strength_bits = []

    if label == "Good Fit" and score_pct >= 80:
        strength_bits.append("strong overall match to the job requirements.")
    elif label == "Potential Fit":
        strength_bits.append("partial match with room to grow into the role.")
    else:
        strength_bits.append(
            "limited alignment with the role based on current resume content."
        )

    if sim >= 0.8:
        strength_bits.append(
            "high semantic similarity between the resume and job description."
        )
    elif sim >= 0.65:
        strength_bits.append(
            "moderate semantic similarity, suggesting some relevant experience."
        )

    if kw >= 0.6:
        strength_bits.append(
            "good coverage of key role-related keywords in the resume."
        )
    elif kw >= 0.4:
        strength_bits.append(
            "some important keywords matching the job description."
        )

    if orgs:
        strength_bits.append(
            f"experience with notable organisations (e.g. {', '.join(orgs[:3])})."
        )

    if pers:
        strength_bits.append(
            "clear identification of personal profile details in the resume."
        )

    if not strength_bits:
        return "No standout signals detected. The candidate may require manual review."

    sentence = "This candidate shows " + " ".join(strength_bits)
    return sentence.strip()

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
        return "Interview Priority: HIGH"
    elif score >= 0.6:
        return "Interview Priority: MEDIUM"
    else:
        return "Interview Priority: LOW"

# ============================================================
# 3. UI LAYOUT
# ============================================================

st.title("🔍 MatchAI: Candidate Suitability Screening")
st.caption("Screen and prioritise candidates against a specific job description.")

col_left, col_right = st.columns([1, 1.2])

# ---------------- LEFT: INPUTS ----------------
with col_left:
    st.markdown("### 1. Job Description & Candidate Resumes")

    jd_text = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the job description here (responsibilities, skills, qualifications)...",
        key="jd_text",
    )

    mode = st.radio(
        "Evaluation mode",
        ["Single candidate", "Batch (multiple resumes)"],
        horizontal=True,
    )

    st.markdown("#### Candidate Resume(s)")

    if mode == "Single candidate":
        uploaded_file = st.file_uploader(
            "Upload resume (PDF / Word / TXT, optional)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            key="single_upload",
        )

        resume_text = st.text_area(
            "Or paste resume text",
            height=220,
            placeholder="Paste the candidate's resume here or upload a file above...",
            key="resume_text",
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
                if not st.session_state.get("resume_text", "").strip():
                    st.session_state["resume_text"] = extracted
                st.info(f"Text extracted from: {uploaded_file.name[:60]}")

    else:
        st.caption("You can upload multiple resumes. Maximum 30 per batch.")
        MAX_RESUMES = 30
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF / Word / TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if uploaded_files and len(uploaded_files) > MAX_RESUMES:
            st.error(f"Too many resumes uploaded. Maximum allowed is {MAX_RESUMES}.")
            st.stop()

    st.markdown("---")
    if st.button("Clear all inputs"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# ---------------- RIGHT: SCORE, WEIGHTS, RESULTS ----------------
with col_right:
    st.markdown("### 2. Scoring & Results")

    st.markdown(
        """
        <div class="score-card">
            <div class="section-title">Final Suitability Score</div>
            <p style="margin-bottom: 0.4rem; color: var(--matchai-text-muted);">
                The score combines three signals:
            </p>
            <ul style="margin-top: 0; margin-bottom: 0.4rem; color: var(--matchai-text-muted);">
                <li><b>P(Good Fit)</b> – classifier confidence</li>
                <li><b>Similarity</b> – semantic match between JD and resume</li>
                <li><b>Keyword Coverage</b> – how much of the JD appears in the resume</li>
            </ul>
            <p style="margin-bottom: 0; color: var(--matchai-text-muted);">
                Default formula: <b>Final Score = 0.5 × P(Good Fit) + 0.3 × Similarity + 0.2 × Keyword Coverage</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Adjust scoring weights (optional)", expanded=False):
        st.caption("Weights are normalised automatically. Use this to shift emphasis (e.g. more classifier vs. more similarity).")

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
            f"Normalised weights → P(Good Fit): {weights['classifier']:.2f}, "
            f"Similarity: {weights['similarity']:.2f}, Keywords: {weights['keywords']:.2f}"
        )

        if st.button("Reset to default weights"):
            st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
            st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
            st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]
            st.experimental_rerun()

    st.markdown("---")

    if mode == "Single candidate":
        score_button = st.button("🔍 Score candidate", use_container_width=True)

        if score_button:
            resume_text_val = st.session_state.get("resume_text", "")
            if not jd_text.strip():
                st.error("Please provide a job description.")
            elif not resume_text_val.strip():
                st.error("Please provide resume text or upload a file.")
            else:
                with st.spinner("Evaluating candidate..."):
                    result = evaluate_candidate(jd_text, resume_text_val, weights)

                score = result["final_score"]
                score_pct = score * 100
                label = result["fit"]["label_name"]
                prob_good = result["prob_good_fit"]

                st.markdown("#### Result – Single Candidate")
                st.markdown('<div class="score-card">', unsafe_allow_html=True)
                st.markdown(
                    f"<div class='section-title'>Final Suitability Score</div>"
                    f"<p style='font-size: 2.2rem; margin: 0.2rem 0 0.4rem 0;'><b>{score_pct:.1f}%</b></p>",
                    unsafe_allow_html=True,
                )
                st.write(f"**Predicted fit label:** {label}")
                st.write(f"**P(Good Fit):** {prob_good:.2f}")
                st.write(f"**{map_priority(score)}**")
                st.write(f"**Candidate highlight:** {result['highlight']}")
                st.markdown("</div>", unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.write("**Semantic Similarity**")
                    st.write(f"{result['similarity']:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with m2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.write("**Keyword Coverage**")
                    st.write(f"{result['keyword_score']:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with m3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.write("**Model Fit Confidence**")
                    st.write(f"{prob_good:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("View model-driven details"):
                    st.write("**JD summary:**")
                    st.write(result["jd"]["summary"])
                    st.write("**Resume summary:**")
                    st.write(result["resume"]["summary"])
                    st.write("**Extracted organisations (top 5):**")
                    st.write(result["resume"]["entities"].get("ORG", [])[:5])

    else:
        batch_button = st.button("🔍 Score all resumes (batch)", use_container_width=True)

        if batch_button:
            if not jd_text.strip():
                st.error("Please provide a job description.")
            elif not uploaded_files:
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
                        batch_results = evaluate_batch(jd_text, resumes_list, weights)

                    st.markdown("#### Result – Batch Candidate Ranking")

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

                    names = [
                        r.get("file_name", f"Candidate {i+1}")
                        for i, r in enumerate(batch_results)
                    ]
                    selected_name = st.selectbox(
                        "View details for candidate:",
                        names,
                        index=0,
                    )

                    selected = next(
                        (r for r in batch_results if r.get("file_name") == selected_name),
                        batch_results[0],
                    )

                    st.markdown("##### Candidate highlight & details")
                    st.markdown('<div class="score-card">', unsafe_allow_html=True)
                    st.write(f"**File:** {selected.get('file_name', 'N/A')}")
                    st.write(f"**Final score:** {selected['final_score']*100:.1f}%")
                    st.write(f"**Fit label:** {selected['fit']['label_name']}")
                    st.write(f"**Highlight:** {selected['highlight']}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("View model-driven details for selected candidate"):
                        st.write("**JD summary:**")
                        st.write(selected["jd"]["summary"])
                        st.write("**Resume summary:**")
                        st.write(selected["resume"]["summary"])
                        st.write("**Extracted organisations (top 5):**")
                        st.write(selected["resume"]["entities"].get("ORG", [])[:5])

# ============================================================
# END
# ============================================================
