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
# 0. BASIC SETUP & APPLE-ISH STYLING
# ============================================================

st.set_page_config(
    page_title="MatchAI: Candidate Suitability Screening",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     system-ui, -system-ui, "Helvetica Neue", Arial, sans-serif;
    }
    .main {
        background: radial-gradient(circle at top, #f9fafb 0, #f3f4f6 35%, #eef1f7 100%);
    }
    .block-container {
        max-width: 1200px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .card {
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        background-color: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
    }
    .section-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
        color: #111827;
    }
    .sub-label {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .score-header {
        font-size: 0.92rem;
        color: #6b7280;
        margin-bottom: 0.15rem;
    }
    .score-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.1rem;
    }
    .soft-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background-color: #e5edff;
        color: #1f3bb3;
        font-size: 0.78rem;
        margin-top: 0.1rem;
    }
    .light-hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 0.75rem 0 0.9rem 0;
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
            f"Could not load matchai_config.json at: {path}\n"
            f"Using fallback config. Error: {e}"
        )
        # Fallback config (placeholder models)
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
    ner_pipe = hf_pipeline("ner", model=cfg["ner_model"], grouped_entities=True)

    # Label mapping
    raw_map = cfg.get(
        "label_id2name",
        {"0": "No Fit", "1": "Potential Fit", "2": "Good Fit"},
    )
    label_id2name = {int(k): v for k, v in raw_map.items()}

    return clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name

config = load_matchai_config()
clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name = (
    load_models_and_pipelines(config)
)

DEFAULT_WEIGHTS = config.get(
    "weights",
    {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2},
)

# For clearing uploaders
if "upload_key_single" not in st.session_state:
    st.session_state.upload_key_single = 0
if "upload_key_batch" not in st.session_state:
    st.session_state.upload_key_batch = 0

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

    if label.lower().startswith("good") and score_pct >= 80:
        strength_bits.append("strong overall alignment with the job requirements.")
    elif "potential" in label.lower():
        strength_bits.append("partial match with scope to grow into the role.")
    else:
        strength_bits.append("limited alignment based on the current resume content.")

    if sim >= 0.8:
        strength_bits.append("high semantic similarity between the resume and job description.")
    elif sim >= 0.65:
        strength_bits.append("moderate semantic similarity indicating relevant experience.")

    if kw >= 0.6:
        strength_bits.append("good coverage of key skills and requirements in the job description.")
    elif kw >= 0.4:
        strength_bits.append("some important keywords overlap with the job description.")

    if orgs:
        strength_bits.append(
            f"experience with notable organisations (e.g. {', '.join(orgs[:3])})."
        )

    if pers:
        strength_bits.append(
            "clear personal profile and identity information captured in the resume."
        )

    if not strength_bits:
        return "No strong automated signals detected. This candidate may require manual review."

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
# 3. UI – INPUTS LEFT, OUTPUTS RIGHT
# ============================================================

st.title("🧩 MatchAI: Candidate Suitability Screening")
st.caption("Screen and prioritise candidates against a specific job description.")

left_col, right_col = st.columns([1.0, 1.25])

# ---------- LEFT: ALL INPUTS ----------
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1. Job description</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-label">Paste the job description to be used for this screening.</div>',
        unsafe_allow_html=True,
    )

    jd_text = st.text_area(
        "Job description",
        height=220,
        placeholder="Enter role responsibilities, required skills, and other key details...",
        key="jd_text",
    )

    st.markdown('<div class="section-title" style="margin-top:0.9rem;">2. Candidate resumes</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Evaluation mode",
        ["Single candidate", "Batch (multiple resumes)"],
        horizontal=True,
    )

    uploaded_file = None
    uploaded_files = []

    if mode == "Single candidate":
        st.markdown(
            '<div class="sub-label">Upload a resume file or paste the resume text directly.</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload resume (PDF / Word / TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            key=f"single_uploader_{st.session_state.upload_key_single}",
        )

        resume_text = st.text_area(
            "Candidate resume (text)",
            height=220,
            placeholder="Paste resume text here or use the upload above.",
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
                if not st.session_state.resume_text.strip():
                    st.session_state.resume_text = extracted
                st.info(f"Text extracted from file: {uploaded_file.name[:40]}")

    else:  # Batch mode
        st.markdown(
            '<div class="sub-label">Upload up to 30 resumes in PDF, Word, or TXT format. '
            '<b>Maximum 30 resumes per batch.</b></div>',
            unsafe_allow_html=True,
        )
        MAX_RESUMES = 30
        uploaded_files = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"batch_uploader_{st.session_state.upload_key_batch}",
        )
        if uploaded_files and len(uploaded_files) > MAX_RESUMES:
            st.error(f"Too many resumes uploaded. Maximum allowed is {MAX_RESUMES}.")
            st.stop()

    st.markdown("<hr class='light-hr' />", unsafe_allow_html=True)

    # Clear button
    if st.button("🧹 Clear all inputs"):
        st.session_state.jd_text = ""
        if "resume_text" in st.session_state:
            st.session_state.resume_text = ""
        st.session_state.upload_key_single += 1
        st.session_state.upload_key_batch += 1
        # Reset weights to default
        st.session_state["w_clf"] = DEFAULT_WEIGHTS["classifier"]
        st.session_state["w_sim"] = DEFAULT_WEIGHTS["similarity"]
        st.session_state["w_kw"] = DEFAULT_WEIGHTS["keywords"]
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: SCORING & RESULTS ----------
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">3. Scoring overview</div>', unsafe_allow_html=True)
    st.markdown(
        "The final suitability score blends three signals: "
        "**model confidence**, **semantic similarity**, and **keyword coverage**.",
    )

    with st.expander("How is this score calculated?"):
        st.write(
            "The score is a weighted combination of three components:\n\n"
            "- **P(Good Fit)** – probability that the model predicts a strong match.\n"
            "- **Similarity** – semantic similarity between JD summary and resume summary.\n"
            "- **Keyword coverage** – how many key terms from the JD appear in the resume summary.\n\n"
            "By default:\n"
            "> Final Score = 0.5 × P(Good Fit) + 0.3 × Similarity + 0.2 × Keyword Coverage\n\n"
            "You can adjust these weights below to reflect different HR priorities."
        )

    # Weights expander – close to the explanation
    with st.expander("Adjust scoring weights (optional)"):
        if "w_clf" not in st.session_state:
            st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
        if "w_sim" not in st.session_state:
            st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
        if "w_kw" not in st.session_state:
            st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]

        c1, c2, c3 = st.columns(3)
        with c1:
            w_clf = st.slider(
                "P(Good Fit)",
                0.0,
                1.0,
                float(st.session_state.w_clf),
                0.05,
            )
        with c2:
            w_sim = st.slider(
                "Similarity",
                0.0,
                1.0,
                float(st.session_state.w_sim),
                0.05,
            )
        with c3:
            w_kw = st.slider(
                "Keyword coverage",
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
            f"Normalised weights – P(Good Fit): {weights['classifier']:.2f} | "
            f"Similarity: {weights['similarity']:.2f} | Keywords: {weights['keywords']:.2f}"
        )

        if st.button("Reset to default weights"):
            st.session_state.w_clf = DEFAULT_WEIGHTS["classifier"]
            st.session_state.w_sim = DEFAULT_WEIGHTS["similarity"]
            st.session_state.w_kw = DEFAULT_WEIGHTS["keywords"]
            st.experimental_rerun()

    st.markdown("<hr class='light-hr' />", unsafe_allow_html=True)

    # Single big run button, near the top of the card
    run_label = (
        "▶ Run screening – single candidate"
        if mode == "Single candidate"
        else "▶ Run screening – batch"
    )
    run_button = st.button(run_label, type="primary")

    st.markdown("<hr class='light-hr' />", unsafe_allow_html=True)

    # ========================================================
    # 4. RUN EVALUATION & DISPLAY RESULTS
    # ========================================================

    if run_button:
        jd_val = st.session_state.jd_text.strip()

        if not jd_val:
            st.error("Please provide a job description.")
        else:
            if mode == "Single candidate":
                res_val = st.session_state.get("resume_text", "").strip()
                if not res_val:
                    st.error("Please provide resume text or upload a file.")
                else:
                    with st.spinner("Evaluating candidate..."):
                        result = evaluate_candidate(jd_val, res_val, weights)

                    score = result["final_score"]
                    score_pct = score * 100
                    label = result["fit"]["label_name"]
                    prob_good = result["prob_good_fit"]

                    st.markdown(
                        '<div class="section-title">4. Result – Single candidate</div>',
                        unsafe_allow_html=True,
                    )

                    top_left, top_right = st.columns([0.9, 1.2])
                    with top_left:
                        st.markdown(
                            '<div class="score-header">Final suitability score</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='score-value'>{score_pct:.1f}%</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<span class='soft-tag'>{map_priority(score)}</span>",
                            unsafe_allow_html=True,
                        )
                    with top_right:
                        st.write(f"**Predicted fit label:** {label}")
                        st.write(f"**P(Good Fit):** {prob_good:.2f}")
                        st.write(f"**Candidate highlight:** {result['highlight']}")

                    st.markdown("<hr class='light-hr' />", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.write("**Semantic similarity**")
                        st.write(f"{result['similarity']:.3f}")
                    with m2:
                        st.write("**Keyword coverage**")
                        st.write(f"{result['keyword_score']:.3f}")
                    with m3:
                        st.write("**Model fit confidence (P(Good Fit))**")
                        st.write(f"{prob_good:.3f}")

                    with st.expander("Model-driven summaries"):
                        st.write("**JD summary**")
                        st.write(result["jd"]["summary"])
                        st.write("**Resume summary**")
                        st.write(result["resume"]["summary"])
                        st.write("**Extracted organisations (top 5)**")
                        st.write(result["resume"]["entities"].get("ORG", [])[:5])

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
                            batch_results = evaluate_batch(jd_val, resumes_list, weights)

                        st.markdown(
                            '<div class="section-title">4. Result – Batch candidate ranking</div>',
                            unsafe_allow_html=True,
                        )

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

                        st.markdown("#### Candidate details")

                        idx_options = list(range(len(batch_results)))
                        selected_idx = st.selectbox(
                            "Select a candidate to inspect",
                            options=idx_options,
                            format_func=lambda i: f"{i+1}. {batch_results[i].get('file_name', f'Candidate {i+1}')}",
                        )
                        selected = batch_results[selected_idx]

                        st.write(
                            f"**File:** {selected.get('file_name', 'N/A')}  "
                            f"| **Final score:** {selected['final_score']*100:.1f}%  "
                            f"| **Fit label:** {selected['fit']['label_name']}"
                        )
                        st.write(f"**Highlight:** {selected['highlight']}")

                        with st.expander("Model-driven summaries for selected candidate"):
                            st.write("**JD summary**")
                            st.write(selected["jd"]["summary"])
                            st.write("**Resume summary**")
                            st.write(selected["resume"]["summary"])
                            st.write("**Extracted organisations (top 5)**")
                            st.write(selected["resume"]["entities"].get("ORG", [])[:5])

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# END
# ============================================================
