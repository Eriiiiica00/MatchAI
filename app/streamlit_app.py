import json
import io
import os
import re
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
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     system-ui, sans-serif !important;
    }

    .main {
        background-color: #f5f5f7;
    }

    .matchai-card {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e2e3e7;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03);
        margin-bottom: 0.8rem;
    }

    .matchai-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.3rem;
    }

    .score-card {
        padding: 1rem 1.3rem;
        border-radius: 18px;
        background-color: #ffffff;
        border: 1px solid #e2e3e7;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03);
        margin-bottom: 0.6rem;
    }

    .metric-box {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background-color: #f2f2f7;
        border: 1px solid #e0e0e5;
        font-size: 0.9rem;
    }

    .section-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.4rem;
    }

    .small-muted {
        font-size: 0.8rem;
        color: #6e6e73;
    }

    /* Make file uploader "Browse files" look like our other soft buttons */
    .stFileUploader > label {
        background-color: #f5f5f7;
        border-radius: 999px;
        border: 1px solid #d1d1d6;
        padding: 0.35rem 0.9rem;
        color: #1d1d1f;
        font-weight: 500;
    }
    .stFileUploader > label:hover {
        background-color: #e5e5ea;
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
            f"Using fallback model IDs for testing only. Error: {e}"
        )
        # Fallback config (for testing only)
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
    raw_map = cfg.get("label_id2name", {"0": "No Fit", "1": "Potential Fit", "2": "Good Fit"})
    label_id2name = {int(k): v for k, v in raw_map.items()}

    return clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name

config = load_matchai_config()
clf_tokenizer, clf_model, summarizer, sim_model, ner_pipe, label_id2name = load_models_and_pipelines(config)

DEFAULT_WEIGHTS = config.get(
    "weights",
    {"classifier": 0.5, "similarity": 0.3, "keywords": 0.2}
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
    ents = ner_pipe(text[:1200])
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

def generate_candidate_highlights(result_dict: dict) -> list[str]:
    """
    Generate natural, recruiter-style highlights:
    - Overall impression
    - 1–2 strengths
    - 0–2 watch-outs
    """
    highlights: list[str] = []

    label = result_dict["fit"]["label_name"]
    score = float(result_dict.get("final_score", 0.0))
    score_pct = score * 100
    prob = float(result_dict.get("prob_good_fit", 0.0))
    sim = float(result_dict.get("similarity", 0.0))
    kw = float(result_dict.get("keyword_score", 0.0))

    jd_summary = result_dict["jd"].get("summary", "") or ""
    res_summary = result_dict["resume"].get("summary", "") or ""
    res_raw = result_dict["resume"].get("raw", "") or res_summary

    entities = result_dict["resume"].get("entities", {})
    orgs = entities.get("ORG", [])

    # 1. Overall impression
    if score >= 0.8:
        overall = (
            f"Looks like a strong match for this role "
            f"(overall score around {score_pct:.0f}% and model label **{label}**)."
        )
    elif score >= 0.6:
        overall = (
            f"Reasonable match with some potential "
            f"(overall score around {score_pct:.0f}%, model label **{label}**)."
        )
    else:
        overall = (
            f"Currently reads as a weaker match "
            f"(overall score around {score_pct:.0f}%, model label **{label}**)."
        )
    highlights.append(overall)

    # 2. Strengths
    strength_bits: list[str] = []

    if prob >= 0.8:
        strength_bits.append("The model is very comfortable that this profile fits the role.")
    elif prob >= 0.6:
        strength_bits.append("The model sees a fair amount of overlap with what the role is asking for.")

    if sim >= 0.80:
        strength_bits.append("The responsibilities in the CV line up closely with the job description.")
    elif sim >= 0.65:
        strength_bits.append("There is a decent amount of overlap between the CV and the job description.")

    jd_tokens = {
        w.lower().strip(",.;:()")
        for w in jd_summary.split()
        if len(w) > 4
    }
    res_tokens = {
        w.lower().strip(",.;:()")
        for w in res_summary.split()
        if len(w) > 4
    }
    matched = list(jd_tokens.intersection(res_tokens))
    matched = [m for m in matched if not m.isdigit()]

    if matched:
        matched_sorted = sorted(matched)
        top = matched_sorted[:4]
        strengths_themes = ", ".join(top)
        strength_bits.append(
            f"The CV talks about similar themes as the JD (for example: {strengths_themes})."
        )
    elif kw >= 0.5:
        strength_bits.append(
            "Even where wording differs, the CV still covers several of the main ideas in the JD."
        )

    years_matches = re.findall(
        r"(\d+)\+?\s+(?:years?|yrs?)", res_raw, flags=re.IGNORECASE
    )
    max_years = None
    if years_matches:
        try:
            max_years = max(int(y) for y in years_matches)
        except ValueError:
            max_years = None

    if max_years is not None:
        if max_years >= 10:
            strength_bits.append(f"Experience level appears to be around {max_years}+ years.")
        elif max_years >= 5:
            strength_bits.append(f"Roughly {max_years} years of experience mentioned in the CV.")

    seniority_terms = [
        "intern", "junior", "associate", "senior", "lead",
        "manager", "director", "head", "vice president", "vp"
    ]
    res_lower = res_raw.lower()
    found_seniority = [s for s in seniority_terms if s in res_lower]
    if found_seniority:
        uniq_sen = list(dict.fromkeys(found_seniority))
        strength_bits.append(
            "Role history suggests a level around: " + ", ".join(uniq_sen[:3]) + "."
        )

    cleaned_orgs = []
    if orgs:
        unique_orgs = list(dict.fromkeys(orgs))
        for o in unique_orgs:
            main = o.split(",")[0].strip()
            if len(main) <= 3:
                continue
            if not any(c.isalpha() for c in main):
                continue
            cleaned_orgs.append(main)

        if cleaned_orgs:
            strength_bits.append(
                "Has experience with organisations such as "
                + ", ".join(cleaned_orgs[:3]) + "."
            )

    if strength_bits:
        highlights.append("Strengths: " + " ".join(strength_bits))

    # 3. Watch-outs
    concern_bits: list[str] = []

    if score < 0.6:
        concern_bits.append("Overall score is below the typical interview-priority range.")
    if prob < 0.5:
        concern_bits.append("The model does not see a very strong fit signal yet.")
    if sim < 0.6:
        concern_bits.append("The CV does not strongly mirror the responsibilities in the JD.")
    if kw < 0.4:
        concern_bits.append("Key phrases from the JD do not appear often in the CV.")
    if max_years is not None and max_years < 3:
        concern_bits.append("Experience level looks relatively junior.")

    if concern_bits:
        highlights.append("Watch-outs: " + " ".join(concern_bits[:2]))

    if not highlights:
        highlights.append(
            "The automated signals are quite neutral – this CV is worth a quick manual look."
        )

    return highlights

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
    result["highlights"] = generate_candidate_highlights(result)
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
# 3. UI LAYOUT (APPLE-ISH, CLEAN)
# ============================================================

st.markdown("## 🔍 MatchAI: Candidate Suitability Screening")

st.markdown(
    '<p class="small-muted">Paste a job description, upload one or more resumes, and let MatchAI help you prioritise candidates.</p>',
    unsafe_allow_html=True,
)

# Main layout: inputs left, outputs right
col_left, col_right = st.columns([1.05, 1.15])

# ---------- LEFT: INPUT AREA ----------
with col_left:
    st.markdown('<div class="matchai-card">', unsafe_allow_html=True)
    st.markdown('<div class="matchai-title">Job Description</div>', unsafe_allow_html=True)

    jd_text = st.text_area(
        label="",
        key="jd_text",
        height=220,
        placeholder="Enter role responsibilities, required skills, and qualifications...",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="matchai-card">', unsafe_allow_html=True)
    st.markdown('<div class="matchai-title">Candidate Resumes</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Evaluation mode",
        ["Single candidate", "Batch (multiple resumes)"],
        horizontal=True,
        label_visibility="collapsed",
        key="mode_radio",
    )

    uploaded_file = None
    uploaded_files = None

    if mode == "Single candidate":
        uploaded_file = st.file_uploader(
            "Upload resume (PDF / Word / TXT, optional)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False,
            key="single_resume_file",
        )

        resume_text = st.text_area(
            "Resume (text)",
            key="resume_text",
            height=220,
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
                current_text = st.session_state.get("resume_text", "")
                if not current_text.strip():
                    st.session_state["resume_text"] = extracted
                st.info(f"Text extracted from file: {uploaded_file.name[:40]}")

    else:
        MAX_RESUMES = 30
        st.caption(f"Upload up to **{MAX_RESUMES}** resumes per batch.")
        uploaded_files = st.file_uploader(
            "Upload multiple resumes (PDF / Word / TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="batch_resume_files",
        )
        if uploaded_files and len(uploaded_files) > MAX_RESUMES:
            st.error(f"Too many resumes uploaded. Maximum allowed is {MAX_RESUMES}.")
            st.stop()

    # Clear button – fully resets inputs and files
    if st.button("Clear inputs"):
        st.session_state["jd_text"] = ""
        st.session_state["resume_text"] = ""
        st.session_state["single_resume_file"] = None
        st.session_state["batch_resume_files"] = []
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Advanced weights (close to scoring explanation)
    st.markdown('<div class="matchai-card">', unsafe_allow_html=True)
    st.markdown('<div class="matchai-title">Scoring weights</div>', unsafe_allow_html=True)

    st.caption(
        "ℹ️ The final score blends model confidence, semantic similarity and "
        "keyword coverage. Use the sliders to rebalance what matters most."
    )

    # Store weights in percentage form in session_state
    if "w_clf_pct" not in st.session_state:
        st.session_state["w_clf_pct"] = int(DEFAULT_WEIGHTS["classifier"] * 100)
    if "w_sim_pct" not in st.session_state:
        st.session_state["w_sim_pct"] = int(DEFAULT_WEIGHTS["similarity"] * 100)
    if "w_kw_pct" not in st.session_state:
        st.session_state["w_kw_pct"] = int(DEFAULT_WEIGHTS["keywords"] * 100)

    c1, c2, c3 = st.columns(3)
    with c1:
        w_clf_pct = st.slider(
            "P(Good Fit) %",
            0,
            100,
            int(st.session_state["w_clf_pct"]),
            5,
        )
    with c2:
        w_sim_pct = st.slider(
            "Similarity %",
            0,
            100,
            int(st.session_state["w_sim_pct"]),
            5,
        )
    with c3:
        w_kw_pct = st.slider(
            "Keywords %",
            0,
            100,
            int(st.session_state["w_kw_pct"]),
            5,
        )

    # Update session_state with slider results
    st.session_state["w_clf_pct"] = w_clf_pct
    st.session_state["w_sim_pct"] = w_sim_pct
    st.session_state["w_kw_pct"] = w_kw_pct

    total_pct = w_clf_pct + w_sim_pct + w_kw_pct

    if total_pct == 0:
        weights = {
            "classifier": DEFAULT_WEIGHTS["classifier"],
            "similarity": DEFAULT_WEIGHTS["similarity"],
            "keywords": DEFAULT_WEIGHTS["keywords"],
        }
        display_clf = DEFAULT_WEIGHTS["classifier"] * 100
        display_sim = DEFAULT_WEIGHTS["similarity"] * 100
        display_kw = DEFAULT_WEIGHTS["keywords"] * 100
    else:
        weights = {
            "classifier": w_clf_pct / total_pct,
            "similarity": w_sim_pct / total_pct,
            "keywords": w_kw_pct / total_pct,
        }
        display_clf = weights["classifier"] * 100
        display_sim = weights["similarity"] * 100
        display_kw = weights["keywords"] * 100

    st.caption(
        f"Effective weights (sum to 100%) → P(Good Fit): {display_clf:.0f}%, "
        f"Similarity: {display_sim:.0f}%, Keywords: {display_kw:.0f}%"
    )

    if st.button("Reset to default weights"):
        st.session_state["w_clf_pct"] = int(DEFAULT_WEIGHTS["classifier"] * 100)
        st.session_state["w_sim_pct"] = int(DEFAULT_WEIGHTS["similarity"] * 100)
        st.session_state["w_kw_pct"] = int(DEFAULT_WEIGHTS["keywords"] * 100)
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: OUTPUT AREA ----------
with col_right:
    st.markdown('<div class="matchai-card">', unsafe_allow_html=True)
    st.markdown('<div class="matchai-title">Run evaluation</div>', unsafe_allow_html=True)

    if st.session_state.get("mode_radio", "Single candidate") == "Single candidate":
        score_button = st.button("🔍 Score single candidate", use_container_width=True)
    else:
        score_button = st.button("🔍 Score multiple resumes (batch)", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="matchai-card">', unsafe_allow_html=True)

    mode_current = st.session_state.get("mode_radio", "Single candidate")

    if mode_current == "Single candidate":
        if score_button:
            jd_val = st.session_state.get("jd_text", "")
            res_val = st.session_state.get("resume_text", "")

            if not jd_val.strip():
                st.error("Please provide a job description.")
            elif not res_val.strip():
                st.error("Please provide resume text or upload a file.")
            else:
                with st.spinner("Evaluating candidate..."):
                    result = evaluate_candidate(jd_val, res_val, weights)

                score = result["final_score"]
                score_pct = score * 100
                label = result["fit"]["label_name"]
                prob_good = result["prob_good_fit"]

                st.markdown('<div class="score-card">', unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-title'>Final suitability score</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size: 2.1rem; margin: 0 0 0.4rem;'><b>{score_pct:.1f}%</b></p>",
                    unsafe_allow_html=True,
                )
                st.write(f"**Predicted label:** {label}")
                st.write(f"**P(Good Fit):** {prob_good:.2f}")
                st.write(f"**{map_priority(score)}**")
                st.markdown("</div>", unsafe_allow_html=True)

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
                    st.write("**Model confidence**")
                    st.write(f"{prob_good:.3f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("**Candidate highlights:**")
                for h in result["highlights"]:
                    st.write(f"- {h}")

                with st.expander("Model-driven details"):
                    st.write("**JD summary:**")
                    st.write(result["jd"]["summary"])
                    st.write("**Resume summary:**")
                    st.write(result["resume"]["summary"])
                    st.write("**Extracted organisations (top 5):**")
                    st.write(result["resume"]["entities"].get("ORG", [])[:5])

        else:
            st.write("Run an evaluation to see results here.")

    else:
        if score_button:
            jd_val = st.session_state.get("jd_text", "")

            if not jd_val.strip():
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
                        batch_results = evaluate_batch(jd_val, resumes_list, weights)

                    st.markdown("### Batch candidate ranking")

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

                    # Show details + highlights for any selected candidate
                    st.markdown("#### Candidate details & highlights")

                    names = [
                        f"{row['Rank']}. {row['File']}"
                        for row in table_rows
                    ]
                    selected = st.selectbox(
                        "Select a candidate to view details",
                        options=names,
                    )
                    selected_rank = int(selected.split(".")[0])
                    chosen = batch_results[selected_rank - 1]

                    st.write(f"**File:** {chosen.get('file_name', 'N/A')}")
                    st.write(f"**Final score:** {chosen['final_score']*100:.1f}%")
                    st.write(f"**Fit label:** {chosen['fit']['label_name']}")

                    st.write("**Candidate highlights:**")
                    for h in chosen["highlights"]:
                        st.write(f"- {h}")

                    with st.expander("Model-driven details for this candidate"):
                        st.write("**JD summary:**")
                        st.write(chosen["jd"]["summary"])
                        st.write("**Resume summary:**")
                        st.write(chosen["resume"]["summary"])
                        st.write("**Extracted organisations (top 5):**")
                        st.write(chosen["resume"]["entities"].get("ORG", [])[:5])

        else:
            st.write("Run a batch evaluation to see ranking and highlights here.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# END
# ============================================================
