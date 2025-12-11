import io
import textwrap
from typing import Dict, Any, Tuple, List

import streamlit as st
import numpy as np

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

class AppConfig:
    def __init__(self):
        # Text similarity / embeddings model
        self.sim_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Summarizer model
        self.summarizer_model_name = "sshleifer/distilbart-cnn-12-6"

        # NER model (for skills / entities highlighting)
        self.ner_model_name = "dslim/bert-base-NER"

config = AppConfig()

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_models_and_pipelines(cfg: AppConfig):
    sim_model = SentenceTransformer(cfg.sim_model_name)
    summarizer = pipeline("summarization", model=cfg.summarizer_model_name)
    ner_pipe = pipeline("ner", model=cfg.ner_model_name, grouped_entities=True)
    return sim_model, summarizer, ner_pipe

sim_model, summarizer, ner_pipe = load_models_and_pipelines(config)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def read_uploaded_file(uploaded_file) -> str:
    """
    Read content from an uploaded file (txt, pdf, docx, etc.).
    Currently supports txt directly; for pdf/docx, you can extend later.
    """
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    # Fallback: try to read as text
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def compute_similarity_score(resume_text: str, jd_text: str) -> float:
    """
    Use sentence-transformers to compute cosine similarity between
    resume and job description, return percentage [0, 100].
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    emb_resume = sim_model.encode(resume_text, convert_to_tensor=True)
    emb_jd = sim_model.encode(jd_text, convert_to_tensor=True)
    cos_sim = util.cos_sim(emb_resume, emb_jd).item()
    # Map from [-1, 1] roughly to [0, 100] (in practice it will be [0, 1])
    score = max(0.0, min(1.0, (cos_sim + 1) / 2)) * 100
    return float(score)

def extract_entities(text: str) -> List[str]:
    if not text.strip():
        return []
    ents = ner_pipe(text)
    out = []
    for e in ents:
        val = e.get("word", "").strip()
        if val and val not in out:
            out.append(val)
    return out

def compute_skills_alignment_score(resume_text: str, jd_text: str) -> float:
    """
    Very rough proxy: use NER to identify entities/skills in both texts,
    and compute overlap ratio.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    resume_ents = set(extract_entities(resume_text))
    jd_ents = set(extract_entities(jd_text))

    if not jd_ents:
        return 50.0  # neutral if JD has no entities detected

    overlap = resume_ents.intersection(jd_ents)
    ratio = len(overlap) / len(jd_ents)
    return float(max(0.0, min(1.0, ratio)) * 100)

def compute_experience_alignment_score(resume_text: str, jd_text: str) -> float:
    """
    For now, reuse similarity (particularly on experience-like segments).
    You can refine later (e.g. look for 'experience', 'years', etc.).
    """
    return compute_similarity_score(resume_text, jd_text) * 0.9  # slightly down-weighted

def generate_candidate_highlight(resume_text: str, jd_text: str) -> str:
    """
    Generate a short, natural-sounding highlight paragraph about the candidate,
    focusing on difference vs others & real strengths.
    """
    base_prompt = (
        "You are an assistant summarizing a candidate's strengths for a recruiter.\n\n"
        "Job description:\n"
        f"{jd_text[:2000]}\n\n"
        "Candidate resume:\n"
        f"{resume_text[:4000]}\n\n"
        "Write 3–5 sentences highlighting:\n"
        "- What stands out about this candidate vs a typical applicant\n"
        "- The most relevant experience and skills for THIS job\n"
        "- Any clear red flags or gaps that the recruiter should note\n\n"
        "Use natural language, not bullet points. Be concise and specific."
    )

    # Use summarizer to compress this "prompt" (hacky, but works to keep it short).
    try:
        out = summarizer(
            base_prompt,
            max_length=180,
            min_length=90,
            do_sample=False,
        )
        text = out[0]["summary_text"].strip()
        return text
    except Exception:
        # Fallback simple heuristic
        return (
            "This candidate appears to have several relevant experiences and skills for the role. "
            "They demonstrate exposure to key responsibilities in the job description and have a track record "
            "of delivering results in related areas. There may still be gaps compared with an ideal profile, "
            "so the recruiter should review their experience depth and recency in more detail."
        )

def compute_weighted_score(
    similarity_score: float,
    skills_score: float,
    experience_score: float,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Weighted overall score based on user-selected weights.
    weights keys: 'similarity', 'skills', 'experience'
    """
    w_sim = weights.get("similarity", 40.0)
    w_skill = weights.get("skills", 35.0)
    w_exp = weights.get("experience", 25.0)

    total_weight = w_sim + w_skill + w_exp
    if total_weight == 0:
        # Avoid division by zero; use equal weights if user sets everything to 0
        w_sim = w_skill = w_exp = 1
        total_weight = 3

    # Normalize to sum to 1 internally
    sim_norm = w_sim / total_weight
    skill_norm = w_skill / total_weight
    exp_norm = w_exp / total_weight

    overall = (
        similarity_score * sim_norm
        + skills_score * skill_norm
        + experience_score * exp_norm
    )

    contributions = {
        "similarity": similarity_score * sim_norm,
        "skills": skills_score * skill_norm,
        "experience": experience_score * exp_norm,
    }

    return float(overall), contributions

def init_session_state():
    defaults = {
        "resume_text": "",
        "jd_text": "",
        "similarity_score": None,
        "skills_score": None,
        "experience_score": None,
        "overall_score": None,
        "score_contributions": None,
        "highlight": None,
        "weights": {
            "similarity": 40.0,
            "skills": 35.0,
            "experience": 25.0,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="MatchAI – Resume & Job Description Matching",
        layout="centered",
    )

    init_session_state()

    st.title("MatchAI – Resume & Job Description Matching")

    st.markdown(
        """
        Upload a **resume** and a **job description** to get:
        - A single **overall match score** (0–100)
        - Breakdown by **similarity**, **skills alignment**, and **experience**
        - A concise **candidate highlight** for recruiters
        """
    )

    # -----------------------------------------------------
    # INPUT SECTION
    # -----------------------------------------------------
    st.header("1. Inputs")

    # --- File uploaders ---
    resume_file = st.file_uploader("Upload Resume (txt preferred)", type=["txt", "pdf", "docx"], key="resume_file")
    jd_file = st.file_uploader("Upload Job Description (txt preferred)", type=["txt", "pdf", "docx"], key="jd_file")

    st.markdown("Or paste text directly below:")

    col_text = st.container()
    with col_text:
        st.session_state.resume_text = st.text_area(
            "Resume text",
            value=st.session_state.resume_text,
            height=180,
            key="resume_text_area",
        )
        st.session_state.jd_text = st.text_area(
            "Job description text",
            value=st.session_state.jd_text,
            height=180,
            key="jd_text_area",
        )

    # If files are uploaded, they override text areas
    if resume_file is not None:
        st.session_state.resume_text = read_uploaded_file(resume_file)
    if jd_file is not None:
        st.session_state.jd_text = read_uploaded_file(jd_file)

    # -----------------------------------------------------
    # WEIGHTS
    # -----------------------------------------------------
    st.header("2. Adjust scoring weights (optional)")

    st.markdown(
        "Weights determine how much each component contributes to the final score. "
        "Total **must not exceed 100%**."
    )

    w_sim = st.slider(
        "Weight – Similarity (resume vs JD)",
        min_value=0,
        max_value=100,
        value=int(st.session_state.weights["similarity"]),
        step=5,
        key="w_sim_slider",
    )
    w_skill = st.slider(
        "Weight – Skills alignment",
        min_value=0,
        max_value=100,
        value=int(st.session_state.weights["skills"]),
        step=5,
        key="w_skill_slider",
    )
    w_exp = st.slider(
        "Weight – Experience alignment",
        min_value=0,
        max_value=100,
        value=int(st.session_state.weights["experience"]),
        step=5,
        key="w_exp_slider",
    )

    total_weight = w_sim + w_skill + w_exp
    st.session_state.weights = {
        "similarity": float(w_sim),
        "skills": float(w_skill),
        "experience": float(w_exp),
    }

    if total_weight > 100:
        st.error(
            f"Total weight is **{total_weight}%** – it must be **≤ 100%**. "
            "Please reduce one or more sliders."
        )

    elif total_weight < 100:
        st.info(
            f"Total weight is currently **{total_weight}%**. "
            "The model will internally normalise them to 100%, "
            "but you still have **{100 - total_weight}%** unallocated if you want full usage."
        )
    else:
        st.success("Total weight = **100%** ✅")

    with st.expander("How is the score calculated?"):
        st.markdown(
            """
            **Components:**
            - **Similarity**: semantic similarity between the full resume and job description.
            - **Skills alignment**: overlap between named entities/skills detected in the resume vs the job description.
            - **Experience alignment**: experience-flavoured similarity signal derived from the same semantic model.

            Scores are combined as:

            `overall = w_similarity * similarity + w_skills * skills + w_experience * experience`

            where the weights `w_*` are your sliders, normalised to sum to 1.
            """
        )

    # -----------------------------------------------------
    # ACTION BUTTONS
    # -----------------------------------------------------
    st.header("3. Actions")

    col_buttons = st.columns([1, 1, 1])
    with col_buttons[0]:
        run_btn = st.button("🔎 Run matching")
    with col_buttons[1]:
        clear_inputs_btn = st.button("🧹 Clear inputs")
    with col_buttons[2]:
        clear_outputs_btn = st.button("🧼 Clear outputs")

    # --- Clear inputs ---
    if clear_inputs_btn:
        st.session_state.resume_text = ""
        st.session_state.jd_text = ""
        st.session_state["resume_file"] = None
        st.session_state["jd_file"] = None
        # Also clear outputs for consistency
        st.session_state.similarity_score = None
        st.session_state.skills_score = None
        st.session_state.experience_score = None
        st.session_state.overall_score = None
        st.session_state.score_contributions = None
        st.session_state.highlight = None
        st.rerun()

    # --- Clear outputs only ---
    if clear_outputs_btn:
        st.session_state.similarity_score = None
        st.session_state.skills_score = None
        st.session_state.experience_score = None
        st.session_state.overall_score = None
        st.session_state.score_contributions = None
        st.session_state.highlight = None
        st.rerun()

    # -----------------------------------------------------
    # RUN MATCHING
    # -----------------------------------------------------
    if run_btn:
        if not st.session_state.resume_text.strip() or not st.session_state.jd_text.strip():
            st.warning("Please provide both resume and job description (via upload or text).")
        elif total_weight > 100:
            st.error("Please fix the weights so total is ≤ 100% before running.")
        else:
            with st.spinner("Calculating match score..."):
                sim_score = compute_similarity_score(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                )
                skills_score = compute_skills_alignment_score(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                )
                exp_score = compute_experience_alignment_score(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                )

                overall, contrib = compute_weighted_score(
                    sim_score,
                    skills_score,
                    exp_score,
                    st.session_state.weights,
                )

                highlight = generate_candidate_highlight(
                    st.session_state.resume_text,
                    st.session_state.jd_text,
                )

                st.session_state.similarity_score = sim_score
                st.session_state.skills_score = skills_score
                st.session_state.experience_score = exp_score
                st.session_state.overall_score = overall
                st.session_state.score_contributions = contrib
                st.session_state.highlight = highlight

            st.success("Matching completed!")

    # -----------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------
    st.header("4. Results")

    if st.session_state.overall_score is not None:
        st.subheader(f"Overall match score: **{st.session_state.overall_score:.1f} / 100**")

        col_scores = st.columns(3)
        with col_scores[0]:
            st.metric("Similarity", f"{st.session_state.similarity_score:.1f}")
        with col_scores[1]:
            st.metric("Skills alignment", f"{st.session_state.skills_score:.1f}")
        with col_scores[2]:
            st.metric("Experience alignment", f"{st.session_state.experience_score:.1f}")

        if st.session_state.score_contributions:
            st.markdown("**Weighted contributions (after normalisation):**")
            st.write(
                {
                    "Similarity": round(st.session_state.score_contributions["similarity"], 2),
                    "Skills": round(st.session_state.score_contributions["skills"], 2),
                    "Experience": round(st.session_state.score_contributions["experience"], 2),
                }
            )

        st.subheader("Candidate highlight")
        st.write(st.session_state.highlight)

    else:
        st.info("Run the matching to see scores and highlight.")

    st.caption("Tip: You can tweak the weights, rerun, and compare how the overall score changes.")

if __name__ == "__main__":
    main()
