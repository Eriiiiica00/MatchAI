import json
import io
import os
import re
import string
from datetime import datetime

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

    .preview-box {
        border-radius: 16px;
        background-color: #fbfbfd;
        border: 1px solid var(--border-subtle);
        padding: 0.85rem 1rem;
        margin-top: 0.5rem;
        white-space: pre-wrap;
        line-height: 1.45;
        font-size: 0.95rem;
        color: #1c1c1e;
    }

    .snap-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem 1.2rem;
        margin-top: 0.4rem;
    }

    .snap-item {
        padding: 0.65rem 0.8rem;
        border-radius: 14px;
        background: #fafafa;
        border: 1px solid var(--accent-soft);
    }
    .snap-k { color: var(--text-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }
    .snap-v { color: #111; font-size: 0.98rem; margin-top: 0.15rem; }

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(1)
except Exception:
    pass

# uploader reset nonce
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0

# action state (for disable + label change)
if "pending_action" not in st.session_state:
    st.session_state["pending_action"] = None  # "single" | "batch" | None
if "is_busy" not in st.session_state:
    st.session_state["is_busy"] = False


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
            return json.load(f)
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

    try:
        clf_id = cfg["fine_tuned_model_id"]
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
# 2. TEXT HELPERS + SCORING HELPERS
# ============================================================

STOPWORDS = {
    "the","and","for","with","from","that","this","you","your","are","our","will","have",
    "role","work","team","teams","using","use","used","able","must","should","within",
    "include","including","responsibilities","requirements","years","year","experience",
    "skills","skill","ability","strong","good","excellent","preferred","plus","etc"
}

SECTION_HEADINGS = [
    "SUMMARY", "PROFILE", "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT",
    "SKILLS", "TECHNICAL SKILLS", "PROJECTS", "CERTIFICATIONS", "LANGUAGES", "PUBLICATIONS"
]

BULLET_CHARS = ("•", "·", "●", "▪", "–", "-", "*", "◦", "o", "", "")

def _clean_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\s+", " ", (s or ""))).strip()

def _clean_bullet_prefix(line: str) -> str:
    s = (line or "").strip()
    # remove repeated bullet chars
    while s.startswith(BULLET_CHARS):
        s = s[1:].lstrip()
    # remove weird squares/boxes artifacts sometimes produced by PDF extraction
    s = s.replace("\u25a1", "").replace("\uf0b7", "").replace("\uf0a7", "").strip()
    return s

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
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
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def summarize_text(text: str, max_len: int = 150) -> str:
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

def _top_keywords_from_text(text: str, k: int = 18) -> list[str]:
    toks = _tokens(text)
    freq: dict[str, int] = {}
    for w in toks:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:k]]

def keyword_match_score(jd_keywords, resume_text_or_summary: str) -> float:
    if not jd_keywords:
        return 0.0
    resume_words = set(_tokens(resume_text_or_summary))
    hits = sum(1 for kw in jd_keywords if kw in resume_words)
    return hits / len(jd_keywords)

def map_priority(score: float) -> str:
    if score >= 0.8:
        return "Interview priority: HIGH"
    elif score >= 0.6:
        return "Interview priority: MEDIUM"
    else:
        return "Interview priority: LOW"

# ---------- Snapshot parsing (header/contact + latest edu/role + languages) ----------

def _extract_email(text: str) -> str | None:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    return m.group(0) if m else None

def _extract_phone(text: str) -> str | None:
    # permissive phone matcher
    m = re.search(r"(\+?\d[\d\s().-]{7,}\d)", text or "")
    if not m:
        return None
    ph = _clean_spaces(m.group(1))
    # avoid catching long numeric IDs
    if len(re.sub(r"\D", "", ph)) < 8:
        return None
    return ph

def _extract_linkedin(text: str) -> str | None:
    m = re.search(r"(https?://(www\.)?linkedin\.com/[^\s]+)", text or "", flags=re.I)
    return m.group(1) if m else None

def _extract_name(header_lines: list[str]) -> str | None:
    if not header_lines:
        return None
    # try first non-empty line, strip contact tokens
    for ln in header_lines[:3]:
        s = _clean_bullet_prefix(ln)
        s = re.sub(r"\b(email|phone|mobile|tel)\b.*", "", s, flags=re.I).strip()
        if not s:
            continue
        # If it contains @ it's not a name
        if "@" in s:
            continue
        # reduce overly long lines
        if len(s) > 60:
            s = s[:60].strip()
        # accept if mostly letters/spaces and at least 2 tokens
        toks = re.findall(r"[A-Za-z]+", s)
        if len(toks) >= 2:
            return " ".join(toks[:5]).title()
    return None

def _split_lines(text: str) -> list[str]:
    lines = [l.rstrip() for l in re.split(r"[\n\r]+", text or "") if l.strip()]
    return lines

def _find_header_block(lines: list[str]) -> tuple[list[str], int]:
    """
    Returns (header_lines, start_index_of_body)
    Header ends when a known SECTION heading appears, or after ~12 lines.
    """
    body_start = 0
    header = []
    for i, ln in enumerate(lines[:20]):
        up = re.sub(r"[^A-Za-z ]", "", ln).strip().upper()
        if any(h in up for h in SECTION_HEADINGS):
            body_start = i
            break
        header.append(ln)
        body_start = i + 1
        if body_start >= 12:
            break
    return header, body_start

def _extract_languages(text: str) -> str | None:
    # look for "Languages" section
    m = re.search(r"(LANGUAGES|Language)\s*[:\n]\s*(.+)", text or "", flags=re.I)
    if m:
        tail = m.group(2)
        tail = tail.split("\n")[0]
        tail = _clean_spaces(tail)
        tail = re.sub(r"[,;]\s*$", "", tail)
        if 2 <= len(tail) <= 80:
            return tail
    # heuristic keywords
    langs = []
    for lang in ["English", "Cantonese", "Mandarin", "Chinese", "French", "Japanese", "Korean"]:
        if re.search(rf"\b{lang}\b", text or "", flags=re.I):
            langs.append(lang)
    langs = list(dict.fromkeys(langs))
    if langs:
        return ", ".join(langs[:5])
    return None

def _extract_latest_education(text: str) -> str | None:
    # find education section lines
    lines = _split_lines(text)
    edu_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"\bEDUCATION\b", ln, flags=re.I):
            edu_idx = i
            break
    if edu_idx is None:
        return None
    chunk = lines[edu_idx: edu_idx + 14]
    # pick first meaningful line after heading
    for ln in chunk[1:]:
        s = _clean_bullet_prefix(_clean_spaces(ln))
        if len(s) < 8:
            continue
        # grab a year if exists
        yr = re.search(r"\b(19|20)\d{2}\b", s)
        if yr:
            return f"{s} ({yr.group(0)})"
        # or "Expected 2026"
        exp = re.search(r"(Expected)\s*((19|20)\d{2})", s, flags=re.I)
        if exp:
            return f"{s}"
        return s[:90]
    return None

def _extract_latest_role(text: str) -> str | None:
    lines = _split_lines(text)
    # find experience section
    exp_idx = None
    for i, ln in enumerate(lines):
        if re.search(r"\b(EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT)\b", ln, flags=re.I):
            exp_idx = i
            break
    if exp_idx is None:
        # fallback: just search for "Present"
        m = re.search(r"(.{0,80})(20\d{2}).{0,20}(Present|Current)", text or "", flags=re.I)
        if m:
            return _clean_spaces(m.group(0))[:90]
        return None

    chunk = lines[exp_idx: exp_idx + 20]
    # select first line that looks like a role/company line
    for ln in chunk[1:]:
        s = _clean_bullet_prefix(_clean_spaces(ln))
        if len(s) < 12:
            continue
        if _extract_email(s) or "linkedin.com" in s.lower():
            continue
        # if includes year range, it's likely a role line
        if re.search(r"\b(19|20)\d{2}\b", s) or re.search(r"\bPresent\b", s, flags=re.I):
            return s[:110]
        # or contains typical title words
        if re.search(r"\b(Manager|Director|Vice President|VP|Analyst|Engineer|Associate|Lead|Head)\b", s, flags=re.I):
            return s[:110]
    return None

def build_snapshot(resume_raw: str) -> dict:
    lines = _split_lines(resume_raw)
    header_lines, body_start = _find_header_block(lines)
    header_text = "\n".join(header_lines)

    name = _extract_name(header_lines) or None
    email = _extract_email(header_text) or _extract_email(resume_raw)
    phone = _extract_phone(header_text) or _extract_phone(resume_raw)
    linkedin = _extract_linkedin(header_text) or _extract_linkedin(resume_raw)
    languages = _extract_languages(resume_raw)
    edu = _extract_latest_education(resume_raw)
    role = _extract_latest_role(resume_raw)

    # simple location guess from header
    loc = None
    mloc = re.search(r"\b(Hong Kong|Taipei|Taiwan|Singapore|London|New York|Toronto|Vancouver)\b", header_text, flags=re.I)
    if mloc:
        loc = mloc.group(0)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "location": loc,
        "latest_education": edu,
        "latest_role": role,
        "languages": languages,
        "body_start_line": body_start,
        "header_lines": header_lines,
    }

def strip_header_for_evidence(resume_raw: str) -> str:
    """
    Remove header/contact-heavy lines so they don't become evidence.
    """
    lines = _split_lines(resume_raw)
    header_lines, body_start = _find_header_block(lines)
    body_lines = lines[body_start:] if body_start < len(lines) else lines

    cleaned = []
    for ln in body_lines:
        s = ln.strip()
        if not s:
            continue
        if _extract_email(s) or _extract_phone(s) or "linkedin.com" in s.lower():
            continue
        cleaned.append(s)
    return "\n".join(cleaned)

# ---------- Resume Preview formatting ----------

def format_resume_preview(resume_raw: str, max_chars: int = 1600) -> str:
    """
    Make the extracted text readable:
    - preserve line breaks
    - insert spacing around section headings
    - normalize bullets
    - truncate for preview
    """
    lines = _split_lines(resume_raw)
    out = []
    for ln in lines:
        s = _clean_spaces(ln)
        s = _clean_bullet_prefix(s)
        if not s:
            continue

        up = re.sub(r"[^A-Za-z ]", "", s).strip().upper()
        if any(h == up for h in SECTION_HEADINGS) or any(up.startswith(h) for h in SECTION_HEADINGS):
            out.append("")  # blank line before heading
            out.append(up.title())
            out.append("")  # blank line after heading
            continue

        # if looks like bullet content, add bullet
        if ln.strip().startswith(BULLET_CHARS) or re.match(r"^\s*[\-\*]\s+", ln):
            out.append(f"• {s}")
        else:
            out.append(s)

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n…"
    return text

# ---------- Evidence bullets + Interview kit ----------

def best_evidence_snippets(resume_body: str, jd_keywords: list[str], top_n: int = 5) -> list[str]:
    """
    Pick real lines/sentences from resume body that best match JD keywords.
    Excludes header already.
    Cleans bullets and avoids super long lines.
    """
    if not resume_body or not jd_keywords:
        return []

    lines = [l.strip() for l in re.split(r"[\n\r]+", resume_body) if l.strip()]
    # fallback to sentence split if needed
    if len(lines) < 8:
        lines = re.split(r"(?<=[.!?])\s+", resume_body)

    jd_set = set(jd_keywords)
    scored: list[tuple[int, str]] = []
    for line in lines:
        s = _clean_spaces(line)
        s = _clean_bullet_prefix(s)

        # discard contact-like lines just in case
        if _extract_email(s) or _extract_phone(s) or "linkedin.com" in s.lower():
            continue

        if len(s) < 28:
            continue

        toks = set(_tokens(s))
        score = sum(1 for kw in jd_set if kw in toks)
        if score <= 0:
            continue

        # mild boost if contains numbers (impact)
        if re.search(r"\b\d+(\.\d+)?\b", s):
            score += 1

        scored.append((score, s))

    scored.sort(key=lambda x: (-x[0], len(x[1])))

    snippets = []
    seen = set()
    for _, s in scored:
        s = s[:220] + ("…" if len(s) > 220 else "")
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(s)
        if len(snippets) >= top_n:
            break

    return snippets

def generate_interview_kit(jd_keywords: list[str], matched: list[str], missing: list[str], evidence: list[str], max_q: int = 5) -> list[str]:
    """
    Heuristic interview kit:
    - probe missing keywords (highest value)
    - validate evidence with deeper follow-ups
    """
    qs: list[str] = []

    # Focus on missing first
    for kw in missing[:3]:
        qs.append(f"Your resume doesn’t explicitly mention **{kw}** — can you walk me through any hands-on experience you have with it (scope, tools, and your role)?")

    # Evidence follow-ups
    for ev in evidence[:2]:
        # take a short anchor phrase
        anchor = ev[:120].rstrip("…")
        qs.append(f"You wrote: “{anchor}…” — what was your specific responsibility, and what was the measurable outcome?")

    # Role-style question from JD keywords
    if jd_keywords:
        core = ", ".join(jd_keywords[:3])
        qs.append(f"If you joined tomorrow, what would your **first 30 days** look like to deliver against: {core}?")

    # Deduplicate & cap
    clean = []
    seen = set()
    for q in qs:
        k = re.sub(r"\s+", " ", q.strip().lower())
        if k in seen:
            continue
        seen.add(k)
        clean.append(q)
        if len(clean) >= max_q:
            break
    return clean

def relevance_low(final_score: float, kw_score: float, sim_norm: float) -> bool:
    # conservative: only say "low" when both signals are weak
    return (final_score < 0.45 and kw_score < 0.22 and sim_norm < 0.55)

def evaluate_candidate(jd_text: str, res_text: str, weights: dict):
    jd_summary = summarize_text(jd_text, max_len=140)
    jd_keywords = _top_keywords_from_text(jd_text + " " + jd_summary, k=22)

    snapshot = build_snapshot(res_text)
    resume_body = strip_header_for_evidence(res_text)

    res_summary = summarize_text(resume_body if resume_body else res_text, max_len=140)

    sim_raw = compute_similarity(jd_summary, res_summary)
    sim_norm = (sim_raw + 1) / 2 if sim_raw < 1 else min(sim_raw, 1.0)

    kw_score = keyword_match_score(jd_keywords, (resume_body + " " + res_summary).strip())

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

    # matched/missing keywords
    resume_token_set = set(_tokens((resume_body or "") + " " + res_summary))
    matched = [k for k in jd_keywords if k in resume_token_set]
    missing = [k for k in jd_keywords if k not in resume_token_set]

    # evidence + kit (cap 5)
    evidence = []
    kit = []
    if not relevance_low(final_score, kw_score, sim_norm):
        evidence = best_evidence_snippets(resume_body, jd_keywords, top_n=5)
        kit = generate_interview_kit(jd_keywords, matched, missing, evidence, max_q=5)

    return {
        "jd": {"raw": jd_text, "summary": jd_summary, "keywords": jd_keywords},
        "resume": {"raw": res_text, "body": resume_body, "summary": res_summary},
        "snapshot": snapshot,
        "similarity_raw": sim_raw,
        "similarity": sim_norm,
        "keyword_score": kw_score,
        "fit": fit,
        "prob_good_fit": float(prob_good_fit),
        "final_score": float(final_score),
        "matched_keywords": matched[:10],
        "missing_keywords": missing[:10],
        "evidence": evidence,
        "interview_kit": kit,
        "low_relevance": relevance_low(final_score, kw_score, sim_norm),
    }

def evaluate_batch(jd_text: str, resumes_list: list, weights: dict):
    out = []
    for item in resumes_list:
        name = item.get("name", "Unknown")
        text = item.get("text", "")
        if not text:
            continue
        r = evaluate_candidate(jd_text, text, weights)
        r["file_name"] = name
        out.append(r)
    return sorted(out, key=lambda x: x["final_score"], reverse=True)


# ============================================================
# 3. TOP-LEVEL ACTIONS
# ============================================================

def new_evaluation():
    st.session_state["jd_text"] = ""
    st.session_state["resume_text"] = ""
    st.session_state["upload_note"] = ""
    st.session_state["active_candidate_idx"] = 0

    st.session_state.pop("last_single_result", None)
    st.session_state.pop("last_batch_results", None)

    # reset uploaders
    st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1

    # clear busy/action
    st.session_state["pending_action"] = None
    st.session_state["is_busy"] = False

    st.rerun()


# ============================================================
# 4. HEADER
# ============================================================

st.title("🔍 MatchAI: Candidate Suitability Screening")
st.markdown(
    "<span class='muted'>Screen and prioritise candidates against a specific job description.</span>",
    unsafe_allow_html=True,
)

# Top-level New evaluation button (as requested)
top_btn_col, _ = st.columns([0.25, 0.75])
with top_btn_col:
    st.button("New evaluation", on_click=new_evaluation)


# ============================================================
# 5. INPUTS CARD
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
            key=f"single_upload_{st.session_state['uploader_nonce']}",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # If file uploaded, read it and populate resume_text ONLY if empty
        raw_extracted = None
        if uploaded_file is not None:
            raw_extracted = _read_uploaded_file(uploaded_file)
            if raw_extracted and not st.session_state.get("resume_text", "").strip():
                st.session_state["resume_text"] = raw_extracted
            if raw_extracted:
                st.session_state["upload_note"] = f"Text extracted from: {uploaded_file.name[:60]}"

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

        # Resume preview (formatted) + optional raw expander
        if st.session_state.get("resume_text", "").strip():
            st.markdown("<div class='section-title' style='margin-top:0.35rem;'>Resume preview</div>", unsafe_allow_html=True)
            preview = format_resume_preview(st.session_state["resume_text"])
            st.markdown(f"<div class='preview-box'>{preview}</div>", unsafe_allow_html=True)

            with st.expander("Raw extracted text (debugging)"):
                st.text(st.session_state["resume_text"][:8000])

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

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 6. EVALUATION & RESULTS CARD
# ============================================================

with st.container():
    st.markdown("<div class='round-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-label'>EVALUATION</div>", unsafe_allow_html=True)

    top_row = st.columns([0.5, 0.5])
    with top_row[0]:
        st.markdown("<div class='section-title'>MatchAI score</div>", unsafe_allow_html=True)
    with top_row[1]:
        with st.expander("How is the score calculated (?)", expanded=False):
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
            st.session_state.w_clf = float(DEFAULT_WEIGHTS["classifier"])
        if "w_sim" not in st.session_state:
            st.session_state.w_sim = float(DEFAULT_WEIGHTS["similarity"])
        if "w_kw" not in st.session_state:
            st.session_state.w_kw = float(DEFAULT_WEIGHTS["keywords"])

        c1, c2, c3 = st.columns(3)
        with c1:
            w_clf = st.slider(
                "Weight: P(Good Fit)",
                0.0, 1.0, float(st.session_state.w_clf), 0.05,
                key="slider_w_clf",
            )
        with c2:
            w_sim = st.slider(
                "Weight: Similarity",
                0.0, 1.0, float(st.session_state.w_sim), 0.05,
                key="slider_w_sim",
            )
        with c3:
            w_kw = st.slider(
                "Weight: Keywords",
                0.0, 1.0, float(st.session_state.w_kw), 0.05,
                key="slider_w_kw",
            )

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
            st.session_state.w_clf = float(DEFAULT_WEIGHTS["classifier"])
            st.session_state.w_sim = float(DEFAULT_WEIGHTS["similarity"])
            st.session_state.w_kw = float(DEFAULT_WEIGHTS["keywords"])
            st.rerun()

    # Ensure weights exists even if expander isn't opened
    if "weights" not in locals():
        tot = DEFAULT_WEIGHTS["classifier"] + DEFAULT_WEIGHTS["similarity"] + DEFAULT_WEIGHTS["keywords"]
        weights = {
            "classifier": DEFAULT_WEIGHTS["classifier"] / tot,
            "similarity": DEFAULT_WEIGHTS["similarity"] / tot,
            "keywords": DEFAULT_WEIGHTS["keywords"] / tot,
        }

    st.markdown("---")

    # ---------- Disable button + label change pattern ----------
    # If clicked, we set pending_action and rerun. Next run executes the action while button shows disabled + label.
    def request_action(which: str):
        st.session_state["pending_action"] = which
        st.session_state["is_busy"] = True
        st.rerun()

    busy = bool(st.session_state.get("is_busy", False))
    pending = st.session_state.get("pending_action")

    button_row = st.columns([0.35, 0.35, 0.3])
    with button_row[0]:
        st.button(
            "Scoring… (single)" if (busy and pending == "single") else "🔍 Score single candidate",
            disabled=busy,
            on_click=request_action,
            args=("single",),
        )
    with button_row[1]:
        st.button(
            "Scoring… (batch)" if (busy and pending == "batch") else "🔍 Score candidates (batch)",
            disabled=busy,
            on_click=request_action,
            args=("batch",),
        )

    # ---------- Execute pending action ----------
    if pending == "single":
        jd_val = st.session_state.get("jd_text", "")
        resume_val = st.session_state.get("resume_text", "")

        if not jd_val.strip():
            st.session_state["is_busy"] = False
            st.session_state["pending_action"] = None
            st.error("Please provide a job description in the Inputs section.")
        elif not resume_val.strip():
            st.session_state["is_busy"] = False
            st.session_state["pending_action"] = None
            st.error("Please provide resume text or upload a file in the Inputs section.")
        else:
            with st.spinner("Scoring candidate…"):
                result = evaluate_candidate(jd_val, resume_val, weights)
            st.session_state["last_single_result"] = result
            st.session_state["is_busy"] = False
            st.session_state["pending_action"] = None
            st.rerun()

    if pending == "batch":
        jd_val = st.session_state.get("jd_text", "")

        # retrieve current batch files from dynamic key
        batch_files = None
        if st.session_state.get("mode") == "Batch (multiple resumes)":
            for k in list(st.session_state.keys()):
                if k.startswith("batch_upload_"):
                    batch_files = st.session_state.get(k)
                    break

        if not jd_val.strip():
            st.session_state["is_busy"] = False
            st.session_state["pending_action"] = None
            st.error("Please provide a job description in the Inputs section.")
        elif not batch_files:
            st.session_state["is_busy"] = False
            st.session_state["pending_action"] = None
            st.error("Please upload at least one resume file in the Inputs section.")
        else:
            MAX_RESUMES = 30
            if len(batch_files) > MAX_RESUMES:
                st.session_state["is_busy"] = False
                st.session_state["pending_action"] = None
                st.error(f"Too many files uploaded – maximum {MAX_RESUMES} per batch.")
            else:
                resumes_list = []
                for f in batch_files:
                    txt = _read_uploaded_file(f)
                    if txt.strip():
                        resumes_list.append({"name": f.name, "text": txt})

                if not resumes_list:
                    st.session_state["is_busy"] = False
                    st.session_state["pending_action"] = None
                    st.error("No readable text found in uploaded resumes.")
                else:
                    with st.spinner("Scoring candidates…"):
                        batch_results = evaluate_batch(jd_val, resumes_list, weights)
                    st.session_state["last_batch_results"] = batch_results
                    st.session_state["active_candidate_idx"] = 0
                    st.session_state["is_busy"] = False
                    st.session_state["pending_action"] = None
                    st.rerun()

    # ============================================================
    # SINGLE CANDIDATE VIEW (Final structure)
    # 1) Final score + priority
    # 2) Snapshot
    # 3) Evidence bullets (max 5) OR low relevance message
    # 4) Interview kit (max 5)
    # ============================================================

    if "last_single_result" in st.session_state:
        r = st.session_state["last_single_result"]

        score_pct = r["final_score"] * 100
        label = r["fit"]["label_name"]
        prob_good = r["prob_good_fit"]

        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Result – Single candidate</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='soft-label'>Final suitability score</p>"
            f"<p class='score-number'>{score_pct:.1f}%</p>",
            unsafe_allow_html=True,
        )
        st.write(f"**Predicted fit label:** {label}")
        st.write(f"**P(Good Fit):** {prob_good:.2f}")
        st.write(f"**{map_priority(r['final_score'])}**")
        st.markdown("</div>", unsafe_allow_html=True)

        # Snapshot box (clean header/contact + latest edu/role + languages)
        snap = r.get("snapshot", {}) or {}
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Candidate snapshot</div>", unsafe_allow_html=True)

        def _snap_item(k, v):
            vv = v if (v and str(v).strip()) else "—"
            return f"<div class='snap-item'><div class='snap-k'>{k}</div><div class='snap-v'>{vv}</div></div>"

        snap_html = "<div class='snap-grid'>" + "".join([
            _snap_item("Name", snap.get("name")),
            _snap_item("Email", snap.get("email")),
            _snap_item("Phone", snap.get("phone")),
            _snap_item("LinkedIn", snap.get("linkedin")),
            _snap_item("Location", snap.get("location")),
            _snap_item("Languages", snap.get("languages")),
            _snap_item("Latest education", snap.get("latest_education")),
            _snap_item("Latest role", snap.get("latest_role")),
        ]) + "</div>"

        st.markdown(snap_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Evidence bullets
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Evidence bullets</div>", unsafe_allow_html=True)

        if r.get("low_relevance", False):
            st.write("Relevance looks **low** based on the current resume content. Evidence bullets are not shown to avoid misleading signals.")
        else:
            ev = r.get("evidence", [])[:5]
            if not ev:
                st.write("No clear, keyword-aligned evidence found in the resume body.")
            else:
                for s in ev:
                    s = _clean_bullet_prefix(_clean_spaces(s))
                    st.markdown(f"- “{s}”")

        st.markdown("</div>", unsafe_allow_html=True)

        # Interview kit
        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Interview kit</div>", unsafe_allow_html=True)

        kit = r.get("interview_kit", [])[:5]
        if not kit:
            st.write("No interview prompts generated (insufficient evidence or low relevance).")
        else:
            for q in kit:
                st.markdown(f"- {q}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Metrics row (keep minimal)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Semantic similarity**")
            st.write(f"{r['similarity']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Keyword coverage**")
            st.write(f"{r['keyword_score']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.write("**Model confidence**")
            st.write(f"{r['prob_good_fit']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # BATCH VIEW (Final structure)
    # 1) Ranked table
    # 2) Candidate selector
    # 3) Evidence bullets (short, max 5)
    # 4) Interview focus (short, max 5)
    # ============================================================

    if "last_batch_results" in st.session_state:
        batch_results = st.session_state["last_batch_results"]

        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Result – Batch candidate ranking</div>", unsafe_allow_html=True)

        rows = []
        for rank, rr in enumerate(batch_results, start=1):
            rows.append(
                {
                    "Rank": rank,
                    "File": rr.get("file_name", f"Candidate {rank}"),
                    "Fit label": rr["fit"]["label_name"],
                    "Final score (%)": round(rr["final_score"] * 100, 1),
                    "P(Good Fit)": round(rr["prob_good_fit"], 3),
                    "Similarity": round(rr["similarity"], 3),
                    "Keyword score": round(rr["keyword_score"], 3),
                }
            )

        st.dataframe(rows, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='io-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Candidate details</div>", unsafe_allow_html=True)

        options = [f"{i+1}. {rr.get('file_name','Candidate')}" for i, rr in enumerate(batch_results)]
        current_i = int(st.session_state.get("active_candidate_idx", 0))
        chosen = st.selectbox("Select a candidate", options=options, index=min(current_i, len(options)-1))
        selected_i = int(str(chosen).split(".")[0]) - 1
        st.session_state["active_candidate_idx"] = selected_i

        sel = batch_results[selected_i]
        st.write(f"**File:** {sel.get('file_name', 'N/A')}")
        st.write(f"**Final score:** {sel['final_score']*100:.1f}%")
        st.write(f"**Fit label:** {sel['fit']['label_name']}")
        st.write(f"**{map_priority(sel['final_score'])}**")

        # short evidence
        st.markdown("**Evidence bullets (short)**")
        if sel.get("low_relevance", False):
            st.write("Relevance looks **low**. Evidence bullets are not shown.")
        else:
            ev = sel.get("evidence", [])[:5]
            if not ev:
                st.write("No clear evidence found.")
            else:
                for s in ev:
                    s = _clean_bullet_prefix(_clean_spaces(s))
                    st.markdown(f"- “{s}”")

        st.markdown("**Interview focus (short)**")
        kit = sel.get("interview_kit", [])[:5]
        if not kit:
            st.write("No interview prompts generated.")
        else:
            for q in kit:
                st.markdown(f"- {q}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# END
# ============================================================
