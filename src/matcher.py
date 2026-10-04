"""Interpretable weighted resume-job matching. All weights live in config.py.
The result is a 'Compatibility Score' - a heuristic, NOT a validated probability of being hired."""
import numpy as np
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import SCORE_WEIGHTS, PREFERRED_SKILL_WEIGHT, TEXT_SIM_FULL_SCORE
from .education import LEVEL_NAMES
from .text_processor import preprocess


# ---------------------------------------------------------------- components
def skill_score(candidate_skills, required, preferred=()):
    """Weighted share of job skills the candidate has (required=1.0, preferred=PREFERRED_SKILL_WEIGHT).
    Returns (score or None, matched, missing_required, missing_preferred). Skills are compared as normalised names."""
    cand = {s.lower() for s in candidate_skills}
    req, pref = list(required), [p for p in preferred if p not in set(required)]
    if not req and not pref:
        return None, [], [], []
    m_req = [s for s in req if s.lower() in cand]
    m_pref = [s for s in pref if s.lower() in cand]
    total = len(req) + PREFERRED_SKILL_WEIGHT * len(pref)
    got = len(m_req) + PREFERRED_SKILL_WEIGHT * len(m_pref)
    return got / total, m_req + m_pref, [s for s in req if s.lower() not in cand], [s for s in pref if s.lower() not in cand]


def experience_score(candidate_years, job_min, job_max=None):
    """1.0 if candidate meets the minimum; proportional below it. None if either side is unknown."""
    if candidate_years is None or job_min is None or (isinstance(job_min, float) and np.isnan(job_min)):
        return None
    if job_min <= 0 or candidate_years >= job_min:
        return 1.0
    return max(0.0, candidate_years / job_min)


def education_score(candidate_level, job_level):
    """1.0 if level meets requirement, 0.5 if one level short, else 0. None if either side is unknown."""
    if candidate_level is None or job_level is None or (isinstance(job_level, float) and np.isnan(job_level)):
        return None
    gap = job_level - candidate_level
    return 1.0 if gap <= 0 else (0.5 if gap == 1 else 0.0)


def text_similarity(resume_text, job_text, vectorizer=None):
    """Raw TF-IDF cosine similarity (0-1). Uses a corpus-fitted vectorizer when supplied."""
    a, b = preprocess(resume_text), preprocess(job_text)
    if not a or not b:
        return 0.0
    vec = vectorizer or TfidfVectorizer(ngram_range=(1, 2))
    X = vec.transform([a, b]) if vectorizer else vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def combine(components: dict, weights=None):
    """Weighted average over the components that are available (None = unavailable);
    weights are renormalised so a missing component does not unfairly lower the score."""
    w = weights or SCORE_WEIGHTS
    avail = {k: v for k, v in components.items() if v is not None}
    if not avail:
        return 0.0, {}
    tot = sum(w[k] for k in avail)
    used = {k: w[k] / tot for k in avail}
    return sum(avail[k] * used[k] for k in avail), used


# ------------------------------------------------------------------ main API
def match_resume_to_job(resume: dict, job: dict, vectorizer=None) -> dict:
    """resume: output of parse_resume; job: output of analyze_job. Returns all scores (0-100) and skill lists."""
    sk, matched, miss_req, miss_pref = skill_score(resume["skills"], job["required_skills"], job["preferred_skills"])
    txt_raw = text_similarity(resume.get("match_text") or resume["raw_text"], job["clean_description"], vectorizer)
    txt = min(1.0, txt_raw / TEXT_SIM_FULL_SCORE)
    ex = experience_score(resume["experience"]["years"], job["experience"]["min"], job["experience"]["max"])
    ed = education_score(resume["education"]["level"], job["education"]["min_level"])
    overall, used = combine({"skills": sk, "text": txt, "experience": ex, "education": ed})
    pct = lambda v: None if v is None else round(100 * v, 1)
    notes = []
    if ex is None:
        notes.append("Experience not compared (not stated in the resume or the job).")
    if ed is None:
        notes.append("Education not compared (not stated in the resume or the job).")
    if sk is None:
        notes.append("No skills detected in the job description, so the skill score is unavailable.")
    return {"overall": pct(overall), "skill": pct(sk), "text": pct(txt), "text_raw_cosine": round(txt_raw, 3),
            "experience": pct(ex), "education": pct(ed), "matching_skills": matched, "missing_skills": miss_req,
            "missing_preferred": miss_pref, "weights_used": {k: round(v, 3) for k, v in used.items()}, "notes": notes}


def score_label(score: float) -> str:
    return "Strong match" if score >= 70 else "Moderate match" if score >= 45 else "Weak match"
