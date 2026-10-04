"""Job-description analysis: boilerplate removal, skills (required vs preferred), experience,
education, keywords. Every field records whether it was EXTRACTED from the text/dataset or INFERRED."""
import re
import numpy as np
import pandas as pd
from collections import Counter
from .education import job_min_education, education_level, LEVEL_NAMES
from .skill_extractor import extract_skills, ALL_SKILLS
from .text_processor import tokenize, remove_stopwords

_BOILER = [r"^\s*Job Description\s*", r"Send me Jobs like this", r"Download PPT\b.*$", r"Photo \d+", r"View Contact Details",
           r"Walk-?in.*?(?=\.\s|$)"]
_PREF_CUES = re.compile(r"\b(preferred|preferably|nice to have|good to have|a plus|is a plus|desirable|added advantage|an advantage|"
                        r"bonus|optional|would be an asset)\b|(?<!candidate )\bdesired\b(?! candidate profile)", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s[•*·●]\s|\s-\s|\n")


def clean_job_text(desc: str) -> str:
    """Strip Naukri boilerplate and the company-profile paragraph; keep requirements text."""
    t = (desc or "").replace("\xa0", " ")
    t = re.split(r"Company Profile\s*:", t, maxsplit=1, flags=re.I)[0]
    for b in _BOILER:
        t = re.sub(b, " ", t, flags=re.I)
    t = re.sub(r"(\+?\d[\d\-\s()]{8,}\d)", " ", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)           # keep line breaks: they separate bullets for preferred-skill detection
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def _keyskills_block(text: str) -> str:
    m = re.search(r"Key\s?skills\s*[:\-]?\s*(.*?)(?=Desired Candidate Profile|Education\s*-|Company Profile|$)", text, re.I | re.S)
    return m.group(1) if m else ""


def parse_experience(field=None, text: str = ""):
    """Returns dict(min, max, source). Structured 'x - y yrs' field first, then text pattern."""
    if isinstance(field, str):
        m = re.search(r"(\d+)\s*-\s*(\d+)\s*yrs?", field, re.I)
        if m:
            return {"min": float(m.group(1)), "max": float(m.group(2)), "source": "extracted (dataset field)"}
        m = re.search(r"(\d+)\s*\+\s*yrs?", field, re.I)
        if m:
            return {"min": float(m.group(1)), "max": None, "source": "extracted (dataset field)"}
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*\+?\s*(?:years?|yrs?)", text, re.I)
    if m:
        return {"min": float(m.group(1)), "max": float(m.group(2)), "source": "extracted (text pattern)"}
    m = re.search(r"(?:minimum|min\.?|at least)?\s*(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:relevant\s*|hands[- ]on\s*)?experience", text, re.I)
    if m:
        return {"min": float(m.group(1)), "max": None, "source": "extracted (text pattern)"}
    if re.search(r"\bfresher", text, re.I):
        return {"min": 0.0, "max": 1.0, "source": "inferred (mentions freshers)"}
    return {"min": None, "max": None, "source": None}


def split_required_preferred(text: str):
    """Skills in the key-skills block or in non-qualified sentences are REQUIRED (extracted).
    Skills mentioned only in sentences with 'preferred / nice to have / plus ...' are PREFERRED (inferred)."""
    key = set(extract_skills(_keyskills_block(text)))
    all_sk = set(extract_skills(text))
    if not _PREF_CUES.search(text):
        return sorted(all_sk), []
    pref_only, req = set(), set(key)
    for sent in _SENT_SPLIT.split(text):
        sk = set(extract_skills(sent))
        if not sk:
            continue
        (pref_only if _PREF_CUES.search(sent) else req).update(sk)
    pref = pref_only - req
    return sorted(all_sk - pref), sorted(pref)


def extract_job_keywords(text: str, top: int = 12) -> list:
    skill_words = {w for s in ALL_SKILLS for w in s.lower().split()}
    toks = [t for t in remove_stopwords(tokenize(text)) if len(t) > 3 and t not in skill_words and "." not in t]
    return [w for w, _ in Counter(toks).most_common(top)]


def analyze_job(description: str, title: str = None, experience_field=None, education_field=None) -> dict:
    """Analyse one job (pasted text or dataset row). Missing info is returned as None / empty."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Job description is empty.")
    text = clean_job_text(description)
    if title is None:                                   # pasted JD: try first-line title
        first = description.strip().splitlines()[0].strip()
        title = re.sub(r"^(job\s*title|title|role|position)\s*[:\-]\s*", "", first, flags=re.I) if len(first) < 80 else None
        title_src = "extracted (first line)" if title else None
    else:
        title_src = "extracted (dataset field)"
    req, pref = split_required_preferred(text)
    edu_text = education_field if isinstance(education_field, str) else None
    edu_src = "extracted (dataset field)" if edu_text else None
    if not edu_text:
        m = re.search(r"Education\s*-\s*(.*?)(?=Company Profile|$)", description, re.I | re.S)
        edu_text = m.group(1).strip() if m else None
        edu_src = "extracted (text pattern)" if edu_text else None
        if not edu_text and (m := re.search(r"((?:B\.?\s?Tech|B\.?\s?Sc|M\.?\s?Tech|MBA|MCA|BCA|Bachelor'?s?|Master'?s?)[^;\n]{0,50})", description, re.I)):
            edu_text, edu_src = m.group(1).strip(), "inferred (degree mentioned in text)"
    level = job_min_education(edu_text) if edu_text else None
    return {"title": title, "title_source": title_src, "clean_description": text,
            "required_skills": req, "required_source": "extracted (skill dictionary match)",
            "preferred_skills": pref, "preferred_source": "inferred (wording such as 'preferred', 'plus', 'nice to have')" if pref else None,
            "experience": parse_experience(experience_field, text),
            "education": {"text": edu_text, "min_level": level, "level_name": LEVEL_NAMES.get(level) if level is not None else None, "source": edu_src},
            "keywords": extract_job_keywords(text)}


def enrich_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Add analysis columns to every job row; a malformed row never stops the batch."""
    rows = []
    for t, d, e, ed in zip(df.job_title, df.job_description, df.experience, df.education):
        try:
            text = clean_job_text(d)
            req, pref = split_required_preferred(text)
            ex = parse_experience(e, text)
            edu_text = ed if isinstance(ed, str) else None
            rows.append((text, req, pref, ex["min"], ex["max"], job_min_education(edu_text) if edu_text else None))
        except Exception:
            rows.append(("", [], [], None, None, None))
    out = df.copy()
    out["clean_description"], out["req_skills"], out["pref_skills"], out["exp_min"], out["exp_max"], out["edu_level"] = zip(*rows)
    out["n_skills"] = out.req_skills.map(len) + out.pref_skills.map(len)
    return out
