"""Job recommendation over the Naukri dataset: ranks jobs by the weighted compatibility score."""
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import PROCESSED_DIR, MIN_MATCHED_SKILLS
from .classifier import weak_label
from .matcher import combine, education_score, experience_score, skill_score, text_similarity, TEXT_SIM_FULL_SCORE
from .text_processor import preprocess


class JobIndex:
    """Holds cleaned jobs plus a TF-IDF index over their descriptions (cached on disk)."""

    def __init__(self, jobs: pd.DataFrame):
        self.jobs = jobs[jobs.is_valid].reset_index(drop=True)
        cache = PROCESSED_DIR / "job_tfidf.pkl"
        if cache.exists() and cache.stat().st_mtime >= (PROCESSED_DIR / "jobs_clean.pkl").stat().st_mtime:
            self.vec, self.X = joblib.load(cache)
        else:
            texts = self.jobs.clean_description.map(preprocess)
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=40000, sublinear_tf=True)
            self.X = self.vec.fit_transform(texts)
            joblib.dump((self.vec, self.X), cache)

    def sims(self, text: str) -> np.ndarray:
        v = self.vec.transform([preprocess(text)])
        return (self.X @ v.T).toarray().ravel()

    def similar_job_positions(self, text: str, n: int = 300) -> np.ndarray:
        s = self.sims(text)
        return np.argsort(s)[::-1][:n]


def recommend_jobs(resume: dict, index: JobIndex, top_n: int = 10, min_job_skills: int = 1) -> pd.DataFrame:
    """Rank all valid jobs for a resume. Only skills, experience, education and text are used."""
    jobs = index.jobs
    cand_text = resume.get("match_text") or resume["raw_text"]
    sims = index.sims(cand_text)
    rows = []
    need = min(MIN_MATCHED_SKILLS, max(1, len(resume["skills"])))   # small skill lists still get results
    cand_years, cand_edu = resume["experience"]["years"], resume["education"]["level"]
    for i, (req, pref, emin, elv) in enumerate(zip(jobs.req_skills, jobs.pref_skills, jobs.exp_min, jobs.edu_level)):
        if len(req) + len(pref) < min_job_skills:
            continue
        sk, matched, miss, miss_pref = skill_score(resume["skills"], req, pref)
        if sk is None or len(matched) < need:
            continue
        txt = min(1.0, float(sims[i]) / TEXT_SIM_FULL_SCORE)
        overall, _ = combine({"skills": sk, "text": txt, "experience": experience_score(cand_years, emin),
                              "education": education_score(cand_edu, elv)})
        rows.append((i, overall, sk, txt, matched, miss, miss_pref))
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows, columns=["pos", "score", "skill", "text", "matching_skills", "missing_skills", "missing_preferred"])
    r = r.sort_values("score", ascending=False)
    r["_key"] = jobs.job_title.iloc[r.pos].str.lower().values + "|" + jobs.company.fillna("").astype(str).str.lower().iloc[r.pos].values
    r = r.drop_duplicates("_key").head(top_n)
    j = jobs.iloc[r.pos.values].reset_index(drop=True)
    r = r.reset_index(drop=True)
    out = pd.DataFrame({
        "job_title": j.job_title, "company": j.company, "location": j.location, "industry": j.industry,
        "match_score": (100 * r.score).round(1), "skill_score": (100 * r.skill).round(1), "text_score": (100 * r.text).round(1),
        "matching_skills": r.matching_skills, "missing_skills": r.missing_skills, "missing_preferred": r.missing_preferred,
        "experience": j.experience, "education": j.education, "payrate": j.payrate,
        "job_description": j.clean_description, "req_skills": j.req_skills, "pref_skills": j.pref_skills,
        "exp_min": j.exp_min, "exp_max": j.exp_max, "edu_level": j.edu_level})
    return out


_SENIORITY = re.compile(r"\b(sr|senior|jr|junior|lead|principal|associate|asst|assistant|trainee|intern|urgent|hiring|opening[s]?|required|wanted|vacancy|for|walk-?in|immediate|requirement|executive|manager)\b|\(.*?\)|\d+(\.\d+)?\s*\+?\s*(yrs?|years?)|[\d+]+", re.I)


def normalize_role(title: str) -> str:
    t = re.split(r"\s[-–|/]\s|,|\||:", title)[0]
    t = _SENIORITY.sub(" ", t)
    t = re.sub(r"[^A-Za-z.# ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().title()
    return t or title.title()


_EXTRA_FAMILIES = [
    ("QA / Test Engineer", r"\bqa\b|test(?:er|ing)|quality assurance|selenium"),
    ("DevOps / Cloud Engineer", r"devops|cloud|aws|azure|system administrator|sysadmin|infrastructure|network engineer"),
    ("Database / ETL Developer", r"database|dba|etl|data warehouse|\bsql\b|oracle|informatica|data engineer"),
    ("Project / Product Manager", r"project manager|product manager|program manager|scrum|delivery manager"),
    ("Sales / Marketing", r"sales|marketing|business development|seo"),
]


def role_family(title: str) -> str:
    """Six project roles via the title rules, then a few extra families, else a cleaned title."""
    r = weak_label(title)
    if r:
        return r
    for name, pat in _EXTRA_FAMILIES:
        if re.search(pat, title, re.I):
            return name
    return "Other: " + normalize_role(title)[:30]


def recommended_roles(recs: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Group recommended jobs into role families with count and average match score."""
    if recs is None or recs.empty:
        return pd.DataFrame(columns=["role", "jobs", "avg_score"])
    g = recs.assign(role=recs.job_title.map(role_family)).groupby("role").agg(jobs=("match_score", "size"), avg_score=("match_score", "mean"))
    return g.sort_values(["jobs", "avg_score"], ascending=False).head(top).reset_index().round({"avg_score": 1})
