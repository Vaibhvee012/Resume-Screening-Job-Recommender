"""Dataset discovery, column mapping, cleaning. Original files are never modified;
cleaned copies are written to data/processed/."""
import re
import pandas as pd
from .config import DATA_DIRS, PROCESSED_DIR

JOB_ALIASES = {
    "job_title": ["jobtitle", "job_title", "title", "position"],
    "job_description": ["jobdescription", "job_description", "description", "jd"],
    "skills": ["skills", "keyskills", "key_skills", "skill"],
    "experience": ["experience", "exp", "experience_required"],
    "education": ["education", "qualification", "educational_requirements"],
    "company": ["company", "company_name", "employer"],
    "location": ["joblocation_address", "location", "joblocation", "job_location", "city"],
    "industry": ["industry", "sector"],
    "payrate": ["payrate", "salary"],
    "job_id": ["uniq_id", "jobid", "job_id", "id"],
}
RESUME_ALIASES = {
    "resume_text": ["resume_str", "resume_text", "resume", "text", "cv"],
    "category": ["category", "label", "role", "job_category"],
}


class DatasetError(Exception):
    """Raised with a user-friendly message when a dataset is missing or incompatible."""


def _map_columns(columns, aliases):
    low = {c.lower().strip(): c for c in columns}
    mapping = {}
    for std, opts in aliases.items():
        for o in opts:
            if o in low:
                mapping[std] = low[o]
                break
    return mapping


def _read_any(path):
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise DatasetError(f"Unsupported file format: {path.name}")


def discover_datasets():
    """Scan dataset dirs; classify every table file as 'jobs' or 'resumes' by its columns."""
    found = {"jobs": None, "resumes": None, "other": []}
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".csv", ".xlsx", ".xls", ".parquet"):
                continue
            try:
                head = (pd.read_csv(p, nrows=2, encoding="latin-1") if p.suffix.lower() == ".csv"
                        else _read_any(p).head(2))
            except Exception:
                found["other"].append(p)
                continue
            jm, rm = _map_columns(head.columns, JOB_ALIASES), _map_columns(head.columns, RESUME_ALIASES)
            if {"job_title", "job_description"} <= set(jm) and found["jobs"] is None:
                found["jobs"] = p
            elif {"resume_text", "category"} <= set(rm) and "job_description" not in jm and found["resumes"] is None:
                found["resumes"] = p
            else:
                found["other"].append(p)
    return found


# ----------------------------------------------------------------- jobs
_JUNK_PATTERNS = re.compile(r"(?:\d[\s-]?){9,}|whats?app|wtsap|click here", re.I)


def load_raw_jobs(path=None) -> pd.DataFrame:
    path = path or discover_datasets()["jobs"]
    if path is None:
        raise DatasetError("No job dataset found. Place the Naukri CSV in the 'dataset/' folder "
                           "(needs at least job title and job description columns).")
    df = _read_any(path)
    mapping = _map_columns(df.columns, JOB_ALIASES)
    missing = {"job_title", "job_description"} - set(mapping)
    if missing:
        raise DatasetError(f"Job dataset is missing required columns {sorted(missing)}. "
                           f"Columns found: {list(df.columns)}")
    out = pd.DataFrame({std: df[src] for std, src in mapping.items()})
    for std in JOB_ALIASES:                      # optional fields -> present but empty
        if std not in out:
            out[std] = pd.NA
    return out


def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty/duplicate postings; flag junk titles. Returns a processed copy."""
    n0 = len(df)
    df = df.dropna(subset=["job_description", "job_title"]).copy()
    df["job_description"] = df["job_description"].astype(str).str.replace("\xa0", " ")
    df["job_title"] = df["job_title"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["job_title", "company", "job_description"]).reset_index(drop=True)
    df["location"] = df["location"].astype("string").str.split(r"[,/]").str[0].str.strip() \
        .replace({"Bangalore": "Bengaluru", "Hyderabad": "Hyderabad", "Delhi/NCR": "Delhi"})
    df["is_valid"] = ~df["job_title"].str.contains(_JUNK_PATTERNS) & (df["job_title"].str.len() < 90)
    df.attrs["dropped"] = n0 - len(df)
    return df


def load_jobs(use_cache: bool = True) -> pd.DataFrame:
    """Load cleaned jobs (with extracted skills/experience/education), cached in data/processed/."""
    cache = PROCESSED_DIR / "jobs_clean.pkl"
    src = discover_datasets()["jobs"]
    if use_cache and cache.exists() and src is not None and cache.stat().st_mtime > src.stat().st_mtime:
        return pd.read_pickle(cache)
    from .job_analyzer import enrich_jobs          # local import avoids circularity
    df = enrich_jobs(clean_jobs(load_raw_jobs(src)))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    df.drop(columns=["req_skills", "pref_skills"]).to_csv(PROCESSED_DIR / "jobs_clean.csv", index=False)
    return df


# -------------------------------------------------------------- resumes
def load_resume_dataset():
    """Return (DataFrame[resume_text, category], path) or (None, None) if no resume dataset exists."""
    path = discover_datasets()["resumes"]
    if path is None:
        return None, None
    df = _read_any(path)
    m = _map_columns(df.columns, RESUME_ALIASES)
    out = pd.DataFrame({"resume_text": df[m["resume_text"]], "category": df[m["category"]]})
    out = out.dropna().drop_duplicates()
    out["resume_text"] = out["resume_text"].astype(str)
    out["category"] = out["category"].astype(str).str.strip()
    return out[out.resume_text.str.len() > 50].reset_index(drop=True), path
