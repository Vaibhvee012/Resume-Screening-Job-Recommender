"""Skill-gap analysis: matching vs missing skills, and a learning list prioritised by how often each
missing skill appears in the most similar job postings of the dataset."""
import pandas as pd
from .recommender import JobIndex


def analyze_gap(candidate_skills, required, preferred, index: JobIndex, job_text: str, n_similar: int = 300) -> dict:
    cand = {s.lower() for s in candidate_skills}
    req, pref = list(required), [p for p in preferred if p not in set(required)]
    matching = [s for s in req + pref if s.lower() in cand]
    miss_req = [s for s in req if s.lower() not in cand]
    miss_pref = [s for s in pref if s.lower() not in cand]
    pos = index.similar_job_positions(job_text, n_similar)
    sim_jobs = index.jobs.iloc[pos]
    freq = {}
    for rs, ps in zip(sim_jobs.req_skills, sim_jobs.pref_skills):
        for s in set(rs) | set(ps):
            freq[s] = freq.get(s, 0) + 1
    rows = [{"skill": s, "type": "Required" if s in miss_req else "Preferred",
             "seen_in_similar_jobs_pct": round(100 * freq.get(s, 0) / len(sim_jobs), 1)} for s in miss_req + miss_pref]
    learn = pd.DataFrame(rows, columns=["skill", "type", "seen_in_similar_jobs_pct"])
    if not learn.empty:
        learn["_t"] = (learn.type == "Required").astype(int)
        learn = learn.sort_values(["_t", "seen_in_similar_jobs_pct"], ascending=False).drop(columns="_t").reset_index(drop=True)
    return {"matching": matching, "missing_required": miss_req, "missing_preferred": miss_pref,
            "learn": learn, "n_similar_jobs": len(sim_jobs)}
