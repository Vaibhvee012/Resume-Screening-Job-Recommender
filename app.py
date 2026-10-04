"""AI-Powered Resume Screening & Job Recommendation System - Streamlit app."""
import html
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import classifier, history
from src.config import PALETTE as C, SAMPLES_DIR, SCORE_WEIGHTS, TARGET_ROLES
from src.data_loader import DatasetError, discover_datasets, load_jobs
from src.job_analyzer import analyze_job
from src.skill_extractor import NON_TECH
from src.matcher import match_resume_to_job, score_label
from src.recommender import JobIndex, recommend_jobs, recommended_roles
from src.resume_parser import ResumeParseError, extract_text, parse_resume
from src.skill_gap import analyze_gap

st.set_page_config(page_title="Resume Screening & Job Recommender", page_icon="📄", layout="wide")

PAGES = ["Dashboard", "Resume Analyzer", "Job Matcher", "Job Recommendations", "Skill Gap", "Model Evaluation", "About Project"]

_css_path = Path(__file__).parent / "assets" / "style.css"
_css = _css_path.read_text(encoding="utf-8") if _css_path.exists() else ""
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

NAV_ICONS = {"Dashboard": "📊", "Resume Analyzer": "📄", "Job Matcher": "🎯", "Job Recommendations": "💼",
             "Skill Gap": "🧩", "Model Evaluation": "🧪", "About Project": "ℹ️"}
PAGE_SUBTITLES = {
    "Dashboard": "Overview of the job market, your analyses and model performance",
    "Resume Analyzer": "Upload a resume and extract skills, education and experience",
    "Job Matcher": "Compare your resume against a specific job posting",
    "Job Recommendations": "Best-fitting jobs ranked by compatibility score",
    "Skill Gap": "See which skills you have, which you lack, and what to learn next",
    "Model Evaluation": "Performance of the role classification model",
    "About Project": "Problem, methodology and limitations",
}


# ------------------------------------------------------------------ helpers
def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def page_header(title):
    st.markdown(f'<div class="page-head"><div class="ico">{NAV_ICONS[title]}</div>'
                f'<div><div class="t">{title}</div><div class="s">{PAGE_SUBTITLES[title]}</div></div></div>',
                unsafe_allow_html=True)


def chips(items, kind="info", empty="None"):
    if not items:
        return f'<span style="color:#9CA3AF">{empty}</span>'
    return "".join(f'<span class="chip {kind}">{html.escape(str(i))}</span>' for i in items)


def tag(src):
    if not src:
        return ""
    return f'<span class="tag{" inf" if src.startswith("inferred") else ""}">{html.escape(src)}</span>'


def score_color(v):
    return C["success"] if v >= 70 else C["warning"] if v >= 45 else C["critical"]


def scorecard(label, v, big=False):
    if v is None:
        st.markdown(f'<div class="scorecard"><div class="v" style="color:#9CA3AF">n/a</div><div class="l">{label}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="scorecard"><div class="v" style="color:{score_color(v)}">{v:.1f}%</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def style_fig(fig, h=330):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", color="#334155", size=12),
                      title=dict(font=dict(size=14, color=C["dark"]), x=0.01), legend_title_text="",
                      hoverlabel=dict(bgcolor="#0F172A", font_color="#F8FAFC", bordercolor="#0F172A"))
    fig.update_xaxes(gridcolor="#EEF2F7", zeroline=False, linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False, linecolor="#E2E8F0")
    return fig


@st.cache_data(show_spinner="Loading and analysing the job dataset (first run takes ~40 s)…")
def get_jobs():
    return load_jobs()


@st.cache_resource(show_spinner="Building job search index…")
def get_index():
    return JobIndex(get_jobs())


def get_model():
    try:
        return classifier.load_model()
    except classifier.ModelNotTrained:
        return None


def need_jobs():
    try:
        return get_jobs(), get_index()
    except DatasetError as e:
        st.error(f"Dataset problem: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Could not load the job dataset: {e}")
        st.stop()


def go(page):
    st.session_state.nav = page


def build_job(title, desc, experience=None, education=None, company=None, location=None, source=""):
    a = analyze_job(desc, title=title, experience_field=experience, education_field=education)
    a.update(company=company, location=location, source=source)
    return a


def render_job_analysis(a):
    st.markdown(f"**Job title:** {html.escape(a['title'] or 'Not detected')} {tag(a['title_source'])}", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Required skills** {tag(a['required_source'])}", unsafe_allow_html=True)
        st.markdown(chips(a["required_skills"], "info", "No skills detected"), unsafe_allow_html=True)
        st.markdown(f"**Preferred skills** {tag(a['preferred_source'])}", unsafe_allow_html=True)
        st.markdown(chips(a["preferred_skills"], "pref", "None identified (no 'preferred / nice to have' wording)"), unsafe_allow_html=True)
    with c2:
        e, ed = a["experience"], a["education"]
        if e["min"] is None:
            ex = "Not specified"
        else:
            ex = f"{e['min']:g}" + (f" – {e['max']:g}" if e["max"] is not None else "+") + " years"
        st.markdown(f"**Experience:** {ex} {tag(e['source'])}", unsafe_allow_html=True)
        st.markdown(f"**Education:** {html.escape(ed['text'][:120]) if ed['text'] else 'Not specified'} "
                    f"{('→ minimum ' + ed['level_name']) if ed['level_name'] else ''} {tag(ed['source'])}", unsafe_allow_html=True)
        st.markdown("**Important keywords** " + tag("extracted (term frequency)"), unsafe_allow_html=True)
        st.markdown(chips(a["keywords"], "info"), unsafe_allow_html=True)


def render_match(res):
    cols = st.columns(5)
    with cols[0]:
        scorecard(f"Compatibility Score · {score_label(res['overall'])}", res["overall"])
    with cols[1]: scorecard("Skill score", res["skill"])
    with cols[2]: scorecard(f"Text similarity (cosine {res['text_raw_cosine']})", res["text"])
    with cols[3]: scorecard("Experience score", res["experience"])
    with cols[4]: scorecard("Education score", res["education"])
    for n in res["notes"]:
        st.info(n)
    w = ", ".join(f"{k} {v:.0%}" for k, v in res["weights_used"].items())
    st.caption(f"Weights applied: {w}. The Compatibility Score is a heuristic for comparing profiles to a posting - not a validated probability of being hired.")
    a, b = st.columns(2)
    with a:
        st.markdown("**✅ Matching skills**")
        st.markdown(chips(res["matching_skills"], "ok", "No overlapping skills detected"), unsafe_allow_html=True)
    with b:
        st.markdown("**❌ Missing skills**")
        st.markdown(chips(res["missing_skills"], "miss", "No required skills missing"), unsafe_allow_html=True)
        if res["missing_preferred"]:
            st.markdown("**Missing preferred skills**")
            st.markdown(chips(res["missing_preferred"], "pref"), unsafe_allow_html=True)


def need_resume():
    r = st.session_state.get("resume")
    if r is None:
        st.warning("No resume analysed yet. Go to **Resume Analyzer** and upload a resume (or load a sample).")
        st.button("Open Resume Analyzer", on_click=go, args=("Resume Analyzer",))
        st.stop()
    return r


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">📄</div><div><div class="name">Resume Screening</div>'
                '<div class="tagline">AI Job Matching &amp; Recommender</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Menu</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed",
                    format_func=lambda p: f"{NAV_ICONS[p]}   {p}")
    st.divider()
    res_state = st.session_state.get("resume")
    st.markdown('<div class="nav-label">Session</div>', unsafe_allow_html=True)
    if res_state:
        st.markdown(f'<div class="side-card"><div class="k">Current resume</div>'
                    f'<div class="val"><span class="dot on"></span>{html.escape(str(st.session_state.get("resume_name")))}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-card"><div class="k">Current resume</div>'
                    '<div class="val"><span class="dot off"></span>None loaded</div></div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Reset session", use_container_width=True):
        for k in ["resume", "resume_name", "roles", "recs", "job", "match", "gap"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown('<div class="side-foot">AI Resume Screening System</div>', unsafe_allow_html=True)


# ================================================================= PAGES
def page_dashboard():
    page_header("Dashboard")
    jobs, index = need_jobs()
    valid = jobs[jobs.is_valid]
    hist = history.load_history()
    resumes = [h for h in hist if h["type"] == "resume"]
    scores = [s for h in hist for s in h.get("scores", [])]
    meta = (get_model() or {}).get("meta")
    k = st.columns(5)
    with k[0]: kpi("Resumes analysed", len(resumes), "from analysis history")
    with k[1]: kpi("Jobs available", f"{len(valid):,}", f"{len(jobs) - len(valid)} junk-title postings excluded")
    with k[2]: kpi("Average match score", f"{sum(scores) / len(scores):.1f}%" if scores else "–", f"over {len(scores)} scores recorded" if scores else "analyse a resume first")
    with k[3]: kpi("Job categories", valid.industry.nunique(), "industries in dataset")
    with k[4]: kpi("Classifier accuracy", f"{meta['accuracy']:.1%}" if meta else "–", "held-out test set" if meta else "model not trained")
    st.write("")
    a, b = st.columns(2)
    tech_only = a.toggle("Technical skills only (hide soft / business skills)", value=True)
    sk = pd.Series([s for l in valid.req_skills for s in l if not (tech_only and s in NON_TECH)]).value_counts().head(15).iloc[::-1]
    with a:
        fig = px.bar(x=sk.values, y=sk.index, orientation="h", title="Most common skills in job postings (dataset)", color_discrete_sequence=[C["primary"]])
        fig.update_layout(xaxis_title="Job postings", yaxis_title="")
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    with b:
        ind = valid.industry.fillna("Unknown").value_counts()
        top = pd.concat([ind.head(8), pd.Series({"Other industries": ind.iloc[8:].sum()})])
        fig = px.pie(values=top.values, names=top.index, hole=0.55, title="Job categories (industry)", color_discrete_sequence=px.colors.sequential.Purples_r[:7] + [C["primary"], C["secondary"], "#C7D2FE"])
        st.plotly_chart(style_fig(fig, 420), width="stretch")
    c, d = st.columns(2)
    with c:
        recs = st.session_state.get("recs")
        if recs is not None and len(recs):
            roles = recommended_roles(recs).iloc[::-1]
            fig = px.bar(roles, x="jobs", y="role", orientation="h", title="Top recommended roles (current resume)", color="avg_score",
                         color_continuous_scale=["#C7D2FE", C["primary"]], hover_data=["avg_score"])
            fig.update_layout(xaxis_title="Jobs in top recommendations", yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            st.info("Recommended roles appear here after you analyse a resume and open **Job Recommendations**.")
    with d:
        if scores:
            fig = px.histogram(x=scores, nbins=10, title="Distribution of recorded match scores", color_discrete_sequence=[C["secondary"]])
            fig.update_layout(xaxis_title="Match score (%)", yaxis_title="Count")
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            st.info("The match-score histogram appears after the first resume analysis.")
    e, f = st.columns(2)
    with e:
        gap = st.session_state.get("gap")
        if gap is not None and not gap["learn"].empty:
            g = gap["learn"].head(10).iloc[::-1]
            fig = px.bar(g, x="seen_in_similar_jobs_pct", y="skill", orientation="h", color="type", title="Skill gap: missing skills by demand in similar jobs",
                         color_discrete_map={"Required": C["critical"], "Preferred": C["warning"]})
            fig.update_layout(xaxis_title="% of similar postings listing the skill", yaxis_title="")
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            miss = pd.Series([s for h in hist for s in h.get("missing", [])]).value_counts().head(10).iloc[::-1] if hist else pd.Series(dtype=int)
            if len(miss):
                fig = px.bar(x=miss.values, y=miss.index, orientation="h", title="Most frequently missing skills (all analyses)", color_discrete_sequence=[C["critical"]])
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                st.info("The skill-gap chart appears after a job match or skill-gap analysis.")
    with f:
        exp = valid.exp_min.dropna().astype(int).clip(upper=15).value_counts().sort_index()
        fig = px.bar(x=exp.index, y=exp.values, title="Minimum experience required (dataset)", color_discrete_sequence=[C["secondary"]])
        fig.update_layout(xaxis_title="Years (15 = 15+)", yaxis_title="Job postings")
        st.plotly_chart(style_fig(fig), width="stretch")
    st.subheader("Recent analyses")
    if hist:
        rows = pd.DataFrame(hist[::-1][:10])
        cols = [c for c in ["time", "type", "resume", "predicted_role", "job", "score", "n_skills"] if c in rows]
        st.dataframe(rows[cols], width="stretch", hide_index=True)
        if st.button("Clear history"):
            history.clear_history(); st.rerun()
    else:
        st.caption("No analyses yet.")


def page_resume():
    page_header("Resume Analyzer")
    up = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    samples = sorted(p for p in (SAMPLES_DIR / "resumes").glob("*.*") if p.suffix.lower() in (".pdf", ".docx", ".txt")) if (SAMPLES_DIR / "resumes").exists() else []
    cols = st.columns([2, 1])
    choice = cols[0].selectbox("…or use a sample resume", ["–"] + [s.name for s in samples])
    text, name = None, None
    try:
        if up is not None:
            text, name = extract_text(up.getvalue(), up.name), up.name
        elif choice != "–":
            p = SAMPLES_DIR / "resumes" / choice
            text, name = extract_text(p.read_bytes(), p.name), p.name
    except ResumeParseError as e:
        st.error(str(e)); st.stop()
    if text is None:
        st.info("Upload a resume or pick a sample to begin.")
        if st.session_state.get("resume") is None:
            return
    if text is not None and st.session_state.get("resume_name") != name:
        with st.spinner("Parsing resume…"):
            r = parse_resume(text)
            model = get_model()
            roles = classifier.predict_roles(r["match_text"], model) if model else None
        st.session_state.update(resume=r, resume_name=name, roles=roles)
        for k in ["recs", "job", "match", "gap"]:
            st.session_state.pop(k, None)
        history.add_record({"type": "resume", "resume": name, "n_skills": len(r["skills"]), "predicted_role": roles[0][0] if roles else None})
    r, name = st.session_state.get("resume"), st.session_state.get("resume_name")
    if r is None:
        return
    st.success(f"Analysed **{name}** · {r['n_words']} words")
    if not r["skills"]:
        st.warning("No known skills were detected. The resume may be image-based, very short, or use skills outside the dictionary.")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Candidate information")
        nf = lambda v: html.escape(v) if v else '<span style="color:#9CA3AF">Not found</span>'
        st.markdown(f"**Name:** {nf(r['name'])}  \n**Email:** {nf(r['email'])}  \n**Phone:** {nf(r['phone'])}  \n"
                    f"**LinkedIn:** {nf(r['linkedin'])}  \n**GitHub:** {nf(r['github'])}", unsafe_allow_html=True)
        st.subheader("Education")
        ed = r["education"]
        st.markdown(f"**Degrees detected:** {', '.join(ed['degrees']) if ed['degrees'] else 'Not found'}  \n"
                    f"**Highest level:** {ed['level_name'] or 'Not determined'}")
        if ed["section_text"]:
            st.caption(ed["section_text"])
        st.subheader("Experience")
        ex = r["experience"]
        st.markdown(f"**Total experience:** {str(ex['years']) + ' years' if ex['years'] is not None else 'Not determined'}"
                    + (f"  ·  _{ex['method']}_" if ex["method"] else ""))
        if ex["section_text"]:
            with st.expander("Experience section text"):
                st.text(ex["section_text"])
    with right:
        st.subheader(f"Skills ({len(r['skills'])})")
        st.markdown(chips(r["skills"], "info", "None detected"), unsafe_allow_html=True)
        st.subheader("Relevant keywords")
        st.markdown(chips(r["keywords"], "info", "None"), unsafe_allow_html=True)
        st.subheader("Sections detected")
        st.markdown(chips(r["sections_found"], "info", "No clear sections found"), unsafe_allow_html=True)
        st.subheader("Predicted job category")
        roles = st.session_state.get("roles")
        if roles:
            fig = px.bar(x=[s for _, s in roles][::-1], y=[a for a, _ in roles][::-1], orientation="h", color_discrete_sequence=[C["primary"]])
            fig.update_layout(xaxis_title="Relative model score", yaxis_title="", xaxis_range=[0, 1])
            st.plotly_chart(style_fig(fig, 220), width="stretch")
            src = (get_model()["meta"] or {}).get("data_source")
            st.caption("Scores rank the roles; they are not calibrated probabilities. " +
                       ("This model was trained on weak labels from job postings (no resume dataset provided) - treat the prediction as a hint."
                        if src == "naukri_weak_labels" else ""))
        else:
            st.warning("Role classifier is not trained yet. Open **Model Evaluation** and click *Train model*.")
    with st.expander("Extracted resume text"):
        st.text(r["raw_text"][:6000])
    st.button("Find matching jobs →", on_click=go, args=("Job Recommendations",))


def page_matcher():
    page_header("Job Matcher")
    r = need_resume()
    jobs, index = need_jobs()
    mode = st.radio("Job source", ["Select from job dataset", "Paste a job description"], horizontal=True)
    job = None
    if mode == "Paste a job description":
        sample = SAMPLES_DIR / "job_descriptions" / "sample_data_scientist_jd.txt"
        if st.button("Load sample job description") and sample.exists():
            st.session_state.jd_text = sample.read_text(encoding="utf-8")
        text = st.text_area("Job description", key="jd_text", height=220, placeholder="Paste the full job description here…")
        if st.button("Analyse & match", type="primary"):
            if not text.strip():
                st.error("The job description is empty."); st.stop()
            job = build_job(None, text, source="pasted")
    else:
        q = st.text_input("Search job titles", value="data scientist")
        pool = index.jobs[index.jobs.job_title.str.contains(q, case=False, na=False, regex=False) & (index.jobs.n_skills >= 2)] if q else index.jobs[index.jobs.n_skills >= 3]
        pool = pool.sort_values("n_skills", ascending=False).head(200)
        if pool.empty:
            st.warning("No jobs match that search."); st.stop()
        pos = st.selectbox("Select a job", pool.index.tolist(), format_func=lambda i: f"{pool.job_title[i][:60]} — {str(pool.company[i])[:30]} ({pool.location[i]})")
        if st.button("Analyse & match", type="primary"):
            row = pool.loc[pos]
            job = build_job(row.job_title, row.job_description, row.experience, row.education, row.company, row.location, "dataset")
    if job:
        st.session_state.job = job
        st.session_state.match = match_resume_to_job(r, job, index.vec)
        m = st.session_state.match
        history.add_record({"type": "match", "resume": st.session_state.resume_name, "job": job["title"] or "Pasted JD",
                            "score": m["overall"], "scores": [m["overall"]], "missing": m["missing_skills"]})
        st.session_state.pop("gap", None)
    if st.session_state.get("match") and st.session_state.get("job"):
        job, m = st.session_state.job, st.session_state.match
        st.divider()
        st.subheader("Job analysis")
        render_job_analysis(job)
        st.subheader("Compatibility")
        render_match(m)
        st.button("Skill gap for this job →", on_click=go, args=("Skill Gap",))


def page_recs():
    page_header("Job Recommendations")
    r = need_resume()
    jobs, index = need_jobs()
    n = st.slider("Number of recommendations", 5, 25, 10)
    with st.spinner("Scoring all jobs…"):
        pool = recommend_jobs(r, index, top_n=60)
    if pool.empty:
        st.warning("No jobs share enough skills with this resume. Try a resume that lists more technical skills.")
        return
    st.session_state.recs = pool
    sig = ("recs", st.session_state.resume_name)
    if st.session_state.get("_rec_logged") != sig:
        history.add_record({"type": "recommendation", "resume": st.session_state.resume_name, "scores": pool.match_score.head(n).tolist(),
                            "predicted_role": recommended_roles(pool).role.iloc[0], "missing": [s for l in pool.missing_skills.head(n) for s in l]})
        st.session_state._rec_logged = sig
    recs = pool.head(n)
    st.caption("Ranked by the weighted Compatibility Score using skills, text similarity, experience and education only. "
               "No personal attributes (name, gender, age, location of the candidate) are used.")
    top = st.columns([3, 2])
    with top[0]:
        fig = px.bar(recs.iloc[::-1], x="match_score", y=recs.job_title.str[:45].iloc[::-1], orientation="h", title="Top matches", color="match_score",
                     color_continuous_scale=["#C7D2FE", C["primary"]], range_color=(0, 100))
        fig.update_layout(xaxis_title="Match score (%)", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 60 + 32 * n), width="stretch")
    with top[1]:
        roles = recommended_roles(pool)
        fig = px.bar(roles.iloc[::-1], x="jobs", y="role", orientation="h", title="Recommended roles (top 60 jobs)", color_discrete_sequence=[C["secondary"]])
        fig.update_layout(xaxis_title="Jobs", yaxis_title="")
        st.plotly_chart(style_fig(fig, 60 + 32 * n), width="stretch")
    for i, row in recs.iterrows():
        label = f"#{i + 1}  {row.job_title[:70]} — {row.company} · {row.location if pd.notna(row.location) else 'Location n/a'}  |  {row.match_score:.1f}%"
        with st.expander(label):
            a, b, c = st.columns(3)
            with a: scorecard("Match score", row.match_score)
            with b: scorecard("Skill score", row.skill_score)
            with c: scorecard("Text similarity", row.text_score)
            st.markdown("**Matching skills**"); st.markdown(chips(row.matching_skills, "ok"), unsafe_allow_html=True)
            st.markdown("**Missing skills**"); st.markdown(chips(row.missing_skills, "miss", "None - all required skills matched"), unsafe_allow_html=True)
            st.markdown(f"**Required experience:** {row.experience if pd.notna(row.experience) else 'Not specified'}  \n"
                        f"**Education:** {row.education if pd.notna(row.education) else 'Not specified'}  \n"
                        f"**Industry:** {row.industry if pd.notna(row.industry) else 'n/a'}  ·  **Pay:** {row.payrate if pd.notna(row.payrate) else 'n/a'}")
            if st.toggle("Show full job description", key=f"d{i}"):
                st.text(row.job_description[:3500])
            def pick(i=i, row=row):
                st.session_state.job = build_job(row.job_title, row.job_description, row.experience, row.education, row.company, row.location, "dataset")
                st.session_state.match = match_resume_to_job(st.session_state.resume, st.session_state.job, get_index().vec)
                st.session_state.pop("gap", None)
                go("Skill Gap")
            st.button("Analyse skill gap for this job", key=f"g{i}", on_click=pick)


def page_gap():
    page_header("Skill Gap")
    r = need_resume()
    jobs, index = need_jobs()
    job = st.session_state.get("job")
    if not job:
        st.warning("Select a job first - use **Job Matcher** or open a job from **Job Recommendations**.")
        c = st.columns(2)
        c[0].button("Go to Job Matcher", on_click=go, args=("Job Matcher",))
        c[1].button("Go to Recommendations", on_click=go, args=("Job Recommendations",))
        st.stop()
    st.subheader(f"{job['title'] or 'Pasted job description'}" + (f" — {job['company']}" if job.get("company") else ""))
    gap = analyze_gap(r["skills"], job["required_skills"], job["preferred_skills"], index, job["clean_description"])
    st.session_state.gap = gap
    if not job["required_skills"] and not job["preferred_skills"]:
        st.warning("No skills could be detected in this job description, so a skill gap cannot be computed.")
        return
    tot = len(job["required_skills"]) + len(job["preferred_skills"])
    st.progress(len(gap["matching"]) / tot, text=f"{len(gap['matching'])} of {tot} job skills found in your resume")
    a, b = st.columns(2)
    with a:
        st.markdown("### ✅ Matching skills")
        st.markdown(chips(gap["matching"], "ok", "No matching skills"), unsafe_allow_html=True)
    with b:
        st.markdown("### ❌ Missing skills")
        st.markdown("**Required**"); st.markdown(chips(gap["missing_required"], "miss", "None"), unsafe_allow_html=True)
        st.markdown("**Preferred** " + tag("inferred"), unsafe_allow_html=True); st.markdown(chips(gap["missing_preferred"], "pref", "None"), unsafe_allow_html=True)
    st.markdown("### 📚 Recommended skills to learn")
    if gap["learn"].empty:
        st.success("You already cover every skill detected for this job.")
    else:
        g = gap["learn"].head(10)
        fig = px.bar(g.iloc[::-1], x="seen_in_similar_jobs_pct", y="skill", orientation="h", color="type",
                     color_discrete_map={"Required": C["critical"], "Preferred": C["warning"]})
        fig.update_layout(xaxis_title=f"% of the {gap['n_similar_jobs']} most similar postings listing the skill", yaxis_title="")
        st.plotly_chart(style_fig(fig, 80 + 34 * len(g)), width="stretch")
        st.dataframe(gap["learn"].rename(columns={"skill": "Skill", "type": "Type", "seen_in_similar_jobs_pct": "In similar jobs (%)"}), hide_index=True, width="stretch")
    st.caption("Skills are ordered by whether the job requires them, then by how often they appear in the most similar postings of the dataset. "
               "Learning a skill does not guarantee a job offer.")


def page_eval():
    page_header("Model Evaluation")
    bundle = get_model()
    if bundle is None:
        st.warning("The role classifier has not been trained yet.")
        if st.button("Train model", type="primary"):
            with st.spinner("Training (about a minute)…"):
                need_jobs(); classifier.train_and_save()
            st.rerun()
        return
    m = bundle["meta"]
    if m["data_source"] == "naukri_weak_labels":
        st.warning("**Training data:** " + m["data_description"])
    else:
        st.success("**Training data:** " + m["data_description"])
    k = st.columns(5)
    with k[0]: kpi("Model", m["model_name"], "selected by 5-fold CV macro-F1")
    with k[1]: kpi("Accuracy", f"{m['accuracy']:.1%}", f"{m['n_test']} held-out samples")
    with k[2]: kpi("Precision (macro)", f"{m['macro']['precision']:.3f}", f"weighted {m['weighted']['precision']:.3f}")
    with k[3]: kpi("Recall (macro)", f"{m['macro']['recall']:.3f}", f"weighted {m['weighted']['recall']:.3f}")
    with k[4]: kpi("F1-score (macro)", f"{m['macro']['f1']:.3f}", f"weighted {m['weighted']['f1']:.3f}")
    if m["flagged_small_classes"]:
        st.warning(f"Few training examples for: **{', '.join(m['flagged_small_classes'])}**. Their metrics rest on very small test sets and are unreliable.")
    if m.get("dropped_classes"):
        st.error(f"Classes dropped (fewer than 10 samples): {', '.join(m['dropped_classes'])}")
    if m.get("missing_target_roles"):
        st.error(f"No examples found for project roles: {', '.join(m['missing_target_roles'])}")
    a, b = st.columns(2)
    with a:
        cm = pd.DataFrame(m["confusion_matrix"], index=m["classes"], columns=m["classes"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#EEF2FF", C["primary"]], labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (test set)")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 460), width="stretch")
    with b:
        cc = pd.Series(m["class_counts"]).sort_values()
        fig = px.bar(x=cc.values, y=cc.index, orientation="h", title="Class distribution (all samples)", color_discrete_sequence=[C["secondary"]])
        fig.update_layout(xaxis_title="Samples", yaxis_title="", xaxis_type="log")
        st.plotly_chart(style_fig(fig, 460), width="stretch")
    st.subheader("Classification report (test set)")
    rep = pd.DataFrame(m["report"]).T.round(3)
    rep["support"] = rep["support"].astype(int)
    st.dataframe(rep, width="stretch")
    st.subheader("Model comparison")
    comp = pd.DataFrame({n: {"CV macro-F1 (mean)": round(v["cv_macro_f1_mean"], 3), "CV macro-F1 (std)": round(v["cv_macro_f1_std"], 3),
                             "Test accuracy": round(m["test_accuracy_by_model"][n], 3)} for n, v in m["cv"].items()}).T
    st.dataframe(comp, width="stretch")
    st.caption(f"Stratified 80/20 split · {m['n_train']} train / {m['n_test']} test · TF-IDF (1-2 grams) · class-weight balanced.")
    if m.get("title_rules"):
        with st.expander("Weak-label rules used to create training labels"):
            for role, pat in m["title_rules"].items():
                st.markdown(f"**{role}** — `{pat}`")
    if st.button("Retrain model"):
        with st.spinner("Training…"):
            classifier.train_and_save()
        st.rerun()


def page_about():
    page_header("About Project")
    jobs = get_jobs() if discover_datasets()["jobs"] else None
    st.markdown(f"""
### Problem statement
Recruiters screen hundreds of resumes quickly, and candidates rarely know how their profile lines up with a posting. This project builds a transparent, explainable tool that parses a resume, analyses a job description, scores the compatibility and shows the skill gap.

### Objectives
Parse PDF/DOCX/TXT resumes · analyse job descriptions (required vs preferred, extracted vs inferred) · compute an interpretable Compatibility Score · classify the candidate's role · recommend jobs · analyse skill gaps · present everything in a dashboard.

### Technologies
Python · Streamlit · Pandas / NumPy · scikit-learn (TF-IDF, Logistic Regression, Linear SVM) · regex + a skill dictionary · PyMuPDF · python-docx · Plotly · joblib.

### Datasets
- **Jobs on Naukri.com** (Kaggle) - `dataset/dataset.csv`: {f"{len(jobs):,} cleaned postings (22,000 raw)" if jobs is not None else "not found"}.
- **Resume dataset** (Kaggle, Snehaan Bhawal) - {"**found**" if discover_datasets()["resumes"] else "**not provided**; the classifier therefore uses weak labels from Naukri job titles (see Model Evaluation)"}.

### ML methodology
Text → cleaning → stop-word removal → TF-IDF (unigrams+bigrams) → stratified 80/20 split → Logistic Regression vs Linear SVM compared by 5-fold CV macro-F1 → best model evaluated once on the held-out test set.

### Matching methodology
Compatibility Score = weighted average of: **Skills {SCORE_WEIGHTS['skills']:.0%}** (normalised skill names; required=1, preferred=0.5), **Text similarity {SCORE_WEIGHTS['text']:.0%}** (TF-IDF cosine), **Experience {SCORE_WEIGHTS['experience']:.0%}**, **Education {SCORE_WEIGHTS['education']:.0%}**. If a component cannot be determined it is skipped and the remaining weights are renormalised. Weights are configurable in `src/config.py`.

### Limitations
- Skills are found by dictionary matching: skills outside the dictionary are not detected; resume skills are assumed from mentions, not verified.
- No OCR: scanned/image-only PDFs cannot be read.
- Without a resume dataset the classifier is trained on rule-derived labels from job postings and applied to resumes (domain shift); Data Scientist and ML Engineer have very few examples.
- The Naukri data is from 2016-17 and is India-centric; many postings are non-IT.
- The Compatibility Score is a heuristic, not a validated hiring probability, and must not be the sole basis of hiring decisions.
""")


{"Dashboard": page_dashboard, "Resume Analyzer": page_resume, "Job Matcher": page_matcher, "Job Recommendations": page_recs,
 "Skill Gap": page_gap, "Model Evaluation": page_eval, "About Project": page_about}[page]()