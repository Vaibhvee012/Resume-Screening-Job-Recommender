# AI-Powered Resume Screening & Job Recommendation System

*College major project · Python · Streamlit · scikit-learn · NLP*

## 1. Problem statement
Recruiters skim hundreds of resumes quickly, and candidates rarely know how their profile lines up with a posting. This project builds a
transparent, explainable tool that parses a resume, analyses a job description, scores the compatibility, recommends jobs from a real
22,000-posting dataset and shows the skill gap.

## 2. Objectives
Parse PDF/DOCX/TXT resumes · analyse job descriptions (required vs preferred skills, extracted vs inferred) · compute an interpretable
**Compatibility Score** · classify the candidate's role · recommend ranked jobs · skill-gap analysis · interactive dashboard.

## 3. Features
| Page | What it does |
|---|---|
| Dashboard | KPIs, most common skills, job categories, recommended roles, match-score histogram, skill-gap chart, recent analyses - computed from the real datasets and your analysis history |
| Resume Analyzer | Upload PDF/DOCX/TXT → name/contact, skills, education, experience, keywords, predicted role. Missing fields are shown as "Not found" |
| Job Matcher | Pick a job from the dataset or paste a JD → analysis + Compatibility Score with skill/text/experience/education breakdown |
| Job Recommendations | Top-N jobs ranked by score with matching/missing skills, experience, education, full description |
| Skill Gap | Matching, missing, and *skills to learn* (ordered by demand in the 300 most similar postings) |
| Model Evaluation | Accuracy, precision, recall, F1, confusion matrix, classification report, model comparison |
| About Project | Methodology and limitations |

## 4. Architecture
```
Resume ─► resume_parser ─► text_processor ─► skill_extractor ─┐
                                                              ├─► matcher (weighted score) ─► recommender ─► skill_gap ─► app.py
Job (dataset / pasted) ─► job_analyzer ───────────────────────┘
Resume text ─► TF-IDF ─► classifier (LogReg / LinearSVM)  ─► predicted role
dataset/dataset.csv ─► data_loader (detect columns, clean) ─► data/processed/ (cached copies)
```
`src/config.py` holds paths, **score weights**, palette. Original datasets are never modified.

## 5. Technologies
Python 3.10+ · Streamlit · Pandas/NumPy · scikit-learn · regex + skill dictionary · PyMuPDF · python-docx · Plotly · Matplotlib/Seaborn (notebook) · joblib · custom CSS (`assets/style.css`, Inter font).

## 6. User interface
The Streamlit interface uses a custom design layer (`assets/style.css`) on top of the built-in theme (`.streamlit/config.toml`):
- **Dark sidebar** with brand header, icon-based navigation with a highlighted active page, a live *current resume* status card and a reset button.
- **Page headers** with an icon and a one-line description on every page.
- **KPI and score cards** with hover effect, colour-coded scores (green / amber / red) and refined skill chips (matching, missing, preferred).
- **Consistent Plotly charts** (shared font, grid, tooltip style) shown in rounded white cards.
- Restyled buttons, file uploader, expanders and tables; default Streamlit menu and footer hidden for a cleaner look.

Colours and chart styling are defined in `assets/style.css`, `src/config.py` (`PALETTE`) and `style_fig()` in `app.py`.

## 7. Datasets (external)
Both datasets are **external Kaggle datasets**, not created by this project:
- **Jobs on Naukri.com** - https://www.kaggle.com/datasets/PromptCloudHQ/jobs-on-naukricom → provided as `dataset/dataset.csv` (22,000 × 14; 21,848 after removing empty/duplicate rows).
- **Resume Dataset** (Snehaan Bhawal) - https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset → **not included in the provided files**. Drop the CSV into `dataset/` or `data/resumes/`; it is auto-detected.

Full inspection: [`docs/dataset_inspection.md`](docs/dataset_inspection.md). Note: the Naukri `skills` column holds functional areas, not skills, so skills are extracted from the descriptions.

## 8. Installation
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 9. How to run
```bash
streamlit run app.py
```
First launch analyses the job dataset (~40 s) and caches it in `data/processed/`. A trained model is included in `models/`;
to retrain: *Model Evaluation → Retrain model*, or `python -c "from src.classifier import train_and_save; train_and_save(verbose=True)"`.
Demo files are in `samples/`. Tests: `python tests/test_core.py` and `python tests/app_smoke_test.py`.

**Demo flow:** Dashboard → Resume Analyzer (load `sample_data_scientist.txt`) → Job Matcher (select/paste) → Job Recommendations → Skill Gap → Dashboard → Model Evaluation.

## 10. ML methodology
TF-IDF (1–2-grams) → Logistic Regression vs Linear SVM (balanced class weights), selected by 5-fold stratified CV macro-F1 on the training split; evaluated once on a stratified 20% test set.

**Important - training data.** No resume dataset was provided, so the classifier is trained on Naukri job postings using **weak labels**: ordered keyword rules on the job title assign postings to the six project roles (rules in `src/classifier.py`, shown in the app). Postings matching no rule are excluded, never forced into a class. The title field and Naukri's own `Role:` metadata are not used as features. When a resume CSV is added, the classifier retrains on its *original* categories automatically.

## 11. Matching methodology
`Compatibility Score = Σ wᵢcᵢ / Σ wᵢ` over available components (weights in `src/config.py`):
Skills 50% (normalised names; required = 1, preferred = 0.5) · TF-IDF cosine text similarity 25% · Experience 15% · Education 10%.
Components that cannot be determined (e.g. job states no experience) are skipped and the weights renormalised - the UI says so.
Preferred skills are *inferred* from wording like "preferred / nice to have / plus" and labelled as inferred. This is a heuristic comparison score, **not a validated probability**.

## 12. Evaluation (held-out test set, 896 samples)
| Model | CV macro-F1 | Test accuracy |
|---|---|---|
| Logistic Regression | 0.756 | 0.868 |
| **Linear SVM (selected)** | 0.757 | **0.872** |

Selected model: accuracy **0.872**, macro precision **0.761**, macro recall **0.715**, macro F1 **0.735**. Per-class results, confusion matrix and the full report are in the app and the notebook.
Training examples per role: Software Developer 3,115 · Web Developer 907 · Data Analyst 295 · Cyber Security Analyst 82 · **Data Scientist 49 · Machine Learning Engineer 30** - the last two are too small for reliable metrics (test support 10 and 6). Headline accuracy is driven by the large Software Developer class; macro-F1 is the fairer number. Metrics were not inflated or tuned on the test set.

## 13. Project structure
```
app.py  requirements.txt  README.md
assets/style.css            custom UI styling (sidebar, cards, chips, buttons)
.streamlit/config.toml      theme colours and server settings
dataset/dataset.csv         original Naukri data (untouched)
data/{jobs,resumes}/        dataset setup notes ·  data/processed/ generated caches
models/                     role_classifier.pkl, tfidf_vectorizer.pkl, label_encoder.pkl, classifier_metadata.pkl
src/                        config, data_loader, text_processor, skill_extractor, education, resume_parser,
                            job_analyzer, matcher, classifier, recommender, skill_gap, history
notebooks/                  AI_Resume_Screening_Project.ipynb (EDA → preprocessing → training → evaluation → matching experiment)
samples/ tests/ docs/
```

## 14. Limitations
- Skills come from a ~145-skill dictionary: unlisted skills are missed, and mentions are not proof of proficiency. Soft/business terms can add noise to non-IT postings.
- No OCR - scanned/image-only PDFs are rejected with a clear message.
- Classifier trained on rule-derived labels from job postings, applied to resumes (domain shift); Data Scientist / ML Engineer are tiny classes. Replace with the Kaggle resume data for a stronger result.
- Naukri data dates from 2016-17 and is India-centric; many postings are non-IT.
- Experience is estimated from date ranges relative to today's date, so old resumes with "Present" will be over-counted.
- The score must not be the sole basis of hiring decisions. Learning a skill does not guarantee a job.

## 15. Future enhancements
Add the Kaggle resume dataset and compare; contextual skill extraction (spaCy NER / embeddings); semantic similarity (sentence embeddings); OCR for scans; user-adjustable weights in the UI; export reports as PDF; dark-mode toggle for the main area.

## 16. Team contribution
| Member | Contribution |
|---|---|
| _Name 1_ | _e.g. resume parsing & NLP_ |
| _Name 2_ | _e.g. ML model & evaluation_ |
| _Name 3_ | _e.g. Streamlit UI & documentation_ |