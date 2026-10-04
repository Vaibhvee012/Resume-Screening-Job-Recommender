"""Job-role classifier: TF-IDF -> Logistic Regression / Linear SVM, selected by cross-validated macro-F1.

TRAINING DATA
 * If a labelled resume dataset is found (dataset/ or data/resumes/), it is used with its ORIGINAL categories.
 * Otherwise (the case for this repo's provided files) the classifier is trained on the Naukri jobs using
   WEAK LABELS derived from job-title keywords for the six project roles. This is documented everywhere it is shown."""
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from .config import MIN_SAMPLES_PER_CLASS, MODELS_DIR, RANDOM_STATE, TARGET_ROLES, TEST_SIZE
from .text_processor import preprocess

# Ordered: first match wins (specific roles before generic ones). Documented in README / About page.
TITLE_RULES = [
    ("Cyber Security Analyst", r"cyber|information security|infosec|security analyst|network security|(?<!sap )security engineer|penetration|soc analyst|vapt|ethical hack|it security|security architect|security consultant"),
    ("Data Scientist", r"data scien"),
    ("Machine Learning Engineer", r"machine learning|\bml\b|deep learning|computer vision|\bnlp\b|\bai\b|artificial intelligence"),
    ("Data Analyst", r"data analy|business analy|\bbi\b|business intelligence|mis (?:analyst|executive)|reporting analyst|analytics (?:analyst|consultant|engineer|manager)|risk analytics"),
    ("Web Developer", r"web develop|php|front[\s-]?end|full[\s-]?stack|ui develop|wordpress|angular|react|node\.?js|javascript|web application|drupal|magento|html|asp\.net|web programmer"),
    ("Software Developer", r"software (?:engineer|developer|programmer)|programmer|java|\.net|python|c\+\+|\bsde\b|application developer|developer|android|\bios\b|mobile app"),
]
_RULES = [(r, re.compile(p, re.I)) for r, p in TITLE_RULES]
FILES = {"model": "role_classifier.pkl", "vec": "tfidf_vectorizer.pkl", "enc": "label_encoder.pkl", "meta": "classifier_metadata.pkl"}


def weak_label(title: str):
    for role, rx in _RULES:
        if rx.search(title or ""):
            return role
    return None


def _job_training_text(row) -> str:
    """Description part only: Naukri's own Role/Salary metadata and the title FIELD are excluded
    (anti-leakage); role words that naturally appear inside the description text are kept, as in real resumes."""
    t = row.clean_description.split("Salary:")[0]
    kw = re.search(r"Key\s?skills\s*(.*?)(?=Desired Candidate Profile|Education\s*-|$)", row.clean_description, re.I | re.S)
    t = t + " " + (kw.group(1) if kw else "")
    return t


def build_training_data():
    """Returns (texts, labels, info dict). Uses a resume dataset when available, else weak-labelled jobs."""
    from .data_loader import load_resume_dataset, load_jobs
    res, path = load_resume_dataset()
    if res is not None:
        return res.resume_text.map(preprocess), res.category, {
            "source": "resume_dataset", "description": f"Labelled resume dataset ({path.name}), original categories kept.",
            "n_raw": len(res)}
    jobs = load_jobs()
    jobs = jobs[jobs.is_valid].copy()
    jobs["label"] = jobs.job_title.map(weak_label)
    lab = jobs.dropna(subset=["label"])
    texts = lab.apply(_job_training_text, axis=1).map(preprocess)
    return texts, lab.label.reset_index(drop=True), {
        "source": "naukri_weak_labels",
        "description": ("No resume dataset was provided. Trained on Naukri job postings, labelled by job-title keyword rules "
                        "for the six project roles (title field and Naukri 'Role' metadata excluded from the text). "
                        "Labels are rule-derived, not human-annotated, and the model is applied to resumes (domain shift)."),
        "n_raw": len(jobs)}


def train_and_save(verbose: bool = False) -> dict:
    texts, labels, info = build_training_data()
    texts, labels = pd.Series(texts).reset_index(drop=True), pd.Series(labels).reset_index(drop=True)
    counts = labels.value_counts()
    too_small = counts[counts < 10].index.tolist()          # cannot be split/stratified at all
    keep = ~labels.isin(too_small)
    texts, labels = texts[keep].reset_index(drop=True), labels[keep].reset_index(drop=True)
    counts = labels.value_counts()
    flagged = counts[counts < MIN_SAMPLES_PER_CLASS].index.tolist()
    enc = LabelEncoder().fit(labels)
    y = enc.transform(labels)
    Xtr_t, Xte_t, ytr, yte = train_test_split(texts, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=40000, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(Xtr_t), vec.transform(Xte_t)
    candidates = {"Logistic Regression": LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
                  "Linear SVM": LinearSVC(C=1.0, class_weight="balanced")}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv = {}
    for name, m in candidates.items():
        s = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1_macro")
        cv[name] = {"cv_macro_f1_mean": float(s.mean()), "cv_macro_f1_std": float(s.std())}
        if verbose:
            print(name, cv[name])
    best = max(cv, key=lambda k: cv[k]["cv_macro_f1_mean"])
    model = candidates[best].fit(Xtr, ytr)
    pred = model.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
    test_by_model = {}
    for name, m in candidates.items():
        mm = m if name == best else m.fit(Xtr, ytr)
        test_by_model[name] = float(accuracy_score(yte, mm.predict(Xte)))
    meta = {"model_name": best, "data_source": info["source"], "data_description": info["description"],
            "classes": list(enc.classes_), "cv": cv, "test_accuracy_by_model": test_by_model,
            "n_samples": int(len(y)), "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "accuracy": float(accuracy_score(yte, pred)),
            "macro": {"precision": p, "recall": r, "f1": f}, "weighted": {"precision": pw, "recall": rw, "f1": fw},
            "report": classification_report(yte, pred, target_names=enc.classes_, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "flagged_small_classes": flagged, "dropped_classes": too_small,
            "missing_target_roles": [r for r in TARGET_ROLES if r not in enc.classes_ and info["source"] == "naukri_weak_labels"],
            "title_rules": {r: p for r, p in TITLE_RULES} if info["source"] == "naukri_weak_labels" else None}
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / FILES["model"])
    joblib.dump(vec, MODELS_DIR / FILES["vec"])
    joblib.dump(enc, MODELS_DIR / FILES["enc"])
    joblib.dump(meta, MODELS_DIR / FILES["meta"])
    return meta


class ModelNotTrained(Exception):
    pass


def load_model():
    try:
        return {k: joblib.load(MODELS_DIR / FILES[k]) for k in FILES}
    except FileNotFoundError:
        raise ModelNotTrained("Role classifier has not been trained yet. Open 'Model Evaluation' and click 'Train model'.")


def predict_roles(resume_text: str, bundle: dict = None, top_k: int = 3):
    """Returns [(role, relative_score 0-1)] sorted. Scores are softmax-normalised model outputs,
    useful for ranking - not calibrated probabilities."""
    b = bundle or load_model()
    X = b["vec"].transform([preprocess(resume_text)])
    m = b["model"]
    if hasattr(m, "predict_proba"):
        s = m.predict_proba(X)[0]
    else:
        d = m.decision_function(X)[0]
        e = np.exp(d - d.max())
        s = e / e.sum()
    idx = np.argsort(s)[::-1][:top_k]
    return [(b["enc"].classes_[i], float(s[i])) for i in idx]
