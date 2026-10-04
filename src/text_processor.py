"""Text cleaning and NLP preprocessing (regex + scikit-learn stop words; no downloads needed)."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS) | {"yrs", "yr", "year", "years", "etc", "including", "work",
                                       "experience", "candidate", "job", "company", "required", "using", "built", "developed", "implemented",
                                       "deployed", "created", "worked", "used", "responsible", "performed", "applied", "wrote", "collaborated",
                                       "maintained", "analyzed", "good", "strong", "knowledge", "ability", "skills", "role", "team", "title", "location", "description", "looking", "join", "talented"}
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"(\+?\d[\d\-\s()]{8,}\d)")
_NONWORD = re.compile(r"[^a-z0-9+#.\s]")      # keep c++, c#, .net, node.js style tokens
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, remove URLs/emails/phones/noise, normalise whitespace."""
    if not isinstance(text, str):
        return ""
    t = text.replace("\xa0", " ").lower()
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _PHONE.sub(" ", t)
    t = _NONWORD.sub(" ", t)
    t = re.sub(r"(?<![a-z0-9])\.+|\.+(?![a-z0-9+#])", " ", t)   # strip stray dots, keep "node.js"
    return _SPACES.sub(" ", t).strip()


def tokenize(text: str) -> list:
    return [w for w in clean_text(text).split() if len(w) > 1]


def remove_stopwords(tokens: list) -> list:
    return [w for w in tokens if w not in STOPWORDS and not w.isdigit()]


def _stem(w: str) -> str:
    """Very light normaliser (plural / -ing / -ed); avoids heavy NLTK data downloads."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)]
    return w


def preprocess(text, stem: bool = True) -> str:
    """Full pipeline: clean -> tokenize -> stopwords -> light normalisation."""
    toks = remove_stopwords(tokenize(text))
    if stem:
        toks = [_stem(w) for w in toks]
    return " ".join(toks)
