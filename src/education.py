"""Education-level detection shared by the resume parser and job analyzer.
Levels: 0 = no requirement, 1 = diploma / 12th, 2 = bachelor, 3 = master, 4 = doctorate."""
import re

LEVEL_NAMES = {0: "No requirement", 1: "Diploma / 12th", 2: "Bachelor's", 3: "Master's", 4: "Doctorate"}
_E = r"(?=[\s,.)\-/(]|$)"      # right boundary
# (level, regex, case_sensitive) - abbreviations like B.E / M.E are case-sensitive to avoid "be"/"me"
_PATTERNS = [
    (4, r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b", False),
    (3, r"\bm\.?\s?tech\b|\bmca\b|\bm\.?\s?sc\b|\bmba\b|\bpgdm\b|\bm\.?\s?com\b|\bmaster'?s?\b|\bpost[\s-]?graduat(?:e|ion)\b|\bm\.?\s?phil\b|\bmca\b", False),
    (3, r"\bM\.?E\.?" + _E, True),
    (2, r"\bb\.?\s?tech\b|\bbca\b|\bb\.?\s?sc\b|\bb\.?\s?com\b|\bbba\b|\bbachelor'?s?\b|\bgraduat(?:e|ion)\b|\bb\.?\s?arch\b|\bundergraduate\b", False),
    (2, r"\bB\.?E\.?" + _E, True),
    (1, r"\bdiploma\b|\b12th\b|\bhsc\b|\bhigher secondary\b|\bpolytechnic\b|\bintermediate\b", False),
]
_COMPILED = [(lvl, re.compile(p, 0 if cs else re.I)) for lvl, p, cs in _PATTERNS]
_DEGREE_RX = re.compile(
    r"(Ph\.?\s?D\.?|M\.?\s?Tech|B\.?\s?Tech|MBA|MCA|BCA|M\.?\s?Sc\.?|B\.?\s?Sc\.?|M\.?\s?Com|B\.?\s?Com|BBA|PGDM|"
    r"M\.?E\.?|B\.?E\.?|Bachelor(?:'s)?(?: of [A-Za-z ]+)?|Master(?:'s)?(?: of [A-Za-z ]+)?|Diploma(?: in [A-Za-z ]+)?)")


def education_level(text) -> int | None:
    """Highest level mentioned in text, or None if nothing recognisable."""
    if not isinstance(text, str) or not text.strip():
        return None
    best = None
    for lvl, rx in _COMPILED:
        if rx.search(text):
            best = lvl if best is None else max(best, lvl)
    return best


def job_min_education(text) -> int | None:
    """Minimum education a Naukri-style string requires, e.g. 'UG: B.Tech/B.E. PG:M.Tech'.
    UG entry is the entry-level requirement; PG/Doctorate entries are alternatives, not extra requirements."""
    if not isinstance(text, str) or not text.strip():
        return None
    ug = re.search(r"UG\s*:\s*(.*?)(?=PG\s*:|Doctorate\s*:|$)", text, re.I | re.S)
    if ug:
        seg = ug.group(1)
        if re.search(r"not required", seg, re.I):
            return 0
        return education_level(seg) or 2        # "UG: Any Graduate" -> bachelor
    pg = re.search(r"PG\s*:\s*(.*?)(?=Doctorate\s*:|$)", text, re.I | re.S)
    if pg and not re.search(r"not required", pg.group(1), re.I):
        return 3
    return education_level(text)


def extract_degrees(text: str, limit: int = 6) -> list:
    """Degree strings as written in a resume (for display), de-duplicated."""
    seen, out = set(), []
    for m in _DEGREE_RX.finditer(text or ""):
        d = re.sub(r"\s+", " ", m.group(1)).strip()
        if d.lower() not in seen:
            seen.add(d.lower())
            out.append(d)
    return out[:limit]
