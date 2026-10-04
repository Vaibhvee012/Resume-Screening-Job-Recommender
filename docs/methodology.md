# Methodology

## Pipeline
Resume (PDF/DOCX/TXT) -> text extraction (PyMuPDF / python-docx) -> cleaning -> section split -> regex + dictionary extraction
(contact, degrees, experience years, skills, keywords).
Job -> boilerplate removal -> required skills (dictionary) / preferred skills (inferred from "preferred / nice to have / plus ..." wording)
-> experience and education parsing.

## Compatibility score (src/config.py)
`score = sum(w_i * c_i) / sum(w_i)` over available components: Skills 50% (required 1.0, preferred 0.5, normalised names),
Text similarity 25% (TF-IDF cosine, vectoriser fitted on the job corpus; cosine >= 0.5 counts as 100%), Experience 15%
(candidate/required years, capped at 1), Education 10% (1 / 0.5 if one level short / 0). Unavailable components are skipped and weights
renormalised (the UI says so). Recommendations additionally require >= 3 matched skills so one-skill postings cannot score 100%.
Not a validated probability.

## Classifier
TF-IDF (1-2 grams, sublinear) -> Logistic Regression vs Linear SVM (class_weight=balanced); 5-fold stratified CV macro-F1 on the
training split chooses the model; one evaluation on the stratified 20% test set.

**Weak labels (no resume dataset):** ordered title rules (see `src/classifier.py: TITLE_RULES`) assign Naukri jobs to the six roles;
non-matching titles are excluded. Counts: Software Developer 3,115 - Web Developer 907 - Data Analyst 295 - Cyber Security 82 -
Data Scientist 49 - Machine Learning Engineer 30. Last two are flagged as too small for reliable metrics.
Title field and Naukri's `Role:` metadata are not features.

## Fairness
Only skills, experience, education and text similarity are used; name/contact details are removed from the text used for similarity.
