# Resume dataset (optional but recommended)

**Kaggle Resume Dataset** by Snehaan Bhawal - https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Place the CSV (columns such as `Resume_str` and `Category`) in this folder or in `dataset/`.
It is auto-detected (columns `Resume_str/resume_text/resume/text` + `Category/label`).
When present, the role classifier is retrained on the real resume categories (original categories are kept and never
renamed) instead of the Naukri weak labels: open **Model Evaluation → Retrain model**, or run
`python -c "from src.classifier import train_and_save; train_and_save(verbose=True)"`.

**This file was NOT part of the provided project files.**
