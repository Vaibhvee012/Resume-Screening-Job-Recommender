# Job dataset

**Jobs on Naukri.com** (Kaggle) - https://www.kaggle.com/datasets/PromptCloudHQ/jobs-on-naukricom

The project reads it from `dataset/dataset.csv` (any `.csv/.xlsx/.parquet` in `dataset/`, `data/jobs/` or `data/resumes/`
is auto-detected by its columns - file names are not hard-coded). Required columns: job title + job description
(`jobtitle`, `jobdescription`). Optional: `skills, experience, education, company, joblocation_address, industry, payrate`.

The original file is never modified. Cleaned copies are written to `data/processed/`.
