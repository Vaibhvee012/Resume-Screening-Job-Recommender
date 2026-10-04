# Dataset inspection report

## Files found
| File | Type | Rows x Cols | Role |
|---|---|---|---|
| `dataset/dataset.csv` (52 MB) | Naukri jobs | 22,000 x 14 | **Job dataset** |
| (none) | Resume dataset | - | **Not provided** |

## Job dataset columns -> internal schema
| Raw column | Internal | Notes |
|---|---|---|
| jobtitle | job_title | 17,564 unique; some junk titles (phone numbers, "WhatsApp") flagged `is_valid=False` |
| jobdescription | job_description | 4 missing; boilerplate ("Send me Jobs like this", "Download PPT...") removed; company-profile paragraph cut |
| skills | skills | **Not real skills** - only 45 unique functional-area labels ("ITES", "Accounts"). Skills are extracted from the text instead |
| experience | experience -> exp_min / exp_max | "2 - 7 yrs" parsed |
| education | education -> edu_level | "UG: B.Tech/B.E. PG:M.Tech"; 1,996 missing |
| company, joblocation_address, industry, payrate | company, location, industry, payrate | location normalised (Bangalore/Bengaluru, first city of multi-city values) |
| uniq_id | job_id | unique |
| numberofpositions, postdate, site_name, jobid | (unused) | 80% / 0.1% / 82% missing / low value |

## Missing values (raw)
education 1,996 - joblocation_address 501 - skills 528 - payrate 97 - postdate 23 - industry 5 - company 4 - experience 4 - jobdescription 4 - numberofpositions 17,536 - site_name 18,013.

## Duplicates
0 fully identical rows; 90 repeated `jobid`; 148 repeated (title, company, description) -> removed.
Result: **21,848 cleaned postings**, of which **21,474 valid** for recommendation.

## Resume dataset
Not provided. See `data/resumes/README.md`. Until it is added, the classifier uses weak labels (see `methodology.md`).
Previous `models/*.pkl` in the upload were trained on synthetic data (`data_source: 'synthetic'`, 100% accuracy) and were **deleted** and replaced.
