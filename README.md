# CampusPulse AI

**AI-Powered Student Feedback Intelligence**

CampusPulse AI is a student feedback analysis prototype that uses natural language processing and machine learning to classify feedback sentiment, summarize topic-level patterns, identify potential complaint themes, and suggest areas for human review.

## Problem Statement

Educational institutions receive feedback about teaching, course content, examinations, laboratory work, library facilities, and extracurricular activities. Manually reviewing large volumes of comments can make it difficult to identify recurring concerns.

CampusPulse AI helps organize feedback into interpretable summaries so that administrators can investigate potential issues more efficiently.

## Key Features

- Sentiment classification: positive, neutral, and negative.
- Topic-wise feedback summaries across six education areas.
- Interactive charts and feedback tables.
- Rule-based complaint-theme detection.
- Suggested improvement actions for human review.
- CSV upload and export.
- Model evaluation metrics.
- Automated tests for prediction and theme-detection behavior.

## Technology Stack

- Python
- Pandas and NumPy
- scikit-learn
- TF-IDF and Logistic Regression
- Streamlit
- Plotly
- Joblib
- pytest

## Project Structure

```text
CampusPulse AI/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── sentiment_pipeline.joblib
├── reports/
│   └── model_metrics.json
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── topic_analysis.py
│   ├── complaint_analysis.py
│   ├── complaint_themes.py│   ├── evaluation.py
│   └── inspect_data.py
│   └── recommendations.py
├── tests/
│   └── test_predictions.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare the processed dataset:

```powershell
python src/preprocessing.py
```

Train and evaluate the model if the saved model is not available:

```powershell
python src/train.py
```

Run the dashboard:

```powershell
python -m streamlit run app.py
```

Run automated tests:

```powershell
python -m pytest -v
```

## Model Evaluation

The current saved evaluation report records:

- Test accuracy: approximately 86.01%
- Macro F1 score: approximately 0.8105

These are results from the current held-out test split. They are preliminary and do not guarantee performance on new institutions, new datasets, or every individual comment.

## Complaint-Theme Analysis

Complaint themes are currently detected with transparent keyword rules. A comment can match multiple themes, and unmatched comments are flagged for review. Theme counts are not independently verified complaint classifications.

Suggested actions are generated from predefined rules and should be reviewed by people familiar with the institution and its context.

## Limitations and Responsible Use

- Sentiment predictions can be incorrect, especially for ambiguous, sarcastic, or context-dependent comments.
- Dataset labels and topic coverage affect the quality of the results.
- Keyword-based themes can produce false matches or miss relevant comments.
- Feedback counts do not necessarily represent unique students.
- Negative-feedback percentages indicate patterns in the available data, not proven causes.
- The system supports human review and should not be used as the sole basis for decisions about students, teachers, or staff.
- Uploaded feedback should be handled with appropriate privacy safeguards.

## Future Scope

- Evaluate on larger, independently labeled datasets.
- Improve complaint-theme classification and measure its precision and recall.
- Add more robust validation for uploaded files.
- Collect feedback from human reviewers to improve theme quality.
- Evaluate generalization across different institutions.

## Project Status

CampusPulse AI is a working prototype under development. The current automated test suite contains 13 passing tests.
