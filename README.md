# CampusConnect University Chatbot

An ML-based University FAQ Chatbot built with Python and Streamlit.

## Models

1. Naive Bayes — TF-IDF + Multinomial Naive Bayes
2. SVM — TF-IDF + linear Support Vector Machine
3. LSTM — tokenization + embedding + bidirectional LSTM

All three models are trained from the same `dataset.json` and evaluated on the same stratified 80/20 hold-out split.

## Main features

- Streamlit user interface
- Naive Bayes / SVM / LSTM model selector
- Intent classification
- Confidence score and Top-3 alternatives
- Confidence-based fallback for uncertain questions
- Top-1 vs Top-2 margin check
- Simple entity extraction for course, programme, level, department, place, course code and gate
- Structured data retrieval for fees, timetable, exams, operating hours, events, departments and campus locations
- `responses.json` controlled response layer
- Same-question comparison of all three models
- Accuracy, Precision, Recall, weighted F1, Macro F1
- Confusion matrix
- Per-intent metrics
- LSTM training curves
- Standard hold-out evaluation plus unseen challenge-set evaluation
- Dataset quality checks and class-balance visualization
- Persistent thumbs-up / thumbs-down feedback in `feedback.csv`
- Feedback analytics page
- CSS embedded directly in `app.py`

## Dataset

`dataset.json` is the prepared dataset derived from the user-provided University chatbot datasets. The original source files are preserved under `source_data/`.

The prepared dataset contains 34 unique intents, 1,392 training patterns, 95 prepared response entries and structured records covering timetable, exams, operating hours, events, department contacts, campus places and course fees.

Before submission, verify any university-specific factual information against the current official university sources.

## Run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Command-line evaluation

```bash
python train_models.py
```

This writes a fresh `evaluation_summary.json` using the actual run.

Run the smoke test:

```bash
python test_engine.py
```

## Project structure

```text
university_chatbot_complete/
├── app.py
├── engine.py
├── dataset.json
├── responses.json
├── challenge_test.json
├── train_models.py
├── test_engine.py
├── evaluation_summary.json
├── feedback.csv              # created after users rate answers
├── requirements.txt
├── README.md
├── TEST_CASES.md
├── DATASET_NOTES.md
└── source_data/
    ├── dataset_1_original.json
    ├── dataset_finance_original.json
    ├── dataset_general_original.json
    ├── dataset_library_original.json
    └── dataset_tour_original.json
```

## Assignment mapping

This project follows Option 1: build the chatbot using machine learning techniques.

### Preprocessing

- lowercase text
- punctuation cleanup
- whitespace normalization
- tokenization for LSTM
- TF-IDF word and character features for Naive Bayes and SVM

### Intent classification

The three models classify the user's question into one of the supported university intents.

### Response layer

The application primarily uses controlled response retrieval. Selected intents use structured data lookup so the answer can be specific to a course, fee, timetable, exam, department or campus place.

### Evaluation

The application reports Accuracy, Precision, Recall, weighted F1, Macro F1, confusion matrices, per-intent metrics, training time, inference time and challenge-set performance.

The chatbot also collects user feedback for a simple usability signal.

### BLEU / ROUGE note

The implemented response layer is retrieval-based rather than free-form generative text. Therefore BLEU and ROUGE are not used as the primary metrics. Intent-classification metrics and user feedback align more directly with the implementation.
