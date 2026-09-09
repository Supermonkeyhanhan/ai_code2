# CampusConnect University Chatbot

## 1. Project Overview

This is an ML-based **University FAQ Chatbot** implemented with Python and Streamlit.

The application compares three intent-classification approaches:

1. **Naive Bayes** — TF-IDF + Multinomial Naive Bayes
2. **SVM** — TF-IDF + linear Support Vector Machine
3. **LSTM** — tokenization + word embeddings + bidirectional LSTM

All three models use the same labelled intent dataset and the same 80/20 train-test split for a fair comparison.

## 2. System Workflow

```text
University Chatbot
        ↓
   Streamlit UI
        ↓
Select Engine + Question
        ↓
 ┌──────┼──────┐
 ↓      ↓      ↓
 NB     SVM    LSTM
 ↓      ↓      ↓
   Intent Classification
          ↓
 Decision / Structured Data Logic
          ↓
 responses.json + structured dataset
          ↓
      Chatbot Answer
          ↓
     Streamlit UI
```

## 3. Dataset

`dataset.json` is built from the user-provided university datasets.

The prepared dataset contains:
- 34 unique intents
- 1,392 training patterns
- 95 prepared response entries across intents
- Structured records for timetable, exams, operating hours, events, department contacts, campus places and course fees.

The project also preserves the original user-provided files under `source_data/`.

### Dataset cleaning performed

Two source datasets contained overlapping/duplicated intents and several patterns assigned to conflicting intents. The prepared `dataset.json`:
- merges the two main sources,
- removes exact duplicate patterns/responses, and
- reassigns eight conflicting patterns to the more specific intent so the ML training labels are not contradictory.

The exact source files remain under `source_data/` for traceability.

## 4. Functionalities

### Chatbot
- Choose Naive Bayes, SVM or LSTM.
- Enter a question.
- Show predicted intent.
- Show confidence.
- Show alternative intent predictions.
- Retrieve prepared responses.
- Use structured data for timetable, exam, fees, campus places and department contacts.
- Collect thumbs-up / thumbs-down feedback during the session.

### Model Evaluation
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Macro F1
- Confusion matrix
- Per-intent Precision / Recall / F1

### Dataset Explorer
- Intent count
- Pattern count
- Response count
- Intent examples
- Structured data browser

### System Workflow
- Shows the system architecture used in the project.

## 5. Installation

Recommended: Python 3.11 or newer.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Run the Chatbot

```bash
streamlit run app.py
```

## 7. Optional Command-Line Evaluation

```bash
python train_models.py
```

This writes `evaluation_summary.json`.

## 8. Notes for the Assignment

This project follows **Option 1: Build the chatbot using machine learning techniques**.

### Preprocessing
- lowercasing
- punctuation cleanup
- whitespace normalization
- tokenization for LSTM
- TF-IDF feature extraction for Naive Bayes and SVM

### Intent Classification
The model predicts one supported university intent.

### Response Layer
The chatbot primarily uses **response retrieval** from `responses.json`, while selected intents use structured university data for more specific answers. This is intentionally controlled rather than free-form response generation.

### Evaluation
The application computes Precision, Recall, F1 Score and Accuracy using the same held-out test set for all three models.

Do not report model performance numbers until you run the application and record the actual metrics shown on your run.

## 9. Before Submission

Replace any placeholder/illustrative university information in the dataset with the latest official information required by your lecturer. The current dataset supplied with this project is the source provided for this development exercise and should be verified before claiming it as official university information.

## 10. Interpretation of the three models

A traditional TF-IDF + SVM classifier can perform strongly on FAQ-style intent classification because the training data is phrase-rich and labelled by intent. Naive Bayes serves as a lightweight probabilistic baseline. LSTM is included to satisfy the neural-network comparison requirement, but its performance may be lower on a relatively small, domain-specific FAQ dataset.

The included `evaluation_summary.json` is a **sample build-environment run**, not a result that should automatically be copied into a report. Re-run `python train_models.py` and use the new metrics from your own environment.

## 11. BLEU / ROUGE

The response layer is retrieval-based (`responses.json`) rather than free-form text generation. Because there can be multiple valid prepared responses for an intent, BLEU/ROUGE is not used as the primary metric in this implementation. Intent-classification metrics plus in-session user feedback are more directly aligned with the implemented response mechanism.


## CSS setup

The Streamlit CSS is embedded directly inside `app.py`. No separate `styles.css` file is required. The app injects the CSS through a `<style>` block so the stylesheet is not rendered as visible page text.
