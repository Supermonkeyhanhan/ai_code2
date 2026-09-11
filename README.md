# CampusConnect University Chatbot — Public / Engineer Access

## Access model

### Public Use
- Chatbot only
- Single Engine View only
- Algorithm selector remains inside the chatbot answer card
- Can switch Naive Bayes / SVM / LSTM for the displayed answer
- Can submit answer feedback
- Cannot access Model Evaluation, Dataset Explorer, Feedback Analytics, or Compare View
- Confidence controls and system workflow are hidden

### Engineer Use
- Chatbot
- Single Engine View
- Compare View
- Model Evaluation
- Dataset Explorer
- Feedback Analytics
- Compare View is engineer-only

## Engineer login

Default local demo password: `engineer123`

For a different password, set either:
- Streamlit secret: `engineer_password`
- Environment variable: `CAMPUSCONNECT_ENGINEER_PASSWORD`

Example on Windows PowerShell:

```powershell
$env:CAMPUSCONNECT_ENGINEER_PASSWORD="your-password"
streamlit run app.py
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
