# CampusConnect University Chatbot

## Answer display modes

### Compare View
One user question is processed independently by:
- Naive Bayes
- SVM
- LSTM

All three model cards are shown together with intent, confidence, answer and response-type support.

### Single Engine View
The same question is evaluated by all three models, but only one answer is shown. The algorithm selector is embedded inside the answer card, so the user can switch between Naive Bayes, SVM and LSTM without retyping the question.

## Visual response decision

After each model predicts an intent, a response-planning layer decides whether the answer should be:
- `image` for location-oriented intents such as library, campus location, parking, hostel and dining;
- `table` for structured information such as fees, timetable, exams and office hours;
- `contact` for department contact questions;
- `text` for general conversational or informational intents.

The visual decision is based on the predicted intent, not on a separate image-generation ML model. This keeps the system deterministic and easy to explain for an ML intent-classification assignment.

Location visuals are illustrative SVG maps rendered directly inside `app.py` and are explicitly labelled as not official campus maps. They are intended to demonstrate the multimodal-response concept. Replace them with approved university images/maps before claiming them as official.

## UI visibility

The sidebar contains only:
- Navigation
- Answer Display

The confidence-control section, system workflow navigation, and detailed dataset inspection controls are hidden from the main UI.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Campus map PDF

The project includes `assets/campus_map.pdf`, the map supplied for this chatbot. Location/map questions can automatically display this PDF alongside the model answer.

Examples include library location, campus location, parking, hostel, dining locations, and questions that ask where a recognized campus place/department/gate is located.
