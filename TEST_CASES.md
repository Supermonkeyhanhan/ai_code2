# Suggested Test Questions

Use the same questions across Naive Bayes, SVM and LSTM so the comparison is fair.

| # | Test question | Expected intent |
|---|---|---|
| 1 | Where is the university library? | library_location |
| 2 | What time does the library open? | library_hours |
| 3 | When is the Data Structures exam? | exam |
| 4 | Where can I pay my tuition fee? | tuition_fee |
| 5 | How much is the Diploma in Computer Science? | course_fee_inquiry |
| 6 | Where can I find the FOCS faculty? | campus_location |
| 7 | How do I reset my university password? | password_problem |
| 8 | My campus Wi-Fi is not working. | wifi_problem |
| 9 | Who handles scholarship applications? | scholarship |
| 10 | I need counselling support. | counselling_booking |
| 11 | What are the class times for Data Structures? | timetable |
| 12 | Where can I park my car? | parking |
| 13 | How do I borrow a library book? | borrowing_books |
| 14 | Where can I get lunch on campus? | campus_dining |
| 15 | Who do I contact about admission requirements? | admission_requirements / department_contact depending on wording |

## Evaluation protocol

1. Select Naive Bayes and submit all questions.
2. Record predicted intent and confidence.
3. Repeat with SVM.
4. Repeat with LSTM.
5. Compare Accuracy, Precision, Recall and F1 Score on the Model Evaluation page.
6. Use the same questions and same dataset for all models.
