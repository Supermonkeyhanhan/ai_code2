# Test Cases

## Normal university questions

| # | Question | Expected area |
|---|---|---|
| 1 | Where is the library? | library_location |
| 2 | What time does the library close? | library_hours |
| 3 | How do I borrow a library book? | borrowing_books |
| 4 | What courses are available? | course_information |
| 5 | Recommend a programming subject | course_recommendation |
| 6 | How much is a diploma course? | course_fee_inquiry |
| 7 | Where can I pay my tuition fee? | tuition_fee |
| 8 | When is my Data Structures exam? | exam |
| 9 | Where can I check my timetable? | timetable |
| 10 | How do I reset my password? | password_problem |
| 11 | Why is campus Wi-Fi not working? | wifi_problem |
| 12 | How do I apply for the hostel? | hostel |
| 13 | Where can I print my assignment? | printing |
| 14 | Who should I contact about finance? | department_contact |
| 15 | Are there any campus events? | campus_events |

## Unknown / fallback tests

These should either classify as `unknown` or trigger the confidence-based fallback depending on model probabilities:

- What's the weather today?
- Tell me a joke.
- Recommend a movie.
- Who won the football match?
- Give me a pizza recipe.

## Comparison test

Enter the same question with each engine selected, then compare:

- predicted intent
- confidence
- fallback decision
- top alternatives

The Model Evaluation page should be used for the formal held-out metrics.
