# CampusConnect Comprehensive Dataset

## Summary
- Unique intents: 34
- Training patterns: 2714
- Prepared responses: 97
- Structured records: 131

## Source
The dataset was built from the four uploaded university JSON files. The most complete structured records are taken from `dataset_finance(1).json`, while intent patterns/responses are merged from all four uploads.

## What was expanded
1. Common FAQ paraphrases
2. Informal and short wording
3. Record-specific questions for courses, exams, timetable, fees, departments, campus places, dining, hostel, parking and events
4. Contact / location / opening-hours question variants
5. Unrelated/off-topic questions for the `unknown` intent
6. Entity dictionaries for programme names, courses, departments, campus locations and gates

## Important
The dataset does not claim to predict literally every possible future wording. It is a broad, practical training set for the supported university domain.

## Compatibility
The JSON keeps the original `intents` format and the structured sections:
`class_timetable`, `exam_schedule`, `operating_hours`, `events`, `department_contacts`, `campus_places`, and `course_fees`.

## Recommended ML use
Use the `patterns` field as training text and `tag` as the target label. Keep the structured sections as the response/data retrieval layer.
