# Dataset Notes

The prepared `dataset.json` is based on the user-provided University chatbot datasets.

## Prepared data

- 34 unique intents
- 1,392 training patterns
- 95 response entries
- structured timetable records
- structured exam records
- structured operating-hour records
- structured event records
- structured department contacts
- structured campus places
- structured course fee records

## Cleaning

Exact duplicate patterns were removed during preparation, and conflicting pattern labels were resolved so a normalized pattern belongs to one intent.

The original user-provided JSON files remain under `source_data/` for traceability.

## Important

University-specific factual information such as dates, fees, opening hours, office contacts and locations should be checked against the latest official university sources before being presented as authoritative in a submitted report or real deployment.
