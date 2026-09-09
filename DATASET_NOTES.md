# Dataset Preparation Notes

Prepared dataset statistics:
- Intents: 34
- Patterns: 1392
- Responses: 95

Source:
- dataset(1).json
- dataset_finance.json
- dataset_general.json
- dataset_library.json
- dataset_tour.json

The three files `dataset_general.json`, `dataset_library.json`, and `dataset_tour.json` are duplicate copies of the same 33-intent dataset in the supplied files, so they are preserved as source files but not counted multiple times during training.

The primary merge uses `dataset(1).json` and `dataset_finance.json`; the latter contains the richer structured records and the additional library/dining intents.
