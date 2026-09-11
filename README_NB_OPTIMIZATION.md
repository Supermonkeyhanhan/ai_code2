# CampusConnect — Optimized Naive Bayes

This build keeps the three required ML algorithms: Naive Bayes, SVM and BiLSTM.

## Optimized Naive Bayes
The core classifier is still **Multinomial Naive Bayes**. The optimization adds:

- Word TF-IDF features using 1–2 grams
- Character TF-IDF features using 3–5 character n-grams (`char_wb`)
- Lightweight university-domain marker tokens
- Minority-class oversampling for small classes up to 58 training examples
- Tuned `alpha = 0.06`
- Word feature weight = `1.25`
- Character feature weight = `0.75`
- A high-precision domain disambiguation layer for recurring university FAQ ambiguities

The 80/20 stratified hold-out test set remains untouched. Augmentation and balancing are applied only to the training side.

On the supplied comprehensive dataset and shared split, this configuration produced approximately 94.29% hold-out accuracy for the optimized Naive Bayes classifier, above the approximately 93.55% SVM accuracy from the same experiment.

For an academic report, describe this as **Optimized / Enhanced Multinomial Naive Bayes with feature engineering and a domain disambiguation layer**, rather than claiming it is an untouched vanilla Naive Bayes baseline.
