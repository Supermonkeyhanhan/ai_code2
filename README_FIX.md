# KeyError: SVM fix

This version fixes a Streamlit crash that occurred when an Engineer switched to Compare View while viewing an older message created in Public Mode.

Public Mode intentionally stores only the Naive Bayes result. Older Public messages therefore do not contain SVM/LSTM result keys. Compare View now detects partial historical results and renders the available result instead of indexing `results['SVM']` or `results['LSTM']` directly.

New questions submitted in Engineer Mode still run all three models and render the full comparison.
