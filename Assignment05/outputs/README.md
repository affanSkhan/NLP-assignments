# Assignment 05 generated outputs

Run the Assignment 05 notebook or Python source to create these files:

- dataset_metadata.json - source counts, deduplication and class distribution
- split_manifest.csv - training/test assignment for each record ID
- test_predictions.csv - gold label, TF-IDF prediction/probability, VADER sentiment scores
- error_analysis.csv - records where either model is wrong
- model_comparison.csv - accuracy, macro precision/recall/F1, positive-class metrics and neutral count
- classification_report_logreg.csv and classification_report_vader.csv
- confusion_matrix_logreg.csv and confusion_matrix_vader.csv
- metrics.json and results.txt
- vader_rule_examples.csv - illustrative rule-sensitive examples
- model_accuracy_comparison.png
- tfidf_logistic_regression.joblib

The data files are generated locally and are not committed because they contain third-party tweet text. The workspace used to author this repository could not download the NLTK corpus, so no numerical results are committed before an actual run.
