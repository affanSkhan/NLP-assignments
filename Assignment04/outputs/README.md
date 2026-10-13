Generated files appear after the pre-trained GloVe vectors are downloaded. Run the script with --output-dir outputs or execute all notebook cells to create:

- predictions.csv — predicted country, rank-1 score, correctness and vocabulary status
- ranked_candidates.csv — top-five candidate countries with cosine similarity for every example
- skipped_oov_examples.csv — examples skipped because an input or gold-answer word is out of vocabulary
- metrics.json — model metadata, evaluated/skipped counts and accuracy
- results.txt — readable summary and candidate scores
- cosine_scores.png — optional top-candidate score chart (requires Matplotlib)

The authoring environment could not download the model package. No numeric prediction table or accuracy value was fabricated; run on an internet-enabled Python environment to obtain actual model outputs.
