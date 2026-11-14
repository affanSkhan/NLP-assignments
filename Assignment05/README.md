# NLP Laboratory Assignment 05 - Sentiment Analysis and Mini Project

Student: Affan Khan Samid Khan  
Roll No.: 421135  
PRN: 22310588  
Faculty: Dr. Kapil N. Vhatkar

## Part A - Sentiment Analysis
Dataset: NLTK twitter_samples, using the labelled positive and negative tweet files (nominally 5,000 per class before duplicate cleaning). The raw corpus downloads automatically on first run and is not copied into this repository. Dataset documentation: https://www.nltk.org/howto/twitter.html.

Pipeline: conservative tweet cleaning; keep hashtags, emoji, emoticons, punctuation and negation; remove empty/duplicate/conflicting-label duplicates; stratified 80/20 split (seed 42); TF-IDF unigrams and bigrams; Logistic Regression.

TF-IDF: min_df=2, max_features=30,000, sublinear_tf=True, L2 normalization. Logistic Regression: C=2.0, liblinear, max_iter=1500, class_weight=balanced.

## Part B - Research Paper Component
Hutto, C. J., & Gilbert, E. (2014), *VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text*, ICWSM 8(1), 216-225. DOI: https://doi.org/10.1609/icwsm.v8i1.14550.

The code uses the authors' open-source Python reference implementation. VADER combines a human-validated sentiment lexicon with five heuristics: exclamation punctuation, ALL-CAPS emphasis, degree modifiers, contrastive 'but', and preceding three-token negation. Compound thresholds: >=0.05 positive, <=-0.05 negative, neutral between. Neutral outputs count as errors against binary gold labels and are reported separately.

## Run
Python 3.10+ and internet access are needed for the first NLTK corpus download.

    pip install -r requirements.txt
    python 421135_Affan_Khan_Samid_Khan_NLP_Assignment05.py

A smaller quick run is available with --max-per-class 500. Do not mix quick-run metrics with full-corpus metrics. The notebook includes a Colab repository-clone setup.

## Outputs
The script creates dataset metadata, split manifest, test predictions, error analysis, model comparison, classification reports, confusion matrices, metrics JSON, results summary, VADER rule examples, charts and a saved TF-IDF model in outputs/.

## Reproducibility
The authoring workspace could not reach the NLTK data download host. Code and notebook are ready to run, but numerical metrics must be generated in an internet-enabled Python/Colab session. No unexecuted scores are fabricated. Raw tweets are not redistributed; follow applicable Twitter/NLTK terms.
