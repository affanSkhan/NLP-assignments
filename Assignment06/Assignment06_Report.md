# NLP Laboratory Assignment 06 — SMS Spam Detection

**Affan Khan Samid Khan | Roll No. 421135 | Academic Year 2026–27**  
**Faculty:** Dr. Kapil N. Vhatkar

## Aim
Develop and evaluate an NLP system that classifies SMS messages as legitimate (ham) or spam.

## Problem and motivation
Unwanted SMS may contain advertising, deceptive prize claims, premium-rate prompts or malicious links. NLP can learn text patterns such as words, phrases, digits and URLs to support automated screening. The task is binary supervised text classification.

## Dataset
UCI SMS Spam Collection: 5,574 English SMS records; 4,827 ham (86.6%) and 747 spam (13.4%). Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection. Dataset is historical and class-imbalanced.

## Objectives
1. Inspect and clean labelled SMS data.
2. Normalize text while retaining potentially useful spam indicators.
3. Extract TF-IDF unigram and bigram features.
4. Train Multinomial Naive Bayes and Logistic Regression.
5. Evaluate with accuracy, spam precision, recall and F1.
6. Inspect incorrect predictions and discuss limitations.

## Methodology
Raw SMS → validation/deduplication → lowercase and whitespace normalization → stratified 80:20 split (seed 42) → TF-IDF (1–2 grams, English stop words, min_df=2, max_df=.98, sublinear TF) → model training → held-out evaluation → error analysis.

## Preprocessing
Missing/invalid rows and exact duplicate label-message pairs are removed. Tokenization and n-gram construction are performed by TfidfVectorizer. Stop words are filtered. Stemming/lemmatization is not applied. Digits, punctuation and URLs are retained because they may signal spam.

## Models and configuration
- Multinomial Naive Bayes: alpha=0.5.
- Logistic Regression: C=2.0, class_weight=balanced, max_iter=1000.
- TF-IDF: word unigrams + bigrams, min_df=2, max_df=0.98, sublinear_tf=True.
- Test split: 20%, stratified, random_state=42.
- Classical scikit-learn models; batch size and epochs are not applicable.

## Evaluation
Accuracy measures overall correctness. Spam precision measures reliability of spam alerts; spam recall measures detected spam; spam F1 balances spam precision and recall. Since ham is the majority class, accuracy alone is insufficient. The notebook computes the exact results from the downloaded dataset and saves a comparison table, confusion matrix and error-analysis CSV. **No numerical scores are fabricated in this report; copy the measured values from the executed notebook output.**

## Error analysis
Inspect outputs/error_analysis.csv for actual false positives and false negatives. Possible causes include ambiguous short texts, unseen vocabulary, URL obfuscation, spelling variation and changed spam patterns.

## Advantages
- Lightweight, fast and reproducible.
- Uses open-source Python libraries.
- Compares two standard classifiers.
- Produces class-sensitive metrics and error examples.

## Limitations
- Historical, English-only data.
- Imbalanced classes and evolving scam language.
- TF-IDF has limited contextual understanding.
- Test results may not generalize to current multilingual phishing.

## Ethics and practical considerations
SMS may contain personal information; minimize collection and retention. False positives can hide legitimate messages and false negatives can expose users to scams. Use human review rather than automatic deletion, evaluate language-group fairness, and communicate uncertainty.

## Future scope
Add recent multilingual data, time-based validation, character n-grams, threshold tuning, contextual transformer models and human-in-the-loop feedback.

## Viva answers
1. **Problem:** Binary SMS ham/spam classification.
2. **Why NLP:** Relevant evidence is present in natural-language message text.
3. **Dataset:** UCI SMS Spam Collection, 5,574 English messages.
4. **Preprocessing:** Validation, deduplication, lowercase, whitespace normalization, TF-IDF tokenization and stop-word filtering.
5. **Model:** TF-IDF with Naive Bayes and Logistic Regression; select by spam F1.
6. **Metrics:** Accuracy, spam precision, recall and F1 due to class imbalance.
7. **Errors:** Ambiguity, unseen terms, obfuscated URLs, short context and historical data bias.
8. **Limitations:** Historical English-only corpus and limited context.
9. **Improvements:** Newer data, multilingual support, character n-grams, threshold tuning and transformers.
10. **Ethics:** Privacy, fairness, false alerts and human oversight.

## References
1. Almeida & Hidalgo (2011), SMS Spam Collection, UCI ML Repository, https://doi.org/10.24432/C5CC84
2. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
3. Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python, JMLR 12, 2825–2830.
4. Manning, Raghavan & Schütze (2008), Introduction to Information Retrieval.
