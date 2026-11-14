# NLP Laboratory — Assignment 06
**Student:** Affan Khan Samid Khan | **Roll No.:** 421135  
**Faculty:** Dr. Kapil N. Vhatkar

## Case study
SMS Spam Detection using NLP, TF-IDF and supervised machine learning.

## Dataset
UCI SMS Spam Collection: 5,574 English SMS messages (4,827 ham; 747 spam).  
Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
The dataset is downloaded by the script/notebook at first run. The original UCI page lists the dataset under CC BY 4.0; retain attribution.

## Run
```bash
pip install -r requirements.txt
python 421135_Affan_Khan_NLP_Assignment06.py
```
Or open the notebook in Jupyter/Colab and run all cells.

## Method
Conservative lowercase/whitespace normalization; TF-IDF word unigrams and bigrams; Multinomial Naive Bayes and Logistic Regression; stratified 80:20 test split with random_state=42. The best model is selected by spam-class F1 (recall as tie-breaker).

## Generated outputs
The run creates `outputs/model_comparison.csv`, `outputs/confusion_matrix.png`, `outputs/test_predictions.csv` and `outputs/error_analysis.csv`. Metrics are computed on the actual downloaded data; do not use guessed values.

## Important
This is a historical English-only corpus. Model predictions are not proof that a message is safe or malicious. Keep human review in any real deployment.
