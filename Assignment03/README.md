# NLP Laboratory Assignment 03 - Survey and Implementation of POS Tagging

**Student:** Affan Khan Samid Khan  
**Roll No.:** 421135  
**PRN:** 22310588  
**Faculty:** Dr. Kapil N. Vhatkar

## Contents
- 421135_Affan_Khan_Samid_Khan_NLP_Assignment03.py — HMM implementation with Viterbi decoding.
- 421135_Affan_Khan_Samid_Khan_NLP_Assignment03.ipynb — notebook front-end.
- tagged_corpus.csv — 48 manually tagged training sentences and 17 held-out gold-labeled test sentences.
- outputs/ — prediction, metric, confusion-matrix and result files generated when the script runs.

The completed 9-page PDF report and editable DOCX are provided in the submission package.

## Run
Requires Python 3.10+; Matplotlib is optional for plots.

    pip install -r requirements.txt
    python 421135_Affan_Khan_Samid_Khan_NLP_Assignment03.py --output-dir outputs --sentence "The new model predicts useful results."

The tagger estimates HMM transition and emission probabilities, applies add-alpha smoothing (alpha=0.1), uses suffix/shape signatures for unseen tokens, and decodes with Viterbi in log space.

## Evaluation caveat
The fixed training and test samples are hand-authored and manually tagged, not a standard benchmark. The HMM scores 89/100 tokens (89.00% accuracy, macro-F1 0.8744). The global majority-tag baseline is 26%; the word-majority baseline is 92%. The report discusses the fact that word-majority is higher on this small, lexically predictable split instead of claiming benchmark performance.
