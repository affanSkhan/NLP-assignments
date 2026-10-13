# NLP Laboratory Assignment 02 - Fundamental Language Processing for Three Languages

**Student:** Affan Khan Samid Khan  
**Roll No.:** 421135  
**PRN:** 22310588  
**Faculty:** Dr. Kapil N. Vhatkar

This folder contains Assignment 02: a reproducible NLP preprocessing pipeline for English, Hindi and Marathi. It uses eight author-created sample sentences per language (24 total); no external dataset download is required.

## Contents

- `421135_Affan_Khan_Samid_Khan_NLP_Assignment02.py` - Python source code.
- `421135_Affan_Khan_Samid_Khan_NLP_Assignment02.ipynb` - Jupyter notebook.
- `requirements.txt` - environment dependencies.
- `outputs/corpus.csv` - the 24 sentence samples.
- `outputs/processing_outputs.csv` - normalized text, tokens, removed stop words, filtered tokens, and stemming outputs per sentence.
- `outputs/comparative_results.csv` - summary statistics.
- `outputs/word_frequencies.csv` - complete ranked token frequency table.
- `outputs/top_word_frequencies.csv` - compact top-five frequency summary per language.
- `outputs/results.txt` - concise output summary.

## Pipeline

1. Unicode NFC normalization, case-folding and whitespace normalization.
2. Sentence segmentation using common punctuation, including Devanagari danda.
3. Regex tokenization for Latin and Devanagari word runs; punctuation is omitted.
4. Compact, curated, language-specific stop-word removal.
5. NLTK Snowball stemming for English only.
6. Word-frequency analysis and cross-language comparison.

A validated Hindi or Marathi morphological stemmer is not configured, so stemming for those languages is explicitly marked as not applied. The stop-word lists are teaching lists, not comprehensive linguistic resources. The corpus is designed to demonstrate the pipeline, not benchmark language quality.

## Run

Requires Python 3.10+ and packages in requirements.txt.

```bash
pip install -r requirements.txt
python 421135_Affan_Khan_Samid_Khan_NLP_Assignment02.py
```

By default, generated files are written to `nlp_assignment02_outputs/`. Set `--output-dir PATH` to choose another location.

## Sample results

| Language | Sentences | Tokens | Unique tokens | Stop words removed | Tokens remaining |
|---|---:|---:|---:|---:|---:|
| English | 8 | 68 | 59 | 13 | 55 |
| Hindi | 8 | 88 | 58 | 37 | 51 |
| Marathi | 8 | 56 | 51 | 6 | 50 |

The counts depend on the authored samples and small curated stop-word lists. They should not be interpreted as intrinsic properties of the languages.
