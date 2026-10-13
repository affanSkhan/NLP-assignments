# NLP Laboratory Assignment 04 - Country Prediction using Word Embeddings

**Student:** Affan Khan Samid Khan  
**Roll No.:** 421135  
**PRN:** 22310588  
**Faculty:** Dr. Kapil N. Vhatkar

## Deliverables
- `421135_Affan_Khan_Samid_Khan_NLP_Assignment04.py` — reusable Python implementation.
- `421135_Affan_Khan_Samid_Khan_NLP_Assignment04.ipynb` — notebook for running the evaluation.
- `analogy_test_cases.csv` — 20 labelled capital-country analogies.
- `requirements.txt` — Python dependencies.
- `outputs/RUN_FIRST.txt` and `outputs/README.md` — instructions for generating numeric outputs.

## Pre-trained model
The implementation uses Gensim's `glove-wiki-gigaword-50`: 50-dimensional vectors with around 400,000 vocabulary entries, about 65 MB in the Gensim model package, based on Wikipedia 2014 + Gigaword 5. Gensim downloads the weights on first run and caches them for later runs.

## Run
```bash
pip install -r requirements.txt
python 421135_Affan_Khan_Samid_Khan_NLP_Assignment04.py --output-dir outputs
```

Custom analogy:
```bash
python 421135_Affan_Khan_Samid_Khan_NLP_Assignment04.py --capital-1 Athens --country-1 Greece --capital-2 Baghdad
```

The formula is `v(country1) - v(capital1) + v(capital2)`. Candidate countries are ranked using cosine similarity, and input terms are excluded. Outputs include `predictions.csv`, `ranked_candidates.csv`, `skipped_oov_examples.csv`, `metrics.json`, `results.txt` and an optional `cosine_scores.png` chart.

## Reproducibility note
The authoring environment could not download the pretrained model weights. Therefore no prediction labels, similarity scores or accuracy values were fabricated. Run the notebook in Google Colab or another internet-enabled Python environment to generate actual numeric results before submission if the evaluator expects output tables to be pre-populated.

## References
- Pennington, Socher & Manning (2014), *GloVe: Global Vectors for Word Representation*: https://nlp.stanford.edu/pubs/glove.pdf
- Stanford GloVe project: https://nlp.stanford.edu/projects/glove/
- Gensim data models: https://github.com/piskvorky/gensim-data
- Gensim downloader docs: https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
