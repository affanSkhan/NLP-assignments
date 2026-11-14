# Assignment 05 Architecture

    NLTK labelled Twitter samples
               |
     normalization / deduplication
               |
           80 / 20 split
               |
       +-------+--------+
       |                |
 TF-IDF uni/bi-grams   VADER lexicon
       |               + five heuristics
 Logistic Regression   |
       |                |
       +-------+--------+
               |
       common held-out set
               |
  accuracy / macro P-R-F1 / error analysis
