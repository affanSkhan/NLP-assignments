# NLP Laboratory Assignment 01 - Bigrams and Sentence Probability

Student: Affan Khan Samid Khan  
Roll No.: 421135  
PRN: 22310588  
Faculty: Dr. Kapil N. Vhatkar

This repository contains the reproducible implementation for Assignment 01. The corpus is a self-created set of ten short English sentences, so no external dataset download is required.

## Included deliverables

- 421135_Affan_Khan_Samid_Khan_NLP_Assignment01.py - standalone Python implementation (standard library only).
- corpus.csv - the ten-sentence corpus used by the implementation.
- bigram_frequencies.csv - observed bigram counts.
- bigram_probabilities.csv - MLE and add-one conditional probabilities.
- sentence_probabilities.csv - probability factors and final probabilities for three test sentences.
- results.txt - a concise summary of calculated results.

## Run

Requires Python 3.10 or newer. From the repository directory, run:

    python 421135_Affan_Khan_Samid_Khan_NLP_Assignment01.py

The script writes CSV files to nlp_assignment01_outputs/. An alternate location can be supplied with --output-dir PATH.

## Method

Text is lowercased and tokenized with a regular expression. Each sentence is wrapped in <s> and </s>, then consecutive token pairs and context counts are collected. MLE probabilities use:

P(current | previous) = count(previous, current) / count(previous)

Laplace smoothing uses:

P+(current | previous) = (count(previous, current) + 1) / (count(previous) + V)

Here, V is the corpus word types plus </s>; <s> is a context marker only.

## Verified summary

- Corpus sentences: 10
- Distinct observed bigrams: 44
- Word types excluding boundary markers: 24
- Smoothing vocabulary size: V = 25
- Unseen bigram example: (we, code) has MLE probability 0 and Laplace probability 0.04.

The complete PDF report and executed Jupyter notebook are provided as separate assignment deliverables.
