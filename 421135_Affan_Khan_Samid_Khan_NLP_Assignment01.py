"""NLP Laboratory Assignment 01: Bigrams and Sentence Probability.

Student: Affan Khan Samid Khan | Roll No.: 421135 | PRN: 22310588
Faculty: Dr. Kapil N. Vhatkar

Run with: python 421135_Affan_Khan_Samid_Khan_NLP_Assignment01.py
Only Python's standard library is required. CSV result files are written to
./nlp_assignment01_outputs by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

STUDENT_NAME = "Affan Khan Samid Khan"
ROLL_NO = "421135"
PRN = "22310588"
FACULTY = "Dr. Kapil N. Vhatkar"

CORPUS = [
    "I love natural language processing.",
    "Natural language processing is interesting.",
    "I love machine learning.",
    "Machine learning helps computers learn.",
    "Computers process natural language.",
    "I study language processing.",
    "Language models predict the next word.",
    "Bigram models use word pairs.",
    "Natural language models learn patterns.",
    "I study machine learning and language models.",
]

TEST_SENTENCES = [
    "I love natural language processing.",
    "Natural language models learn patterns.",
    "I study machine learning.",
]

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
BOUNDARY_START = "<s>"
BOUNDARY_END = "</s>"


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract word-like tokens, excluding punctuation."""
    return TOKEN_PATTERN.findall(text.lower())


def add_sentence_boundaries(sentence: str) -> list[str]:
    """Tokenize a sentence and add its start and end markers."""
    return [BOUNDARY_START, *tokenize(sentence), BOUNDARY_END]


def train_bigram_model(corpus: list[str]):
    """Return prepared sentences, bigram counts, context counts and vocabulary."""
    prepared = [add_sentence_boundaries(sentence) for sentence in corpus]
    bigram_counts: Counter[tuple[str, str]] = Counter()
    context_counts: Counter[str] = Counter()

    for tokens in prepared:
        for previous, current in zip(tokens, tokens[1:]):
            bigram_counts[(previous, current)] += 1
            context_counts[previous] += 1

    word_types = {
        token for tokens in prepared for token in tokens
        if token not in {BOUNDARY_START, BOUNDARY_END}
    }
    # Candidate next-token outcomes include every corpus word plus </s>.
    # The <s> marker is a context only and is never predicted.
    vocabulary_size = len(word_types) + 1
    return prepared, bigram_counts, context_counts, word_types, vocabulary_size


def mle_probability(previous: str, current: str, bigram_counts, context_counts) -> float:
    """Maximum-likelihood P(current | previous)."""
    denominator = context_counts[previous]
    if denominator == 0:
        return 0.0
    return bigram_counts[(previous, current)] / denominator


def laplace_probability(
    previous: str, current: str, bigram_counts, context_counts, vocabulary_size: int
) -> float:
    """Add-one/Laplace-smoothed conditional probability."""
    return (bigram_counts[(previous, current)] + 1) / (
        context_counts[previous] + vocabulary_size
    )


def sentence_probability(sentence: str, bigram_counts, context_counts,
                         vocabulary_size: int, smoothed: bool = False):
    """Return boundary tokens, component factors and the product probability."""
    tokens = add_sentence_boundaries(sentence)
    factors = []
    for previous, current in zip(tokens, tokens[1:]):
        if smoothed:
            probability = laplace_probability(
                previous, current, bigram_counts, context_counts, vocabulary_size
            )
        else:
            probability = mle_probability(previous, current, bigram_counts, context_counts)
        factors.append((previous, current, probability))
    total = math.prod(factor[2] for factor in factors)
    return tokens, factors, total


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a list of row dictionaries to UTF-8 CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("nlp_assignment01_outputs"),
        help="directory for generated CSV outputs (default: %(default)s)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared, bigrams, contexts, word_types, vocab_size = train_bigram_model(CORPUS)

    frequency_rows = [
        {
            "Bigram": f"{previous} {current}",
            "Previous word": previous,
            "Current word": current,
            "Frequency": count,
        }
        for (previous, current), count in bigrams.most_common()
    ]
    write_csv(
        args.output_dir / "bigram_frequencies.csv",
        ["Bigram", "Previous word", "Current word", "Frequency"],
        frequency_rows,
    )

    probability_rows = [
        {
            "Previous word": previous,
            "Current word": current,
            "Bigram count": count,
            "Previous-word count": contexts[previous],
            "MLE probability": f"{mle_probability(previous, current, bigrams, contexts):.8f}",
            "Laplace probability": f"{laplace_probability(previous, current, bigrams, contexts, vocab_size):.8f}",
        }
        for (previous, current), count in bigrams.most_common()
    ]
    write_csv(
        args.output_dir / "bigram_probabilities.csv",
        ["Previous word", "Current word", "Bigram count", "Previous-word count",
         "MLE probability", "Laplace probability"],
        probability_rows,
    )

    sentence_rows = []
    print("NLP LABORATORY ASSIGNMENT 01 - BIGRAMS AND SENTENCE PROBABILITY")
    print(f"Student: {STUDENT_NAME} | Roll No.: {ROLL_NO} | PRN: {PRN}")
    print(f"Corpus sentences: {len(CORPUS)} | Distinct observed bigrams: {len(bigrams)}")
    print(f"Word types: {len(word_types)} | Smoothing vocabulary size V: {vocab_size}\n")

    print("PREPROCESSED CORPUS")
    for index, tokens in enumerate(prepared, start=1):
        print(f"{index:02d}. {' '.join(tokens)}")

    print("\nTOP 15 BIGRAM FREQUENCIES")
    for index, ((previous, current), count) in enumerate(bigrams.most_common(15), start=1):
        print(f"{index:02d}. ({previous}, {current}) -> {count}")

    print("\nTEST SENTENCE PROBABILITIES")
    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        tokens, mle_factors, mle_total = sentence_probability(
            sentence, bigrams, contexts, vocab_size, smoothed=False
        )
        _, laplace_factors, laplace_total = sentence_probability(
            sentence, bigrams, contexts, vocab_size, smoothed=True
        )
        bigram_sequence = " | ".join(f"({a}, {b})" for a, b, _ in mle_factors)
        mle_products = " x ".join(f"{p:.6g}" for _, _, p in mle_factors)
        laplace_products = " x ".join(f"{p:.6g}" for _, _, p in laplace_factors)
        print(f"\nTest {index}: {sentence}")
        print(f"Tokens: {' '.join(tokens)}")
        print(f"Bigrams: {bigram_sequence}")
        print(f"MLE factors: {mle_products}")
        print(f"Unsmoothed probability: {mle_total:.12g}")
        print(f"Laplace factors: {laplace_products}")
        print(f"Laplace-smoothed probability: {laplace_total:.12g}")
        sentence_rows.append({
            "Test sentence": sentence,
            "Bigram sequence": bigram_sequence,
            "MLE factors": mle_products,
            "Unsmoothed probability": f"{mle_total:.12g}",
            "Laplace factors": laplace_products,
            "Laplace-smoothed probability": f"{laplace_total:.12g}",
        })

    write_csv(
        args.output_dir / "sentence_probabilities.csv",
        ["Test sentence", "Bigram sequence", "MLE factors", "Unsmoothed probability",
         "Laplace factors", "Laplace-smoothed probability"],
        sentence_rows,
    )

    previous, current = "we", "code"
    mle = mle_probability(previous, current, bigrams, contexts)
    laplace = laplace_probability(previous, current, bigrams, contexts, vocab_size)
    print("\nZERO-FREQUENCY DEMONSTRATION")
    print(f"Unseen bigram ({previous}, {current}): count={bigrams[(previous, current)]}, "
          f"MLE={mle:.6g}, Laplace={laplace:.6g}")
    print(f"CSV outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
