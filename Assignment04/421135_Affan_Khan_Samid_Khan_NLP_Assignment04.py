"""NLP Assignment 04: Country prediction using pretrained GloVe embeddings.

Student: Affan Khan Samid Khan | Roll No.: 421135 | PRN: 22310588
Faculty: Dr. Kapil N. Vhatkar

First run downloads the Gensim GloVe Wiki-Gigaword 50-dimensional vectors (~65 MB).
Run:
    python 421135_Affan_Khan_Samid_Khan_NLP_Assignment04.py
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
from typing import Iterable

STUDENT_NAME = "Affan Khan Samid Khan"
ROLL_NO = "421135"
PRN = "22310588"
MODEL_NAME = "glove-wiki-gigaword-50"

COUNTRY_TOKENS = [
    "afghanistan", "albania", "algeria", "argentina", "australia", "austria",
    "bangladesh", "belarus", "belgium", "bolivia", "brazil", "bulgaria", "cambodia",
    "cameroon", "canada", "chile", "china", "colombia", "croatia", "cuba", "cyprus",
    "czech_republic", "denmark", "dominican_republic", "ecuador", "egypt", "estonia",
    "ethiopia", "finland", "france", "georgia", "germany", "ghana", "greece", "hungary",
    "iceland", "india", "indonesia", "iran", "iraq", "ireland", "israel", "italy", "jamaica",
    "japan", "jordan", "kazakhstan", "kenya", "kuwait", "latvia", "lebanon", "libya",
    "lithuania", "malaysia", "mexico", "mongolia", "morocco", "nepal", "netherlands",
    "new_zealand", "nigeria", "north_korea", "norway", "pakistan", "panama", "peru",
    "philippines", "poland", "portugal", "qatar", "romania", "russia", "saudi_arabia",
    "serbia", "singapore", "slovakia", "slovenia", "somalia", "south_africa", "south_korea",
    "spain", "sri_lanka", "sudan", "sweden", "switzerland", "syria", "taiwan", "tanzania",
    "thailand", "tunisia", "turkey", "uganda", "ukraine", "united_arab_emirates",
    "united_kingdom", "united_states", "uruguay", "venezuela", "vietnam", "yemen", "zambia", "zimbabwe"
]

def normalize_token(token: str) -> str:
    return "_".join(token.casefold().strip().replace("-", " ").split())

def cosine(a, b) -> float:
    import numpy as np
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def load_model(model_name: str = MODEL_NAME):
    import gensim.downloader as api
    started = time.time()
    print(f"Loading pretrained model: {model_name}")
    print("First run may download approximately 65 MB; later runs use the local cache.")
    wv = api.load(model_name)
    print(f"Loaded vocabulary={len(wv.key_to_index):,}; dimensions={wv.vector_size}; time={time.time()-started:.1f}s")
    return wv

def available_country_candidates(wv, candidates: Iterable[str] = COUNTRY_TOKENS) -> list[str]:
    return [name for name in candidates if name in wv.key_to_index]

def predict_country(capital_1: str, country_1: str, capital_2: str, wv,
                    candidates: Iterable[str] = COUNTRY_TOKENS, topn: int = 5):
    import numpy as np
    cap1, ctry1, cap2 = map(normalize_token, (capital_1, country_1, capital_2))
    missing = [w for w in (cap1, ctry1, cap2) if w not in wv.key_to_index]
    if missing:
        return {"capital_1": capital_1, "country_1": country_1, "capital_2": capital_2,
                "predicted_country": "OOV", "top_candidates": [], "missing_inputs": missing,
                "status": "SKIPPED_OOV"}
    query = wv.get_vector(ctry1) - wv.get_vector(cap1) + wv.get_vector(cap2)
    excluded = {cap1, ctry1, cap2}
    candidate_words = [c for c in available_country_candidates(wv, candidates) if c not in excluded]
    if not candidate_words:
        raise RuntimeError("No country candidate tokens are available in the embedding vocabulary.")
    qnorm = float(np.linalg.norm(query))
    if not qnorm:
        raise RuntimeError("Analogy vector has zero norm.")
    ranked = sorted(
        ((country, cosine(query, wv.get_vector(country))) for country in candidate_words),
        key=lambda pair: pair[1], reverse=True
    )[:topn]
    top = [{"country_token": word, "cosine_similarity": float(score)} for word, score in ranked]
    return {"capital_1": capital_1, "country_1": country_1, "capital_2": capital_2,
            "predicted_country": top[0]["country_token"], "top_candidates": top,
            "missing_inputs": [], "status": "OK"}

def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run_evaluation(wv, test_path: Path, output_dir: Path, topn: int = 5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(test_path).open(encoding="utf-8-sig", newline="") as f:
        test_cases = list(csv.DictReader(f))
    prediction_rows, ranked_rows = [], []
    correct = evaluated = skipped = 0
    for idx, row in enumerate(test_cases, 1):
        res = predict_country(row["capital_1"], row["country_1"], row["capital_2"], wv, topn=topn)
        expected = normalize_token(row["expected_country"])
        if res["status"] != "OK":
            skipped += 1
            predicted, is_correct = "OOV", "NOT_EVALUATED"
        else:
            predicted = res["predicted_country"]
            is_correct = predicted == expected
            evaluated += 1
            correct += int(is_correct)
        top = res["top_candidates"]
        prediction_rows.append({
            "example_id": idx, "capital_1": row["capital_1"], "country_1": row["country_1"],
            "capital_2": row["capital_2"], "expected_country": row["expected_country"],
            "predicted_country": predicted,
            "top1_cosine_similarity": top[0]["cosine_similarity"] if top else "",
            "correct": is_correct, "status": res["status"], "missing_inputs": ",".join(res["missing_inputs"])
        })
        for rank, cand in enumerate(top, 1):
            ranked_rows.append({
                "example_id": idx, "capital_1": row["capital_1"], "country_1": row["country_1"],
                "capital_2": row["capital_2"], "rank": rank, "candidate_country": cand["country_token"],
                "cosine_similarity": cand["cosine_similarity"], "expected_country": row["expected_country"]
            })
    metrics = {
        "model_name": MODEL_NAME, "embedding_dimension": int(wv.vector_size),
        "vocabulary_size": int(len(wv.key_to_index)), "test_examples": len(test_cases),
        "evaluated_examples": evaluated, "skipped_oov_examples": skipped, "correct_predictions": correct,
        "accuracy_over_evaluated_examples": correct / evaluated if evaluated else None,
        "accuracy_over_all_examples": correct / len(test_cases) if test_cases else None,
        "candidate_country_count": len(available_country_candidates(wv)),
        "candidate_filter": "Curated country-name vocabulary; input words removed",
        "vector_formula": "v(country_1) - v(capital_1) + v(capital_2)",
        "similarity": "cosine similarity", "test_source": Path(test_path).name,
        "note": "Small hand-selected instructional test set; not a standard benchmark."
    }
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "ranked_candidates.csv", ranked_rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "NLP Assignment 04 - Country Prediction using Word Embeddings",
        f"Student: {STUDENT_NAME} | Roll No.: {ROLL_NO} | PRN: {PRN}",
        f"Model: {MODEL_NAME}", f"Embedding dimension: {wv.vector_size}",
        f"Vocabulary size: {len(wv.key_to_index):,}", f"Test examples: {len(test_cases)}",
        f"Evaluated examples: {evaluated}", f"Skipped OOV examples: {skipped}",
        f"Correct predictions: {correct}",
        f"Accuracy on evaluated examples: {metrics['accuracy_over_evaluated_examples']}",
        f"Accuracy with skips treated as wrong: {metrics['accuracy_over_all_examples']}",
        f"Available country candidates: {metrics['candidate_country_count']}", "",
        "Predictions and scores:"
    ]
    for r in prediction_rows:
        lines.append(f"{r['example_id']:02d}. {r['capital_1']} : {r['country_1']} :: {r['capital_2']} : {r['predicted_country']} (expected {r['expected_country']}; cosine={r['top1_cosine_similarity']}; {r['status']})")
        for cand in [x for x in ranked_rows if x["example_id"] == r["example_id"]][:3]:
            lines.append(f"    #{cand['rank']} {cand['candidate_country']} ({float(cand['cosine_similarity']):.4f})")
    lines.extend(["", "Small instructional test set; not a benchmark evaluation."])
    (output_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        rows = [r for r in prediction_rows if r["status"] == "OK"]
        labels = [str(r["example_id"]) for r in rows]
        scores = [float(r["top1_cosine_similarity"]) for r in rows]
        colors = ["#27866E" if r["correct"] is True else "#CE7E38" for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(labels, scores, color=colors)
        ax.set_xlabel("Analogy example number")
        ax.set_ylabel("Cosine similarity of top candidate")
        ax.set_title("Top-ranked country score by analogy")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(output_dir / "cosine_scores.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    return metrics, prediction_rows, ranked_rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME, help="Gensim downloader model name")
    parser.add_argument("--test-csv", type=Path, default=Path(__file__).with_name("analogy_test_cases.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--capital-1")
    parser.add_argument("--country-1")
    parser.add_argument("--capital-2")
    parser.add_argument("--topn", type=int, default=5)
    args = parser.parse_args()
    if any([args.capital_1, args.country_1, args.capital_2]) and not all([args.capital_1, args.country_1, args.capital_2]):
        parser.error("Supply --capital-1, --country-1 and --capital-2 together.")
    try:
        wv = load_model(args.model)
    except Exception as exc:
        print("ERROR: pretrained vectors could not be loaded. Check internet access and retry.", file=sys.stderr)
        print("No predictions or scores have been fabricated.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    metrics, predictions, rankings = run_evaluation(wv, args.test_csv, args.output_dir, args.topn)
    print(json.dumps(metrics, indent=2))
    if args.capital_1:
        print(json.dumps(predict_country(args.capital_1, args.country_1, args.capital_2, wv, topn=args.topn), indent=2))

if __name__ == "__main__":
    main()
