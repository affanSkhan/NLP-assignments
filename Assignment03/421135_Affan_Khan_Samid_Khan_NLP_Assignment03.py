"""NLP Assignment 03: supervised HMM POS tagger with Viterbi decoding.

Student: Affan Khan Samid Khan | Roll No.: 421135 | PRN: 22310588
Faculty: Dr. Kapil N. Vhatkar

Run: python 421135_Affan_Khan_Samid_Khan_NLP_Assignment03.py --output-dir outputs
Requires Python 3.10+; matplotlib is optional (figures only).
"""
from __future__ import annotations
import argparse, csv, json, math, re
from collections import Counter
from pathlib import Path

STUDENT, ROLL_NO, PRN = "Affan Khan Samid Khan", "421135", "22310588"
START, END = "<START>", "<END>"
ALPHA = 0.1
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)


def load_corpus(path):
    train, test = [], []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sentence = [tuple(item.rsplit("/", 1)) for item in row["tagged_sequence"].split()]
            (train if row["split"] == "train" else test).append(sentence)
    return train, test


def norm(word):
    return word.lower() if word.isalpha() else word


def token_signature(word):
    w = word.lower()
    if not any(ch.isalnum() for ch in word): return "<UNK_PUNCT>"
    if w.isdigit(): return "<UNK_NUM>"
    if w.endswith("ly"): return "<UNK_LY>"
    if w.endswith("ing"): return "<UNK_ING>"
    if w.endswith("ed"): return "<UNK_ED>"
    if w.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence")): return "<UNK_NOUN_SUFFIX>"
    if w.endswith(("ous", "ful", "ive", "able", "ible", "al", "ent", "ant")): return "<UNK_ADJ_SUFFIX>"
    if w.endswith("s"): return "<UNK_S>"
    return "<UNK_OTHER>"


class HMMPOSTagger:
    """First-order HMM with add-alpha smoothing and Viterbi decoding in log-space."""
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.transitions = Counter()
        self.transition_totals = Counter()
        self.emissions = Counter()
        self.emission_totals = Counter()
        self.signatures = Counter()
        self.signature_totals = Counter()
        self.word_counts = Counter()
        self.tags = []
        self.vocab = set()
        self.signature_vocab = set()

    def fit(self, tagged_sentences):
        for sent in tagged_sentences:
            for word, _ in sent: self.word_counts[norm(word)] += 1
        self.tags = sorted({tag for sent in tagged_sentences for _, tag in sent})
        for sent in tagged_sentences:
            previous = START
            for word, tag in sent:
                word = norm(word)
                self.transitions[(previous, tag)] += 1
                self.transition_totals[previous] += 1
                self.emissions[(tag, word)] += 1
                self.emission_totals[tag] += 1
                if self.word_counts[word] <= 2:
                    sig = token_signature(word)
                    self.signatures[(tag, sig)] += 1
                    self.signature_totals[tag] += 1
                    self.signature_vocab.add(sig)
                previous = tag
            self.transitions[(previous, END)] += 1
            self.transition_totals[previous] += 1
        self.vocab = {w for _, w in self.emissions}
        self.transition_outcomes = self.tags + [END]
        if not self.signature_vocab: self.signature_vocab = {"<UNK_OTHER>"}
        return self

    def log_transition(self, prev, curr):
        k = len(self.transition_outcomes)
        return math.log((self.transitions[(prev, curr)] + self.alpha) /
                        (self.transition_totals[prev] + self.alpha * k))

    def log_emission(self, tag, word):
        w = norm(word)
        if w in self.vocab:
            numerator = self.emissions[(tag, w)] + self.alpha
            denominator = self.emission_totals[tag] + self.alpha * len(self.vocab)
        else:
            sig = token_signature(word)
            numerator = self.signatures[(tag, sig)] + self.alpha
            denominator = self.signature_totals[tag] + self.alpha * len(self.signature_vocab)
        return math.log(numerator / denominator)

    def tag(self, words):
        if not words: return []
        score = {t: self.log_transition(START, t) + self.log_emission(t, words[0]) for t in self.tags}
        backpointers = [{t: None for t in self.tags}]
        for word in words[1:]:
            next_score, back = {}, {}
            for curr in self.tags:
                best, prev = max(
                    ((score[p] + self.log_transition(p, curr), p) for p in self.tags),
                    key=lambda x: x[0])
                next_score[curr] = best + self.log_emission(curr, word)
                back[curr] = prev
            score = next_score
            backpointers.append(back)
        last = max(self.tags, key=lambda t: score[t] + self.log_transition(t, END))
        path = [last]
        for i in range(len(words)-1, 0, -1):
            path.append(backpointers[i][path[-1]])
        return list(zip(words, reversed(path)))


def evaluate(model, test, train):
    gold, pred, sentence_rows = [], [], []
    for i, sent in enumerate(test, 1):
        words = [w for w, _ in sent]
        true_tags = [t for _, t in sent]
        predicted = [t for _, t in model.tag(words)]
        gold.extend(true_tags); pred.extend(predicted)
        errors = [f"{w}: predicted {p}, gold {g}" for w, g, p in zip(words, true_tags, predicted) if g != p]
        ncorrect = sum(g == p for g, p in zip(true_tags, predicted))
        sentence_rows.append({
            "sentence_id": i, "sentence": " ".join(words),
            "gold_tags": " ".join(true_tags), "predicted_tags": " ".join(predicted),
            "correct_tokens": ncorrect, "total_tokens": len(words),
            "accuracy": ncorrect/len(words), "errors": "; ".join(errors) or "None"
        })
    classes = sorted(set(gold) | set(pred))
    per_tag, matrix = [], {g:{p:0 for p in classes} for g in classes}
    for g, p in zip(gold, pred): matrix[g][p] += 1
    for tag in classes:
        tp=sum(g==tag and p==tag for g,p in zip(gold,pred))
        fp=sum(g!=tag and p==tag for g,p in zip(gold,pred))
        fn=sum(g==tag and p!=tag for g,p in zip(gold,pred))
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_tag.append({"tag":tag,"precision":precision,"recall":recall,"f1":f1,
                        "support":sum(g==tag for g in gold),"tp":tp,"fp":fp,"fn":fn})
    correct=sum(g==p for g,p in zip(gold,pred))
    all_tag_counts=Counter(gold)
    majority=all_tag_counts.most_common(1)[0]
    # Memorized most-frequent training tag per surface word (fallback to global training majority).
    word_tags=Counter((norm(w),t) for sent in train for w,t in sent)
    by_word={}
    for (word,tag),count in word_tags.items():
        by_word.setdefault(word,Counter())[tag]+=count
    global_train=Counter(t for sent in train for _,t in sent)
    default=global_train.most_common(1)[0][0]
    word_correct=0
    for sent in test:
        for word, gold_tag in sent:
            tags=by_word.get(norm(word))
            guess=tags.most_common(1)[0][0] if tags else default
            word_correct += guess==gold_tag
    metrics={
        "test_sentences":len(test),"test_tokens":len(gold),"correct_tokens":correct,
        "accuracy":correct/len(gold),"macro_precision":sum(x["precision"] for x in per_tag)/len(per_tag),
        "macro_recall":sum(x["recall"] for x in per_tag)/len(per_tag),
        "macro_f1":sum(x["f1"] for x in per_tag)/len(per_tag),
        "majority_baseline_accuracy":majority[1]/len(gold),"majority_tag":majority[0],
        "word_majority_baseline_accuracy":word_correct/len(gold),"tags":classes
    }
    return metrics, per_tag, matrix, sentence_rows


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run(output_dir="outputs"):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    corpus_path=Path(__file__).resolve().with_name("tagged_corpus.csv")
    train,test=load_corpus(corpus_path)
    model=HMMPOSTagger().fit(train)
    metrics, per_tag, matrix, predictions=evaluate(model,test,train)
    train_rows=[{"sentence_id":i,"sentence":" ".join(w for w,_ in s),
                 "tagged_sequence":" ".join(f"{w}/{t}" for w,t in s)} for i,s in enumerate(train,1)]
    test_rows=[{"sentence_id":i,"sentence":" ".join(w for w,_ in s),
                "gold_tagged_sequence":" ".join(f"{w}/{t}" for w,t in s)} for i,s in enumerate(test,1)]
    write_csv(out/"train_corpus.csv",train_rows); write_csv(out/"test_gold_corpus.csv",test_rows)
    write_csv(out/"sentence_predictions.csv",predictions); write_csv(out/"per_tag_metrics.csv",per_tag)
    write_csv(out/"confusion_matrix.csv",[{"gold_tag":g,**row} for g,row in matrix.items()])
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    lines=["NLP Assignment 03 - HMM POS Tagging with Viterbi",
           f"Student: {STUDENT} | Roll No.: {ROLL_NO} | PRN: {PRN}",
           f"Training sentences: {len(train)}",f"Held-out test sentences: {len(test)}",
           f"Held-out tokens: {metrics['test_tokens']}",
           f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_tokens']}/{metrics['test_tokens']})",
           f"Macro precision: {metrics['macro_precision']:.4f}",f"Macro recall: {metrics['macro_recall']:.4f}",
           f"Macro F1: {metrics['macro_f1']:.4f}",
           f"Global majority-tag baseline: {metrics['majority_baseline_accuracy']:.4f} (always {metrics['majority_tag']})",
           f"Word-majority baseline: {metrics['word_majority_baseline_accuracy']:.4f}",
           "Model: first-order supervised HMM; add-alpha smoothing alpha=0.1; unknown words use suffix/shape signatures.",
           "Evaluation data are a small manually annotated course corpus, not a benchmark."]
    for r in predictions:
        lines += ["",f"{r['sentence_id']}. {r['sentence']}",f"  Gold: {r['gold_tags']}",
                  f"  Pred: {r['predicted_tags']} ({r['correct_tokens']}/{r['total_tokens']})"]
        if r["errors"] != "None": lines.append(f"  Errors: {r['errors']}")
    (out/"results.txt").write_text("\\n".join(lines)+"\\n",encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        tags=metrics["tags"]; counts=[sum(matrix[t].values()) for t in tags]
        rec=[100*matrix[t][t]/counts[i] if counts[i] else 0 for i,t in enumerate(tags)]
        fig,ax=plt.subplots(1,2,figsize=(10,4.2))
        ax[0].bar(tags,rec); ax[0].set_title("Held-out recall by tag"); ax[0].set_ylabel("Recall (%)")
        ax[0].tick_params(axis="x",rotation=45)
        ax[1].bar(["HMM + Viterbi","Word-majority"],[100*metrics["accuracy"],100*metrics["word_majority_baseline_accuracy"]])
        ax[1].set_ylim(0,100); ax[1].set_ylabel("Token accuracy (%)"); ax[1].set_title("Baseline comparison")
        for a in ax: a.grid(axis="y",alpha=.25)
        fig.tight_layout(); fig.savefig(out/"evaluation_chart.png",dpi=180,bbox_inches="tight"); plt.close(fig)
        grid=[[matrix[g][p] for p in tags] for g in tags]
        fig,ax=plt.subplots(figsize=(7,6)); image=ax.imshow(grid,cmap="Blues")
        ax.set_xticks(range(len(tags)),tags,rotation=45,ha="right"); ax.set_yticks(range(len(tags)),tags)
        ax.set_xlabel("Predicted tag"); ax.set_ylabel("Gold tag"); ax.set_title("HMM POS-tagging confusion matrix")
        for i in range(len(tags)):
            for j in range(len(tags)): ax.text(j,i,str(grid[i][j]),ha="center",va="center",fontsize=8)
        fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    except ImportError:
        pass
    return model,metrics,per_tag,matrix,predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default="outputs")
    parser.add_argument("--sentence",action="append",help="extra sentence to tag; repeatable")
    args=parser.parse_args()
    model,metrics,per_tag,matrix,predictions=run(args.output_dir)
    print(f"Student: {STUDENT} | Roll No.: {ROLL_NO} | PRN: {PRN}")
    print(f"Training: {len(load_corpus(Path(__file__).with_name('tagged_corpus.csv'))[0])} sentences")
    print(f"Test: {metrics['test_sentences']} sentences / {metrics['test_tokens']} tokens")
    print(f"Accuracy={metrics['accuracy']:.4f}; macro P/R/F1={metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}")
    print(f"Global majority baseline={metrics['majority_baseline_accuracy']:.4f}; word-majority baseline={metrics['word_majority_baseline_accuracy']:.4f}")
    for sentence in args.sentence or []:
        print(sentence)
        print(" ".join(f"{w}/{t}" for w,t in model.tag(TOKEN_RE.findall(sentence))))


if __name__=="__main__":
    main()
