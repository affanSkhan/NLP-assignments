"""NLP Laboratory Assignment 02: English, Hindi and Marathi text processing.

Student: Affan Khan Samid Khan | Roll No.: 421135 | PRN: 22310588
Faculty: Dr. Kapil N. Vhatkar

Requires NLTK. Matplotlib is optional and only used for the comparison chart.
"""
from __future__ import annotations
import argparse, csv, json, re, sys, unicodedata
from collections import Counter
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer

STUDENT = "Affan Khan Samid Khan"
ROLL_NO = "421135"
PRN = "22310588"
CORPUS = {
    "English": [
        "Natural language processing helps computers understand human language.",
        "Students use Python to analyze text and discover patterns.",
        "Machine learning models can classify documents and summarize information.",
        "Tokenization divides text into smaller linguistic units.",
        "Stop words are common words that may carry little task-specific information.",
        "Stemming reduces related words to approximate root forms.",
        "Language tools must handle spelling, punctuation, and ambiguity.",
        "Researchers compare systems using reliable evaluation measures.",
    ],
    "Hindi": [
        "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर को मानव भाषा समझने में मदद करता है।",
        "विद्यार्थी पाठ का विश्लेषण करने के लिए Python का उपयोग करते हैं।",
        "मशीन लर्निंग मॉडल दस्तावेज़ों का वर्गीकरण कर सकते हैं।",
        "शब्दों का टोकनीकरण पाठ को छोटे हिस्सों में बाँटता है।",
        "सामान्य शब्दों को हटाने से कभी-कभी उपयोगी अर्थ भी खो सकता है।",
        "भाषा के अनुसार शब्दों के रूप और वाक्य की रचना बदलती है।",
        "शोधकर्ता परिणामों की तुलना करने के लिए मापदंडों का उपयोग करते हैं।",
        "अच्छे डेटा से मॉडल बेहतर सीख सकते हैं।",
    ],
    "Marathi": [
        "नैसर्गिक भाषा प्रक्रिया संगणकांना मानवी भाषा समजण्यास मदत करते.",
        "विद्यार्थी मजकूर तपासण्यासाठी पायथनचा वापर करतात.",
        "मशीन लर्निंग मॉडेल दस्तऐवजांचे वर्गीकरण करू शकतात.",
        "शब्दांचे टोकनीकरण मजकूर लहान भागांत विभागते.",
        "सामान्य शब्द काढल्याने काही वेळा महत्त्वाचा अर्थही हरवू शकतो.",
        "भाषेनुसार शब्दांचे रूप आणि वाक्यरचना बदलते.",
        "संशोधक परिणामांची तुलना करण्यासाठी मापदंड वापरतात.",
        "चांगल्या डेटामुळे मॉडेल अधिक चांगले शिकू शकतात.",
    ],
}
STOPWORDS = {
    "English": set("a an and are as be by can for from in is it of on or the to use using with may that this must into their".split()),
    "Hindi": set("और का की के को में से है हैं यह कर के लिए पर भी कभी अनुसार तथा द्वारा एक हो सकते सकता सकती".split()),
    "Marathi": set("आणि चा ची चे ला मध्ये पासून आहे आहेत हे ही साठी काही वेळा नुसार होऊ शकतो शकतात च्या ने".split()),
}
TOKEN_RE = re.compile(r"[\u0900-\u0963\u0966-\u096F\u0971-\u097F]+|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+")
STEMMER = SnowballStemmer("english")

def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())

def segment_sentences(text):
    return [s.strip() for s in SENTENCE_RE.split(normalize(text)) if s.strip()]

def tokenize(text):
    return TOKEN_RE.findall(normalize(text))

def stem(language, tokens):
    if language == "English":
        return [STEMMER.stem(t) for t in tokens], "NLTK SnowballStemmer (English)"
    return None, "Not applied: no validated Hindi/Marathi stemmer configured"

def analyse():
    per_sentence, comparison, frequencies = [], [], []
    for language, sentences in CORPUS.items():
        all_tokens, kept, removed = [], [], 0
        stem_method = ""
        for sid, sentence in enumerate(sentences, 1):
            norm = normalize(sentence)
            toks = tokenize(norm)
            filtered = [t for t in toks if t not in STOPWORDS[language]]
            stems, stem_method = stem(language, filtered)
            all_tokens.extend(toks); kept.extend(filtered); removed += len(toks)-len(filtered)
            per_sentence.append({
                "language": language, "sentence_id": sid, "original_sentence": sentence,
                "normalized_text": norm, "tokens_json": json.dumps(toks, ensure_ascii=False),
                "token_count": len(toks),
                "removed_stopwords_json": json.dumps([t for t in toks if t in STOPWORDS[language]], ensure_ascii=False),
                "filtered_tokens_json": json.dumps(filtered, ensure_ascii=False),
                "stems_json": json.dumps(stems, ensure_ascii=False) if stems is not None else "Not applied",
                "stemming_method": stem_method,
            })
        count = Counter(kept)
        for rank, (token, n) in enumerate(count.most_common(), 1):
            frequencies.append({"language":language,"rank":rank,"token":token,"frequency":n})
        comparison.append({
            "language":language,"sentences":len(sentences),"tokens":len(all_tokens),
            "unique_tokens":len(set(all_tokens)),"stopwords_removed":removed,
            "tokens_after_stopword_removal":len(kept),
            "stemming_or_lemmatization":stem_method,"additional_task":"Word-frequency analysis",
        })
    return per_sentence, comparison, frequencies

def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def run(output_dir):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    per_sentence, comparison, frequencies = analyse()
    write_csv(out/"corpus.csv", [
        {"language":lang,"sentence_id":i,"sentence":s}
        for lang, sentences in CORPUS.items() for i,s in enumerate(sentences,1)
    ])
    write_csv(out/"processing_outputs.csv", per_sentence)
    write_csv(out/"comparative_results.csv", comparison)
    write_csv(out/"word_frequencies.csv", frequencies)
    lines=[
        "NLP Assignment 02 - Fundamental Language Processing for Three Languages",
        f"Student: {STUDENT} | Roll No.: {ROLL_NO} | PRN: {PRN}",
        "Corpus: 8 author-created sentences per language; 24 total.",
        "Tokenizer: Unicode NFC, case-folding, Latin/Devanagari regex; punctuation omitted.",
        "Stop-word lists: compact teaching lists, not comprehensive resources.",
        "Stemming: NLTK Snowball for English; not applied to Hindi/Marathi without validated stemmers.",
        "Additional task: word-frequency analysis.",
        "",
    ]
    for row in comparison:
        top=[r for r in frequencies if r["language"]==row["language"]][:5]
        lines.append(f"{row['language']}: sentences={row['sentences']}; tokens={row['tokens']}; unique={row['unique_tokens']}; stopwords_removed={row['stopwords_removed']}; remaining={row['tokens_after_stopword_removal']}")
        lines.append("  Top tokens: "+", ".join(f"{r['token']} ({r['frequency']})" for r in top))
    (out/"results.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        labels=[r["language"] for r in comparison]
        total=[r["tokens"] for r in comparison]
        uniq=[r["unique_tokens"] for r in comparison]
        removed=[r["stopwords_removed"] for r in comparison]
        xs=list(range(3)); w=.24
        fig,ax=plt.subplots(figsize=(9,5.3))
        ax.bar([x-w for x in xs],total,w,label="All tokens")
        ax.bar(xs,uniq,w,label="Unique tokens")
        ax.bar([x+w for x in xs],removed,w,label="Stopwords removed")
        ax.set_xticks(xs,labels); ax.set_ylabel("Count across 8 sentences")
        ax.set_title("Multilingual preprocessing statistics"); ax.grid(axis="y",alpha=.25)
        ax.legend(frameon=False,ncol=3); fig.tight_layout()
        fig.savefig(out/"comparative_statistics.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    except ImportError:
        pass
    return comparison, frequencies

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",default="nlp_assignment02_outputs")
    args=ap.parse_args()
    comparison, frequencies=run(args.output_dir)
    print("NLP Assignment 02 - English, Hindi and Marathi")
    print(f"Student: {STUDENT} | Roll No.: {ROLL_NO} | PRN: {PRN}")
    print("\nComparative results:")
    for r in comparison:
        print(f"{r['language']}: sentences={r['sentences']}, tokens={r['tokens']}, unique={r['unique_tokens']}, stop words removed={r['stopwords_removed']}, remaining={r['tokens_after_stopword_removal']}")
    print("\nTop 5 remaining token frequencies:")
    for language in CORPUS:
        top=[r for r in frequencies if r["language"]==language][:5]
        print(language+": "+", ".join(f"{r['token']} ({r['frequency']})" for r in top))
    print("Outputs:",Path(args.output_dir).resolve())

if __name__=="__main__":
    main()
