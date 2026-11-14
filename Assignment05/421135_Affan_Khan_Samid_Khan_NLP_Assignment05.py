"""NLP Assignment 05: TF-IDF sentiment classifier + VADER paper component.
Student: Affan Khan Samid Khan | Roll: 421135 | PRN: 22310588
Run: pip install -r requirements.txt && python 421135_Affan_Khan_Samid_Khan_NLP_Assignment05.py
The NLTK twitter_samples corpus downloads on first run.
"""
import argparse, html, json, re
from pathlib import Path
import joblib, nltk, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS=["negative","positive"]
TOKENIZER=TweetTokenizer(preserve_case=False,reduce_len=True,strip_handles=False)

def clean_tweet(text):
    text=html.unescape(str(text))
    text=re.sub(r"^\s*RT\s+","",text,flags=re.I)
    text=re.sub(r"https?://\S+|www\.\S+"," URLTOKEN ",text,flags=re.I)
    text=re.sub(r"(?<!\w)@[A-Za-z0-9_]+"," USERTOKEN ",text)
    text=re.sub(r"#([A-Za-z0-9_]+)",r"\1",text)
    return re.sub(r"\s+"," ",text).strip()

def load_data(cap=None):
    try:
        pos=twitter_samples.strings("positive_tweets.json")
        neg=twitter_samples.strings("negative_tweets.json")
    except LookupError:
        if not nltk.download("twitter_samples"):
            raise RuntimeError("Cannot download twitter_samples; use an internet-enabled notebook.")
        pos=twitter_samples.strings("positive_tweets.json")
        neg=twitter_samples.strings("negative_tweets.json")
    if cap:
        pos,neg=pos[:cap],neg[:cap]
    df=pd.DataFrame([{"text":x,"label":"positive"} for x in pos]+[{"text":x,"label":"negative"} for x in neg])
    df["clean"]=df.text.map(clean_tweet)
    df["key"]=df.clean.str.casefold().str.strip()
    df=df[df.key.ne("")].copy()
    conflicts=set(df.groupby("key").label.nunique().loc[lambda s:s>1].index)
    df=df[~df.key.isin(conflicts)].drop_duplicates("key").reset_index(drop=True)
    df["record_id"]=np.arange(1,len(df)+1)
    meta={"dataset":"NLTK twitter_samples","source":"https://www.nltk.org/howto/twitter.html",
          "positive_loaded":len(pos),"negative_loaded":len(neg),"records_after_cleaning":len(df),
          "class_counts":df.label.value_counts().to_dict(),"language":"English","task":"binary sentiment"}
    return df,meta

def build_model():
    tfidf=TfidfVectorizer(preprocessor=clean_tweet,tokenizer=TOKENIZER.tokenize,
       token_pattern=None,lowercase=False,ngram_range=(1,2),min_df=2,
       max_features=30000,sublinear_tf=True,norm="l2")
    lr=LogisticRegression(C=2.0,max_iter=1500,solver="liblinear",
                          class_weight="balanced",random_state=42)
    return Pipeline([("tfidf",tfidf),("classifier",lr)])

def vader_predict(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer=SentimentIntensityAnalyzer()
    scores=[analyzer.polarity_scores(t) for t in texts]
    labels=["positive" if s["compound"]>=0.05 else "negative" if s["compound"]<=-0.05 else "neutral" for s in scores]
    return labels,scores

def metrics(y,pred,scores=None):
    pr,re,_,_=precision_recall_fscore_support(y,pred,labels=LABELS,average="macro",zero_division=0)
    pp,rr,pf,_=precision_recall_fscore_support(y,pred,labels=["positive"],average="macro",zero_division=0)
    return {"accuracy":float(accuracy_score(y,pred)),"macro_precision":float(pr),
      "macro_recall":float(re),"macro_f1":float(f1_score(y,pred,labels=LABELS,average="macro",zero_division=0)),
      "positive_precision":float(pp),"positive_recall":float(rr),"positive_f1":float(pf),
      "roc_auc":float(roc_auc_score([int(x=="positive") for x in y],scores)) if scores is not None else None,
      "neutral_predictions":int(sum(x=="neutral" for x in pred))}

def run(outdir="outputs",cap=None):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df,meta=load_data(cap)
    train,test=train_test_split(df,test_size=.2,random_state=42,stratify=df.label)
    model=build_model().fit(train.text,train.label)
    lr_pred=model.predict(test.text).tolist()
    lr_score=model.predict_proba(test.text)[:,list(model.classes_).index("positive")]
    vd_pred,vd_scores=vader_predict(test.text.tolist())
    y=test.label.tolist()
    m_lr=metrics(y,lr_pred,lr_score); m_vd=metrics(y,vd_pred)
    pred=test[["record_id","text","label"]].copy()
    pred["tfidf_logreg_prediction"]=lr_pred; pred["tfidf_positive_probability"]=lr_score
    pred["vader_prediction"]=vd_pred
    for key in ["neg","neu","pos","compound"]: pred["vader_"+key]=[s[key] for s in vd_scores]
    pred.to_csv(out/"test_predictions.csv",index=False)
    pred.loc[(pred.label!=pred.tfidf_logreg_prediction)|(pred.label!=pred.vader_prediction)].to_csv(out/"error_analysis.csv",index=False)
    pd.DataFrame([{"model":"TF-IDF + Logistic Regression",**m_lr},{"model":"VADER",**m_vd}]).to_csv(out/"model_comparison.csv",index=False)
    pd.DataFrame(classification_report(y,lr_pred,labels=LABELS,output_dict=True,zero_division=0)).T.to_csv(out/"classification_report_logreg.csv")
    pd.DataFrame(classification_report(y,vd_pred,labels=LABELS,output_dict=True,zero_division=0)).T.to_csv(out/"classification_report_vader.csv")
    confusion_matrix(y,lr_pred,labels=LABELS)
    pd.DataFrame(confusion_matrix(y,lr_pred,labels=LABELS),index=LABELS,columns=LABELS).to_csv(out/"confusion_matrix_logreg.csv")
    pd.DataFrame(confusion_matrix(y,vd_pred,labels=["negative","positive","neutral"]),index=["negative","positive","neutral"],columns=["negative","positive","neutral"]).to_csv(out/"confusion_matrix_vader.csv")
    train[["record_id","label"]].assign(split="train")._append(test[["record_id","label"]].assign(split="test")).sort_values("record_id").to_csv(out/"split_manifest.csv",index=False)
    joblib.dump(model,out/"tfidf_logistic_regression.joblib")
    result={"dataset":meta,"train_records":len(train),"test_records":len(test),
       "test_size":0.2,"random_state":42,"tfidf":{"ngrams":[1,2],"min_df":2,"max_features":30000},
       "logistic_regression":{"C":2.0,"max_iter":1500,"solver":"liblinear"},
       "vader_neutral_outputs":m_vd["neutral_predictions"],
       "tfidf_logistic_regression":m_lr,"vader":m_vd,
       "note":"VADER neutral outputs count as misses against binary gold labels; this is not the paper's original evaluation protocol."}
    (out/"metrics.json").write_text(json.dumps(result,indent=2))
    pd.DataFrame([{"text":t,**s} for t,s in zip(test.text,vd_scores)]).head(100).to_csv(out/"vader_scores_sample.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(["TF-IDF + LR","VADER"],[m_lr["accuracy"],m_vd["accuracy"]])
    ax.set_ylim(0,1); ax.set_ylabel("Accuracy"); ax.set_title("Held-out sentiment accuracy"); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(out/"model_accuracy_comparison.png",dpi=170); plt.close(fig)
    (out/"results.txt").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",default="outputs")
    ap.add_argument("--max-per-class",type=int,default=None)
    args=ap.parse_args()
    print(json.dumps(run(args.output_dir,args.max_per_class),indent=2))
