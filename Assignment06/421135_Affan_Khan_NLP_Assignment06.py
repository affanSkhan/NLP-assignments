"""NLP Assignment 06: SMS Spam Detection | Affan Khan | Roll 421135
Run: pip install -r requirements.txt && python 421135_Affan_Khan_NLP_Assignment06.py
Downloads the UCI SMS Spam Collection and computes all metrics from a stratified hold-out set.
"""
from pathlib import Path
import io, zipfile, urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"; OUT=ROOT/"outputs"
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
FILE=DATA/"SMSSpamCollection"
URL="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def load_data():
    if not FILE.exists():
        req=urllib.request.Request(URL,headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req,timeout=60) as r: payload=r.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            name=next(n for n in z.namelist() if n.endswith("SMSSpamCollection"))
            FILE.write_bytes(z.read(name))
    d=pd.read_csv(FILE,sep="\\t",header=None,names=["label","message"],encoding="utf-8")
    d=d.dropna(subset=["label","message"]).copy()
    d["label"]=d["label"].astype(str).str.strip().str.lower()
    d["message"]=d["message"].astype(str)
    d=d[d.label.isin(["ham","spam"])].drop_duplicates(["label","message"]).reset_index(drop=True)
    return d

def clean(text): return " ".join(str(text).lower().split())

def main():
    d=load_data()
    d["clean"]=d.message.map(clean); d["target"]=d.label.map({"ham":0,"spam":1}).astype(int)
    print("Records:",len(d),"\\nClass counts:\\n",d.label.value_counts())
    print("Missing:",d[["label","message"]].isna().sum().sum(),
          "Duplicates:",d.duplicated(["label","message"]).sum())
    print("Mean/median chars:",round(d.message.str.len().mean(),2),d.message.str.len().median())
    Xtr,Xte,ytr,yte=train_test_split(d.clean,d.target,test_size=.2,random_state=42,stratify=d.target)
    vec=TfidfVectorizer(lowercase=False,stop_words="english",ngram_range=(1,2),min_df=2,max_df=.98,sublinear_tf=True)
    models={
      "Multinomial Naive Bayes":Pipeline([("tfidf",vec),("clf",MultinomialNB(alpha=.5))]),
      "Logistic Regression":Pipeline([("tfidf",vec),("clf",LogisticRegression(C=2,class_weight="balanced",max_iter=1000,random_state=42))])
    }
    rows=[]; fitted={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte); fitted[name]=m
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,p),
          "Precision (spam)":precision_score(yte,p,zero_division=0),
          "Recall (spam)":recall_score(yte,p,zero_division=0),
          "F1-score (spam)":f1_score(yte,p,zero_division=0)})
    results=pd.DataFrame(rows).sort_values(["F1-score (spam)","Recall (spam)"],ascending=False)
    results.to_csv(OUT/"model_comparison.csv",index=False)
    best=results.iloc[0]["Model"]; model=fitted[best]; pred=model.predict(Xte)
    print("\\nRESULTS\\n",results.to_string(index=False))
    print("\\nSELECTED:",best,"\\n",classification_report(yte,pred,target_names=["ham","spam"],zero_division=0))
    cm=confusion_matrix(yte,pred)
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["ham","spam"],yticklabels=["ham","spam"])
    plt.title("Confusion Matrix — "+best); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
    plt.savefig(OUT/"confusion_matrix.png",dpi=180); plt.close()
    test=pd.DataFrame({"message":d.loc[Xte.index,"message"].values,
      "actual":yte.map({0:"ham",1:"spam"}).values,
      "predicted":pd.Series(pred,index=yte.index).map({0:"ham",1:"spam"}).values})
    test["correct"]=test.actual==test.predicted
    test.to_csv(OUT/"test_predictions.csv",index=False)
    test.loc[~test.correct].to_csv(OUT/"error_analysis.csv",index=False)
    for msg in ["Congratulations! You have won a free prize. Call now to claim.",
                "Hey, are we still meeting for lunch today?"]:
        print(("SPAM" if model.predict([clean(msg)])[0] else "HAM"),"|",msg)
    print("Outputs saved in",OUT)

if __name__=="__main__": main()
