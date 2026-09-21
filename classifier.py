"""Tiny sentiment classifier: TF-IDF + logistic regression, with train/eval/predict CLI."""
import argparse
import csv
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def build_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))


def train_and_eval(texts, labels, test_size=0.25, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)
    model = build_model().fit(Xtr, ytr)
    return model, accuracy_score(yte, model.predict(Xte))


def main(argv=None):
    p = argparse.ArgumentParser(description="Sentiment classifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/reviews.csv")
    t.add_argument("--out", default="model.pkl")
    q = sub.add_parser("predict")
    q.add_argument("text")
    q.add_argument("--model", default="model.pkl")
    a = p.parse_args(argv)
    if a.cmd == "train":
        texts, labels = load_csv(a.data)
        _, acc = train_and_eval(texts, labels)
        model = build_model().fit(texts, labels)  # final model uses all data
        with open(a.out, "wb") as f:
            pickle.dump(model, f)
        print(f"held-out accuracy: {acc:.2f}; saved {a.out}")
    else:
        with open(a.model, "rb") as f:
            print(pickle.load(f).predict([a.text])[0])


if __name__ == "__main__":
    main()
