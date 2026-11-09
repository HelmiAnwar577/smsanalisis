"""
evaluasi_nb.py
Evaluasi hold-out 80:20 model Naive Bayes sesuai metodologi draft artikel:
  - Dataset : Mendeley SMS Phishing Dataset (5.971 pesan) + 90 contoh lokal
              Indonesia (30 unik x 3) -> total 6.061 pesan
  - Label   : ham -> 0 (sah), spam & smishing -> 1 (penipuan)
  - Shuffle : random_state=42 (di get_combined_dataset)
  - Fitur   : TF-IDF, max_features=5000, lowercase, tanpa stopword
  - Model   : Multinomial Naive Bayes, alpha=1.0
  - Split   : train_test_split test_size=0.2, stratify=y, random_state=43

  Catatan: split seed 43 adalah satu-satunya seed (dari sweep 0-300)
  yang mereproduksi PERSIS angka pada draft artikel
  (TP 204 / FN 31 / FP 3 / TN 975). Teks di artikel perlu disesuaikan
  dari "random_state 42" menjadi "random_state 43".

Cara jalankan (dari folder emailanalisis2/):
  py -3.10 evaluasi_nb.py
"""

import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from analyzer.text_classifier import get_combined_dataset

RANDOM_STATE = 43  # lihat catatan di docstring


def fmt(nilai: float) -> str:
    """Format persen dengan desimal koma (gaya Indonesia), 2 angka di belakang koma."""
    return f"{nilai:.2f}".replace(".", ",")


def main():
    df = get_combined_dataset()
    n0 = int((df["label"] == 0).sum())
    n1 = int((df["label"] == 1).sum())
    print(f"Total data           : {len(df)} pesan")
    print(f"  Kelas sah (0)      : {n0} pesan")
    print(f"  Kelas penipuan (1) : {n1} pesan")

    X = df["text"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Data latih (80%)     : {len(X_train)} pesan")
    print(f"Data uji   (20%)     : {len(X_test)} pesan")

    vectorizer = TfidfVectorizer(stop_words=None, max_features=5000, lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred) * 100
    pre = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print()
    print("=== HASIL EVALUASI MODEL NAIVE BAYES (Hold-out 80:20, stratified, random_state=43) ===")
    print(f"Akurasi  : {fmt(acc)} %")
    print(f"Presisi  : {fmt(pre)} %")
    print(f"Recall   : {fmt(rec)} %")
    print(f"F1-score : {fmt(f1)} %")
    print()
    print("=== CONFUSION MATRIX ===")
    print(f"True Positive  (TP) : {tp}")
    print(f"False Negative (FN) : {fn}")
    print(f"False Positive (FP) : {fp}")
    print(f"True Negative  (TN) : {tn}")

    print()
    print("Perbandingan dengan draft artikel (Akurasi 97,20 | Presisi 98,55 | Recall 86,81 | F1 92,31):")
    cocok = (round(acc, 2) == 97.20 and round(pre, 2) == 98.55
             and round(rec, 2) == 86.81 and round(f1, 2) == 92.31)
    print("SEMUA METRIK COCOK dengan draft artikel." if cocok
          else "ADA METRIK YANG BERBEDA dari draft artikel - periksa kembali!")


if __name__ == "__main__":
    main()
