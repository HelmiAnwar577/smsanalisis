"""Pemilihan representasi fitur dan nilai alpha Multinomial Naive Bayes.

5-fold stratified cross validation (random_state=43) pada 913 data latih lama
(hasil pembagian 80:20). Hasil disimpan ke eksperimen/hasil/pemilihan_fitur.csv.
Jalankan dari folder proyek:  python eksperimen/pemilihan_fitur.py
"""
import csv
import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer import text_classifier as tc  # noqa: E402

df = tc.load_indonesian_dataset()
latih, _ = tc.split_dataset(df)
X = latih["text"].astype(str).values
y = latih["label"].values

hasil = []
for fitur in ("kata", "huruf"):
    for alpha in (1.0, 0.1, 0.01, 0.001):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
        f1 = []
        for a, b in skf.split(X, y):
            model, vec = tc.fit_model(X[a], y[a], feature=fitur, alpha=alpha)
            f1.append(f1_score(y[b], model.predict(vec.transform(X[b]))))
        hasil.append({"fitur": fitur, "alpha": alpha, "f1_rata2": round(float(np.mean(f1)), 4)})
        print(f"{fitur:6s} alpha={alpha:<6} F1={np.mean(f1) * 100:.2f}%")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil", "pemilihan_fitur.csv")
with open(out, "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=["fitur", "alpha", "f1_rata2"])
    w.writeheader()
    w.writerows(hasil)
print("Tersimpan:", out)
