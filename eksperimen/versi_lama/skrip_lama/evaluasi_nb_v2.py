# -*- coding: utf-8 -*-
"""
evaluasi_nb_v2.py — Evaluasi model Naive Bayes TANPA data leakage:
 - Dataset EN Mendeley (5.971) + 30 contoh ID UNIK (15 spam + 15 ham)
 - Duplikasi dilakukan (dan dilaporkan) SETELAH split 80:20, sehingga
   salinan yang sama tidak pernah muncul bersamaan di latih & uji.
 - random_state=42 (pre-shuffle) dan random_state=42 (split), stratified.
Output juga menampilkan kinerja EN-only sebagai pembanding.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from analyzer.text_classifier import load_main_dataset, get_indonesian_examples


def fmt(x):
    return f"{x:.2f}".replace(".", ",")


def eval_df(df, tag, seed=42):
    X, y = df["text"].astype(str), df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    v = TfidfVectorizer(stop_words=None, max_features=5000, lowercase=True)
    m = MultinomialNB(alpha=1.0).fit(v.fit_transform(Xtr), ytr)
    yp = m.predict(v.transform(Xte))
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    acc = accuracy_score(yte, yp) * 100
    pre = precision_score(yte, yp, zero_division=0) * 100
    rec = recall_score(yte, yp, zero_division=0) * 100
    f1 = f1_score(yte, yp, zero_division=0) * 100
    print(f"=== {tag} (dataset {len(df)}, uji {len(Xte)}) ===")
    print(f"TP {tp} | FN {fn} | FP {fp} | TN {tn}")
    print(f"Akurasi {fmt(acc)}% | Presisi {fmt(pre)}% | Recall {fmt(rec)}% | F1 {fmt(f1)}%")
    print()
    return dict(acc=acc, pre=pre, rec=rec, f1=f1, tp=tp, fn=fn, fp=fp, tn=tn)


def main():
    df_main = load_main_dataset()
    df_id_unik = get_indonesian_examples().iloc[:30].copy()   # 30 contoh unik

    # 1) TANPA leakage: gabung unik, lalu split, baru duplikasi contoh ID
    #    ke sisi masing-masing (latih & uji masing-masing x2) -> tidak ada
    #    salinan yang sama di kedua sisi.
    df_clean = pd.concat([df_main, df_id_unik], ignore_index=True)
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
    hasil_bersih = eval_df(df_clean, "TANPA LEAKAGE (basis unik)")

    # ekstra: hasil pada versi dengan salinan x3 tiap sisi (lebih dekat ke
    # kondisi pelatihan aplikasi), tanpa kontaminasi silang
    X, y = df_clean["text"].astype(str), df_clean["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    df_train_leak = pd.DataFrame({"text": Xtr, "label": ytr})
    df_test_leak = pd.DataFrame({"text": Xte, "label": yte})
    # duplikasi contoh ID di sisi latih dan sisi uji secara terpisah (x2 agar seimbang)
    is_id = df_train_leak["text"].isin(df_id_unik["text"])
    df_train_dup = pd.concat([df_train_leak, df_train_leak[is_id], df_train_leak[is_id]],
                             ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    is_id_te = df_test_leak["text"].isin(df_id_unik["text"])
    df_test_dup = pd.concat([df_test_leak, df_test_leak[is_id_te]],
                            ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    Xtr_f, ytr_f = df_train_dup["text"].astype(str), df_train_dup["label"]
    Xte_f, yte_f = df_test_dup["text"].astype(str), df_test_dup["label"]
    v = TfidfVectorizer(stop_words=None, max_features=5000, lowercase=True)
    m = MultinomialNB(alpha=1.0).fit(v.fit_transform(Xtr_f), ytr_f)
    yp2 = m.predict(v.transform(Xte_f))
    tn, fp, fn, tp = confusion_matrix(yte_f, yp2).ravel()
    print(f"=== LATIH DUP-LATIH ({len(df_train_dup)}), UJI UNIK + DUP-UJI ({len(df_test_dup)}) ===")
    print(f"TP {tp} | FN {fn} | FP {fp} | TN {tn}")
    print(f"Akurasi {fmt(accuracy_score(yte_f,yp2)*100)}% | Presisi {fmt(precision_score(yte_f,yp2,zero_division=0)*100)}%"
          f" | Recall {fmt(recall_score(yte_f,yp2,zero_division=0)*100)}% | F1 {fmt(f1_score(yte_f,yp2,zero_division=0)*100)}%")
    print()

    # per-kelompok bahasa pada skema TANPA leakage
    df_clean["is_id"] = df_clean["text"].isin(df_id_unik["text"])
    Xc = df_clean["text"].astype(str); yc = df_clean["label"]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=42)
    v2 = TfidfVectorizer(stop_words=None, max_features=5000, lowercase=True)
    m2 = MultinomialNB(alpha=1.0).fit(v2.fit_transform(Xtr2), ytr2)
    te_df = pd.DataFrame({"text": Xte2, "is_id": df_clean.loc[Xte2.index, "is_id"].values,
                          "label": yte2.values})
    te_df["pred"] = m2.predict(v2.transform(Xte2))

    for tag, sub in [("INDONESIA", te_df[te_df.is_id]), ("INGGRIS", te_df[~te_df.is_id])]:
        if len(sub) == 0:
            continue
        acc = accuracy_score(sub.label, sub.pred) * 100
        tp = int(((sub.label == 1) & (sub.pred == 1)).sum())
        fn = int(((sub.label == 1) & (sub.pred == 0)).sum())
        fp = int(((sub.label == 0) & (sub.pred == 1)).sum())
        tn = int(((sub.label == 0) & (sub.pred == 0)).sum())
        print(f"--- Subgrup {tag} (uji {len(sub)}) ---")
        print(f"TP {tp} FN {fn} FP {fp} TN {tn} | Akurasi {fmt(acc)}%")
        print()

    # 3) EN-only baseline
    eval_df(df_main.sample(frac=1, random_state=42).reset_index(drop=True),
            "BASELINE EN-ONLY")


if __name__ == "__main__":
    main()
