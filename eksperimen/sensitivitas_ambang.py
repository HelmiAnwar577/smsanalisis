"""
sensitivitas_ambang.py — Seberapa peka hasil terhadap batas Aman/Waspada (30)?
==============================================================================
    python eksperimen/sensitivitas_ambang.py

Skor sistem v6 dihitung sekali (data 2026: out-of-fold, 5-fold x 5 ulangan;
data uji lama: model yang dilatih tanpa data uji), lalu batas "Aman maksimum"
digeser dari 10 sampai 60. Positif = skor di atas batas.
Juga dicatat: kategori tunggal mana pun maksimal 40 poin, sehingga status Berbahaya (>65)
secara matematis selalu membutuhkan bukti dari minimal dua kategori.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from analyzer import scoring_engine as se  # noqa: E402
from analyzer import sms_analyzer, url_analyzer  # noqa: E402
from analyzer import text_classifier as tc  # noqa: E402
from analyzer.url_extractor import extract_urls  # noqa: E402

REPEATS = 5


def scores(texts, probs, cache):
    return np.array([se.calculate_score(cache[t][1], {}, [], cache[t][0], float(p))["total_score"]
                     for t, p in zip(texts, probs)])


def f1(y, pred):
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0, tp, fp


def main():
    df = tc.load_indonesian_dataset()
    df_train, df_test = tc.split_dataset(df)
    d26 = tc.load_data_2026(include_dev=True)
    dev = d26[d26.file == "sms_judol_pengembangan.csv"][["text", "label"]]
    new = d26[d26.file != "sms_judol_pengembangan.csv"].reset_index(drop=True)
    cache = {t: (sms_analyzer.extract_sms_entities(t), url_analyzer.analyze_urls(extract_urls(t)))
             for t in list(new.text) + list(df_test.text)}
    y_new, y_old = new.label.values == 1, df_test.label.values == 1

    s_new, s_old = [], []
    for rep in range(REPEATS):
        sc = np.zeros(len(new))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(new, new.label):
            train = tc.build_training_set(df_train, pd.concat([new.iloc[a], dev], ignore_index=True))
            m, v = tc.fit_model(train.text, train.label)
            sc[b] = scores(new.iloc[b].text, m.predict_proba(v.transform(new.iloc[b].text))[:, 1], cache)
            s_old.append(scores(df_test.text, m.predict_proba(v.transform(df_test.text))[:, 1], cache))
        s_new.append(sc)

    print(f"{'Batas Aman':>10} | {'F1 2026':>8} {'TP':>5} {'FP':>5} | {'F1 lama':>8} {'TP':>5} {'FP':>5}")
    rows = []
    for th in range(10, 61, 5):
        a = np.mean([f1(y_new, s > th) for s in s_new], axis=0)
        b = np.mean([f1(y_old, s > th) for s in s_old], axis=0)
        rows.append({"batas_aman": th, "f1_2026": a[0], "tp_2026": a[1], "fp_2026": a[2],
                     "f1_lama": b[0], "tp_lama": b[1], "fp_lama": b[2]})
        mark = "  <- dipakai" if th == se.STATUS_THRESHOLDS["aman_max"] else ""
        print(f"{th:>10} | {a[0]*100:7.2f}% {a[1]:5.1f} {a[2]:5.1f} | {b[0]*100:7.2f}% {b[1]:5.1f} {b[2]:5.1f}{mark}")
    out = os.path.join(ROOT, "eksperimen", "hasil", "sensitivitas_ambang.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\nBatas maksimum tiap kategori:", se.CAPS, "-> tidak ada satu kategori yang bisa melewati 65 sendirian.")
    allnew = np.concatenate(s_new)
    for nama, mask in [("penipuan 2026", np.tile(y_new, REPEATS)), ("sah 2026", ~np.tile(y_new, REPEATS))]:
        s = allnew[mask]
        print(f"Sebaran status {nama:14s}: Aman {np.mean(s <= 30)*100:5.1f}% | Waspada {np.mean((s > 30) & (s <= 65))*100:5.1f}% "
              f"| Berbahaya {np.mean(s > 65)*100:5.1f}%")
    print("Disimpan:", out)


if __name__ == "__main__":
    main()
