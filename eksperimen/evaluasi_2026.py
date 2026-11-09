"""
evaluasi_2026.py — Evaluasi sistem pada SMS 2026 + uji ablasi (dapat direproduksi).
=================================================================================
Jalankan dari folder proyek:

    python eksperimen/evaluasi_2026.py

Data:
  - Data uji LAMA : 20% dataset Indonesia 2013-2016 (229 SMS), sama dengan evaluasi.py.
  - Data 2026     : analyzer/data/data_2026/ (penipuan + sah, tanpa SMS berlabel "ragu").
                    5 SMS judol pengembangan TIDAK ikut diuji; hanya ikut data latih.

Cara uji data 2026: validasi silang berstrata 5-fold, diulang 10x dengan acakan berbeda.
Pada tiap lipatan, 4/5 data 2026 boleh ikut melatih Naive Bayes, 1/5 sisanya diuji.
Jadi setiap SMS 2026 dinilai oleh model yang TIDAK PERNAH melihatnya.

Uji ablasi (perbaikan ditambahkan satu per satu):
  A. NB per kata, tanpa data 2026, tanpa deteksi tautan mirip  (= sistem sebelum perbaikan)
  B. A + NB per potongan huruf
  C. B + data 2026 di data latih
  D. C + deteksi tautan mirip / typosquatting
  E. D + aturan fusi direvisi (tautan kritis -> Waspada; bit.ly bukan bukti kedua) (= sistem v6)
Untuk D dan E juga dihitung komponen tunggal: NB saja dan heuristik saja.

Semua mode offline (S_C reputasi domain & S_D redirect dimatikan), sama dengan evaluasi.py.
"""

import json
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

OUT_DIR = os.path.join(ROOT, "eksperimen", "hasil")
REPEATS = 10

CONFIGS = [
    ("A. Sebelum perbaikan (NB per kata)", dict(feature="kata", use_2026=False, typo=False, params=se.PARAMS_V5)),
    ("B. A + NB per potongan huruf",       dict(feature="huruf", use_2026=False, typo=False, params=se.PARAMS_V5)),
    ("C. B + data 2026 di data latih",     dict(feature="huruf", use_2026=True, typo=False, params=se.PARAMS_V5)),
    ("D. C + deteksi tautan mirip",        dict(feature="huruf", use_2026=True, typo=True, params=se.PARAMS_V5)),
    ("E. D + aturan fusi direvisi (v6)",   dict(feature="huruf", use_2026=True, typo=True, params={})),
]


def metrics(tp, fp, fn, tn):
    pre = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "presisi": pre, "recall": rec, "f1": f1,
            "akurasi": (tp + tn) / (tp + fp + fn + tn)}


def system_flags(texts, probs, cache, params=None):
    out = []
    for t, p in zip(texts, probs):
        ent, ua = cache[t]
        out.append(se.calculate_score(ua, {}, [], ent, float(p), params=params)["status"] != "AMAN")
    return np.array(out)


def heuristic_flags(texts, cache, params=None):
    return system_flags(texts, np.zeros(len(texts)), cache, params)


def build_cache(texts):
    return {t: (sms_analyzer.extract_sms_entities(t), url_analyzer.analyze_urls(extract_urls(t))) for t in texts}


def run_config(cfg, df_train, df_test, new, dev):
    url_analyzer.TYPOSQUAT_ENABLED = cfg["typo"]
    cache = build_cache(list(new.text) + list(df_test.text))
    y_new = new.label.values == 1
    y_old = df_test.label.values == 1
    acc = {k: np.zeros(4) for k in ["sistem_2026", "sistem_lama", "nb_2026", "nb_lama", "heur_2026", "heur_lama"]}
    oof_status = None

    for rep in range(REPEATS):
        pred_sys = np.zeros(len(new), bool)
        pred_nb = np.zeros(len(new), bool)
        old_counts = np.zeros(4)
        old_nb = np.zeros(4)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(new, new.label):
            extra = pd.concat([new.iloc[a], dev], ignore_index=True) if cfg["use_2026"] else None
            train = tc.build_training_set(df_train, extra)
            model, vec = tc.fit_model(train.text, train.label, feature=cfg["feature"])
            p = model.predict_proba(vec.transform(new.iloc[b].text))[:, 1]
            pred_nb[b] = p >= 0.5
            pred_sys[b] = system_flags(new.iloc[b].text, p, cache, cfg["params"])
            po = model.predict_proba(vec.transform(df_test.text))[:, 1]
            fo = system_flags(df_test.text, po, cache, cfg["params"])
            old_counts += [(fo & y_old).sum(), (fo & ~y_old).sum(), (~fo & y_old).sum(), (~fo & ~y_old).sum()]
            no = po >= 0.5
            old_nb += [(no & y_old).sum(), (no & ~y_old).sum(), (~no & y_old).sum(), (~no & ~y_old).sum()]
        for key, pr, y in [("sistem_2026", pred_sys, y_new), ("nb_2026", pred_nb, y_new)]:
            acc[key] += [(pr & y).sum(), (pr & ~y).sum(), (~pr & y).sum(), (~pr & ~y).sum()]
        acc["sistem_lama"] += old_counts / 5
        acc["nb_lama"] += old_nb / 5
        if rep == 0:
            oof_status = pred_sys.copy()

    h = heuristic_flags(new.text, cache, cfg["params"])
    acc["heur_2026"] = np.array([(h & y_new).sum(), (h & ~y_new).sum(), (~h & y_new).sum(), (~h & ~y_new).sum()]) * REPEATS
    h = heuristic_flags(df_test.text, cache, cfg["params"])
    acc["heur_lama"] = np.array([(h & y_old).sum(), (h & ~y_old).sum(), (~h & y_old).sum(), (~h & ~y_old).sum()]) * REPEATS
    res = {k: metrics(*(v / REPEATS)) for k, v in acc.items()}
    url_analyzer.TYPOSQUAT_ENABLED = True
    return res, oof_status


def fmt(m):
    return (f"TP={m['TP']:5.1f} FP={m['FP']:5.1f} FN={m['FN']:5.1f} | Presisi {m['presisi']*100:6.2f}%  "
            f"Recall {m['recall']*100:6.2f}%  F1 {m['f1']*100:6.2f}%")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = tc.load_indonesian_dataset()
    df_train, df_test = tc.split_dataset(df)
    df_test = df_test.reset_index(drop=True)
    d26 = tc.load_data_2026(include_dev=True)
    dev = d26[d26.file == "sms_judol_pengembangan.csv"][["text", "label"]]
    new = d26[d26.file != "sms_judol_pengembangan.csv"].reset_index(drop=True)

    print("=" * 110)
    print(f"Data uji lama: {len(df_test)} SMS ({int(df_test.label.sum())} penipuan) | "
          f"Data 2026 diuji: {len(new)} SMS ({int(new.label.sum())} penipuan, {int((new.label == 0).sum())} sah) | "
          f"pengembangan (hanya latih): {len(dev)}")
    print("Data 2026 per jenis:", new.groupby(["label", "jenis"]).size().to_dict())
    hasil = {"data": {"uji_lama": len(df_test), "uji_2026": len(new), "penipuan_2026": int(new.label.sum()),
                      "sah_2026": int((new.label == 0).sum()), "pengembangan": len(dev),
                      "jenis_2026": {f"{k[0]}_{k[1]}": int(v) for k, v in new.groupby(["label", "jenis"]).size().items()}},
             "ablasi": {}}

    for name, cfg in CONFIGS:
        res, oof = run_config(cfg, df_train, df_test, new, dev)
        hasil["ablasi"][name] = res
        print(f"\n{name}")
        print(f"   Sistem, data 2026 : {fmt(res['sistem_2026'])}")
        print(f"   Sistem, data lama : {fmt(res['sistem_lama'])}")
        if name[0] in "DE":
            print(f"   NB saja, 2026     : {fmt(res['nb_2026'])}")
            print(f"   NB saja, lama     : {fmt(res['nb_lama'])}")
            print(f"   Heuristik, 2026   : {fmt(res['heur_2026'])}")
            print(f"   Heuristik, lama   : {fmt(res['heur_lama'])}")
        if name.startswith("E"):
            out = new.copy()
            out["prediksi_v6"] = np.where(oof, "WASPADA/BERBAHAYA", "AMAN")
            out["benar"] = (oof == (out.label == 1))
            out.to_csv(os.path.join(OUT_DIR, "prediksi_2026_v6.csv"), index=False)
            print("\n   Kesalahan v6 (ulangan pertama):")
            for _, r in out[~out.benar].iterrows():
                print(f"     [{'penipuan lolos' if r.label == 1 else 'salah tuduh'}] ({r.jenis}) {r.text[:100]}")

    with open(os.path.join(OUT_DIR, "hasil_evaluasi_2026.json"), "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2, default=float)
    print("\nHasil disimpan di", OUT_DIR)


if __name__ == "__main__":
    main()
