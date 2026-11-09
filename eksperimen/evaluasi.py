"""
evaluasi.py — Semua angka evaluasi untuk artikel, dapat direproduksi.
=====================================================================
Jalankan dari folder proyek (folder yang berisi app.py dan analyzer/):

    python eksperimen/evaluasi.py

Yang dikerjakan:
  1. Bagi dataset SMS Indonesia 80:20 (stratified, random_state=43).
  2. Latih ulang Naive Bayes (= model yang dipakai aplikasi) lalu uji pada 20% data uji.
     Bandingkan dengan model lama (dilatih data Inggris) bila ada di eksperimen/versi_lama/.
  3. Validasi silang 5-fold NB pada seluruh dataset (untuk melihat kestabilan).
  4. Pemilihan aturan fusi skor dengan validasi silang 5-fold di DATA LATIH saja.
  5. Evaluasi sistem terintegrasi pada data uji, mode offline:
     reputasi domain (S_C) dan rantai redirect (S_D) dimatikan, karena SMS di dataset
     berasal dari 2013-2016 dan domainnya sekarang sudah mati/berganti pemilik,
     sehingga hasil WHOIS/redirect hari ini tidak mencerminkan kondisi saat SMS dikirim.
  6. Simpan semua hasil ke eksperimen/hasil/.
"""

import importlib.util
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from analyzer import scoring_engine as se  # noqa: E402
from analyzer import text_classifier as tc  # noqa: E402
from analyzer import sms_analyzer, url_analyzer  # noqa: E402
from analyzer.url_extractor import extract_urls  # noqa: E402

OUT_DIR = os.path.join(ROOT, "eksperimen", "hasil")
OLD_DIR = os.path.join(ROOT, "eksperimen", "versi_lama")
CV_SEED = 43


# ── Utilitas ────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "akurasi": accuracy_score(y_true, y_pred),
        "presisi": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def fmt(m: dict) -> str:
    return (f"TP={m['TP']:>3} FN={m['FN']:>3} FP={m['FP']:>3} TN={m['TN']:>3} | "
            f"Akurasi {m['akurasi']*100:6.2f}%  Presisi {m['presisi']*100:6.2f}%  "
            f"Recall {m['recall']*100:6.2f}%  F1 {m['f1']*100:6.2f}%")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def system_scores(texts, probs, params=None, analyzer=None, url_mod=None, scoring=None):
    """Skor sistem terintegrasi mode offline (S_C dan S_D tidak dihitung)."""
    analyzer = analyzer or sms_analyzer
    url_mod = url_mod or url_analyzer
    scoring = scoring or se
    rows = []
    for text, p in zip(texts, probs):
        ent = analyzer.extract_sms_entities(text)
        ua = url_mod.analyze_urls(extract_urls(text))
        kwargs = {"params": params} if params is not None else {}
        r = scoring.calculate_score(ua, {}, [], ent, float(p), **kwargs)
        rows.append({"skor": r["total_score"], "status": r["status"], "ada_tautan": bool(ua),
                     **{f"{k}": v["score"] for k, v in r["breakdown"].items()}})
    return pd.DataFrame(rows)


def oof_probabilities(df_part: pd.DataFrame) -> np.ndarray:
    """Probabilitas out-of-fold: tiap SMS dinilai model yang tidak pernah melihatnya."""
    df_part = df_part.reset_index(drop=True)
    oof = np.zeros(len(df_part))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for tr_idx, va_idx in skf.split(df_part, df_part["label"]):
        train = tc.build_training_set(df_part.iloc[tr_idx])
        model, vec = tc.fit_model(train["text"], train["label"])
        oof[va_idx] = model.predict_proba(vec.transform(df_part.iloc[va_idx]["text"]))[:, 1]
    return oof


# ── Utama ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hasil = {}

    df = tc.load_indonesian_dataset()
    df_train, df_test = tc.split_dataset(df)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print("=" * 100)
    print(f"DATA: {len(df)} SMS unik | latih {len(df_train)} | uji {len(df_test)}")
    print("Kategori asli di data uji:", df_test["kategori_asli"].value_counts().to_dict())
    hasil["data"] = {
        "total_unik": len(df), "latih": len(df_train), "uji": len(df_test),
        "kategori_total": df["kategori_asli"].value_counts().to_dict(),
        "kategori_uji": df_test["kategori_asli"].value_counts().to_dict(),
        "penipuan_uji": int(df_test["label"].sum()),
        "bukan_penipuan_uji": int((df_test["label"] == 0).sum()),
    }

    # 1) Latih model aplikasi
    model, vec = tc.train_and_save_model()
    hasil["data"]["latih_akhir"] = len(tc.build_training_set(df_train))
    hasil["data"]["vocab"] = len(vec.vocabulary_)
    p_new = model.predict_proba(vec.transform(df_test["text"]))[:, 1]

    # 2) Evaluasi NB pada data uji
    print("\n" + "=" * 100 + "\n[TABEL 3] Naive Bayes pada data uji Indonesia (ambang 0,5)")
    m_new = metrics(df_test["label"], (p_new >= 0.5).astype(int))
    print("  Model baru (data latih Indonesia) :", fmt(m_new))
    hasil["nb_model_baru"] = m_new

    p_old = None
    if os.path.exists(os.path.join(OLD_DIR, "model_nb_lama.pkl")):
        old_model = joblib.load(os.path.join(OLD_DIR, "model_nb_lama.pkl"))
        old_vec = joblib.load(os.path.join(OLD_DIR, "vectorizer_lama.pkl"))
        p_old = old_model.predict_proba(old_vec.transform(df_test["text"]))[:, 1]
        m_old = metrics(df_test["label"], (p_old >= 0.5).astype(int))
        print("  Model lama (data latih Inggris)   :", fmt(m_old))
        hasil["nb_model_lama"] = m_old

    # 3) Validasi silang 5-fold pada seluruh dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    folds = []
    for tr_idx, va_idx in skf.split(df, df["label"]):
        train = tc.build_training_set(df.iloc[tr_idx])
        m, v = tc.fit_model(train["text"], train["label"])
        pred = (m.predict_proba(v.transform(df.iloc[va_idx]["text"]))[:, 1] >= 0.5).astype(int)
        folds.append(metrics(df.iloc[va_idx]["label"], pred))
    cv = {k: {"rata2": float(np.mean([f[k] for f in folds])), "sd": float(np.std([f[k] for f in folds]))}
          for k in ["akurasi", "presisi", "recall", "f1"]}
    print("  5-fold CV (seluruh data)          :",
          "  ".join(f"{k.capitalize()} {v['rata2']*100:.2f}±{v['sd']*100:.2f}%" for k, v in cv.items()))
    hasil["nb_cv5"] = cv

    # 4) Pemilihan aturan fusi — hanya data latih (probabilitas out-of-fold)
    print("\n" + "=" * 100 + "\n[PEMILIHAN ATURAN] Validasi silang 5-fold di DATA LATIH (data uji tidak disentuh)")
    oof = oof_probabilities(df_train)
    kandidat = [
        ("A. Aturan v4 (NB x0,4 tanpa tautan)",
         {"nb_min_score": None, "nb_factor_no_url": 0.4, "link_bonus_needs_evidence": False}),
        ("B. A + pengali 0,4 dihapus",
         {"nb_min_score": None, "nb_factor_no_url": 1.0, "link_bonus_needs_evidence": False}),
        ("C. B + NB>=0,5 minimal 31 poin",
         {"nb_min_score": 31, "nb_factor_no_url": 1.0, "link_bonus_needs_evidence": False}),
        ("D. C + bonus tautan butuh bukti kedua (v5)",
         {"nb_min_score": 31, "nb_factor_no_url": 1.0, "link_bonus_needs_evidence": True}),
    ]
    hasil["pemilihan_aturan"] = []
    for nama, params in kandidat:
        s = system_scores(df_train["text"], oof, params=params)
        m = metrics(df_train["label"], (s["status"] != "AMAN").astype(int))
        print(f"  {nama:46s}", fmt(m))
        hasil["pemilihan_aturan"].append({"nama": nama, "params": params, **m})

    # 5) Evaluasi sistem terintegrasi pada data uji (offline)
    print("\n" + "=" * 100 + "\n[TABEL 4] Sistem pada data uji (offline: S_C & S_D dimatikan). Positif = Waspada/Berbahaya")
    y = df_test["label"]
    # Heuristik saja: tanpa NB, bonus tautan+kata kunci diberikan tanpa syarat (aturan heuristik murni)
    s_heur = system_scores(df_test["text"], np.zeros(len(df_test)), params={"link_bonus_needs_evidence": False})
    s_full = system_scores(df_test["text"], p_new)
    rows = {
        "Heuristik saja (S_A+S_B, tanpa NB)": metrics(y, (s_heur["status"] != "AMAN").astype(int)),
        "Naive Bayes saja": metrics(y, (p_new >= 0.5).astype(int)),
        "Sistem gabungan v5": metrics(y, (s_full["status"] != "AMAN").astype(int)),
    }
    s_old = None
    if p_old is not None:
        old_an = _load_module("sms_analyzer_lama", os.path.join(OLD_DIR, "sms_analyzer_lama.py"))
        old_url = _load_module("url_analyzer_lama", os.path.join(OLD_DIR, "url_analyzer_lama.py"))
        old_sc = _load_module("scoring_engine_lama", os.path.join(OLD_DIR, "scoring_engine_lama.py"))
        s_old = system_scores(df_test["text"], p_old, analyzer=old_an, url_mod=old_url, scoring=old_sc)
        rows = {"Sistem lama (v4 + model Inggris)": metrics(y, (s_old["status"] != "AMAN").astype(int)), **rows}
    for nama, m in rows.items():
        print(f"  {nama:34s}", fmt(m))
    hasil["sistem_uji"] = rows

    # Rincian: dengan/tanpa tautan, dan sebaran status per kategori asli
    print("\n  Deteksi penipuan menurut keberadaan tautan (recall):")
    hasil["sistem_per_tautan"] = {}
    for nama, s in [("lama", s_old), ("v5", s_full)]:
        if s is None:
            continue
        d = {}
        for ada in [True, False]:
            mask = (y == 1).values & (s["ada_tautan"] == ada).values
            n, k = int(mask.sum()), int((s.loc[mask, "status"] != "AMAN").sum())
            d["dengan_tautan" if ada else "tanpa_tautan"] = {"terdeteksi": k, "total": n}
        hasil["sistem_per_tautan"][nama] = d
        print(f"    {nama:5s}: dengan tautan {d['dengan_tautan']['terdeteksi']}/{d['dengan_tautan']['total']}"
              f" | tanpa tautan {d['tanpa_tautan']['terdeteksi']}/{d['tanpa_tautan']['total']}")

    print("\n  Sebaran status sistem v5 per kategori asli (data uji):")
    tab = pd.crosstab(df_test["kategori_asli"], s_full["status"]).reindex(columns=["AMAN", "WASPADA", "BERBAHAYA"], fill_value=0)
    print(tab.to_string().replace("\n", "\n    "))
    hasil["sebaran_v5"] = tab.to_dict(orient="index")
    if s_old is not None:
        tab_old = pd.crosstab(df_test["kategori_asli"], s_old["status"]).reindex(columns=["AMAN", "WASPADA", "BERBAHAYA"], fill_value=0)
        print("  Sebaran status sistem lama per kategori asli (data uji):")
        print(tab_old.to_string().replace("\n", "\n    "))
        hasil["sebaran_lama"] = tab_old.to_dict(orient="index")

    # 6) Kestabilan: 100 pembagian acak berbeda (apakah random_state 43 kebetulan bagus?)
    print("\n" + "=" * 100 + "\n[KESTABILAN] Hold-out 80:20 diulang dengan 100 random_state berbeda (0-99)")
    from sklearn.model_selection import train_test_split
    ent_all = {t: sms_analyzer.extract_sms_entities(t) for t in df["text"]}
    ua_all = {t: url_analyzer.analyze_urls(extract_urls(t)) for t in df["text"]}
    sweep = []
    for seed in range(100):
        tr_s, te_s = train_test_split(df, test_size=tc.TEST_SIZE, stratify=df["label"], random_state=seed)
        train = tc.build_training_set(tr_s)
        m, v = tc.fit_model(train["text"], train["label"])
        p = m.predict_proba(v.transform(te_s["text"]))[:, 1]
        sys_pred = [se.calculate_score(ua_all[t], {}, [], ent_all[t], float(q))["status"] != "AMAN"
                    for t, q in zip(te_s["text"], p)]
        sweep.append({"seed": seed, "nb_f1": f1_score(te_s["label"], p >= 0.5), "sistem_f1": f1_score(te_s["label"], sys_pred)})
    sw = pd.DataFrame(sweep)
    hasil["kestabilan_100_split"] = {}
    for col, nama in [("nb_f1", "Naive Bayes"), ("sistem_f1", "Sistem v5")]:
        v43 = float(sw.loc[sw.seed == tc.RANDOM_STATE, col].iloc[0])
        pct = float((sw[col] < v43).mean() * 100)
        hasil["kestabilan_100_split"][col] = {"rata2": float(sw[col].mean()), "sd": float(sw[col].std()),
                                               "min": float(sw[col].min()), "maks": float(sw[col].max()),
                                               "nilai_seed_43": v43, "persentil_seed_43": pct}
        print(f"  {nama:12s} F1 rata-rata {sw[col].mean()*100:.2f} ± {sw[col].std()*100:.2f} "
              f"(min {sw[col].min()*100:.2f}, maks {sw[col].max()*100:.2f}) | seed 43 = {v43*100:.2f}, persentil {pct:.0f}")

    # 7) Apakah dataset Inggris membantu? Diputuskan dengan validasi silang di DATA LATIH saja
    if os.path.exists(tc.DATASET_EN_CSV):
        print("\n" + "=" * 100 + "\n[DATA INGGRIS] Validasi silang 5-fold di data latih: Indonesia saja vs + dataset Inggris")
        en = tc.load_english_dataset()
        hasil["data_inggris_cv_latih"] = {}
        for nama, extra in [("Indonesia saja", None), ("Indonesia + Inggris", en)]:
            oof_x = np.zeros(len(df_train))
            for tr_idx, va_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED).split(df_train, df_train["label"]):
                train = tc.build_training_set(df_train.iloc[tr_idx])
                if extra is not None:
                    train = pd.concat([train, extra], ignore_index=True)
                m, v = tc.fit_model(train["text"], train["label"])
                oof_x[va_idx] = m.predict_proba(v.transform(df_train.iloc[va_idx]["text"]))[:, 1]
            mm = metrics(df_train["label"], (oof_x >= 0.5).astype(int))
            hasil["data_inggris_cv_latih"][nama] = mm
            print(f"  NB {nama:20s}", fmt(mm))

    # 8) Simpan
    per_sms = df_test.copy()
    per_sms["nb_baru"] = p_new.round(4)
    per_sms = pd.concat([per_sms, s_full.add_prefix("v5_")], axis=1)
    if s_old is not None:
        per_sms["nb_lama"] = p_old.round(4)
        per_sms = pd.concat([per_sms, s_old[["skor", "status"]].add_prefix("lama_")], axis=1)
    per_sms.to_csv(os.path.join(OUT_DIR, "prediksi_data_uji.csv"), index=False)
    with open(os.path.join(OUT_DIR, "hasil_evaluasi.json"), "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2, default=float)
    print("\nHasil disimpan di", OUT_DIR)


if __name__ == "__main__":
    main()
