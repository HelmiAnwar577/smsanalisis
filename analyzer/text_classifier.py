"""
text_classifier.py v3.0
=======================
v3.0: TF-IDF per potongan huruf (char n-gram 2-4) + Multinomial Naive Bayes (alpha 0,01),
dan data SMS 2026 ikut dilatih. Algoritma tetap Multinomial Naive Bayes.

v2.1: teks dinormalisasi (text_normalizer.py) sebelum TF-IDF — "Hdiah", "ju'ta"
dibaca sebagai "hadiah", "juta". Atur lewat USE_NORMALIZER.

Klasifikasi teks SMS dengan TF-IDF + Multinomial Naive Bayes.
Keluaran: probabilitas pesan bergaya penipuan (0-1).

Perubahan dari v1:
- Data latih utama kini dataset SMS berbahasa Indonesia
  (dataset_sms_spam_v1.csv, 1.143 SMS; label 0 normal, 1 penipuan, 2 promo).
  Label dipetakan: 1 penipuan -> 1 | 0 normal & 2 promo -> 0 (bukan penipuan).
  Alasan: sistem mendeteksi SMISHING/penipuan, bukan spam promosi. Promo resmi
  operator seharusnya berstatus Aman.
- Data dibagi 80:20 (stratified, random_state=43) SEBELUM pelatihan.
  Model yang disimpan = model yang dilatih pada 80% data latih, sehingga angka
  evaluasi di artikel adalah angka model yang benar-benar dipakai aplikasi.
- 30 contoh SMS buatan peneliti (pola modus baru: bansos, kurir APK, WhatsApp)
  HANYA masuk data latih, tidak pernah masuk data uji. Contoh ini tetap
  digandakan 3x (oversampling) agar bobotnya cukup terhadap ~900 SMS lama
  (2013-2016). Di v1 salinannya bisa jatuh ke data uji sehingga nilai uji
  terlalu optimis; di v2 hal itu tidak mungkin terjadi.
- Dataset Inggris (Mendeley) opsional lewat USE_ENGLISH_DATASET (default False).
- Import ganda dihapus, path dibuat relatif terhadap file ini, dan model di-cache
  di memori agar tidak dibaca ulang dari disk setiap ada permintaan.
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from analyzer.text_normalizer import normalize_for_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_nb.pkl")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
DATASET_ID_CSV = os.path.join(BASE_DIR, "data", "dataset_sms_spam_v1.csv")
DATASET_EN_CSV = os.path.join(BASE_DIR, "data", "Dataset_5971.csv")

USE_ENGLISH_DATASET = False   # True = tambahkan dataset Mendeley (EN) ke data latih
USE_NORMALIZER = True         # v2.1: normalisasi ejaan samaran sebelum TF-IDF

# v3.0 — cara memotong teks sebelum TF-IDF + Multinomial Naive Bayes
#   "kata"  : per kata utuh (v1-v2). "FR33CHIPS" dan "freechips" = dua kata berbeda.
#   "huruf" : per potongan 2-4 huruf (char n-gram). "fr33chips" dan "freechips" berbagi
#             potongan "chi", "hip", "ips", ... sehingga tetap dikenali mirip.
# ngram (2,4) dan alpha 0,01 dipilih dengan validasi silang 5-fold di DATA LATIH LAMA saja
# (F1 93,85% vs 87,35% untuk "kata"); lihat eksperimen/evaluasi_2026.py.
FEATURE = "huruf"
CHAR_NGRAM = (2, 4)
NB_ALPHA = {"kata": 1.0, "huruf": 0.01}

# v3.0 — data SMS 2026 (analyzer/data/data_2026/) ikut dilatih, digandakan 3x
# (sama dengan contoh peneliti) agar tidak tenggelam di antara ~1.000 SMS lama.
USE_DATA_2026 = True
DATA_2026_OVERSAMPLE = 3
DATA_2026_DIR = os.path.join(BASE_DIR, "data", "data_2026")
MANUAL_OVERSAMPLE = 3         # berapa kali 30 contoh peneliti digandakan di data latih
TEST_SIZE = 0.20
RANDOM_STATE = 43

_cache = {"model": None, "vectorizer": None}


# ── Data ────────────────────────────────────────────────────────────────────
def get_indonesian_examples() -> pd.DataFrame:
    """30 contoh pola modus lokal yang disusun peneliti (15 penipuan, 15 sah). Hanya untuk data latih."""
    spam_data = [
        "Selamat! Nomor Anda terpilih memenangkan hadiah Rp 100 Juta dari Bank BNI. Klik bit.ly/hadiah untuk klaim.",
        "PESAN RESMI: Rekening Anda akan diblokir malam ini. Segera verifikasi data Anda di link berikut.",
        "Dapatkan pinjaman dana tunai hingga 50 juta tanpa agunan. Bunga rendah, cair cepat. Hubungi 0812345678",
        "Paket Anda gagal dikirim. Silakan cek resi dan jadwal ulang pengiriman melalui aplikasi kami: klik tautan ini",
        "Tolong isikan pulsa 50rb ke nomor ini, ini bapak lagi di kantor polisi darurat.",
        "Promo spesial akhir tahun! Diskon 90% untuk semua produk. Beli sekarang di www.tokopalsu.com",
        "Anda mendapat bantuan sosial (Bansos) Rp 600.000. Cairkan dana Anda melalui link resmi ini segera.",
        "INFO: Tagihan kartu kredit Anda sebesar Rp 8.450.000 jatuh tempo hari ini. Abaikan jika sudah membayar.",
        "Selamat siang, kami dari kurir J&T. Ada paket atas nama Anda. Silakan download foto paket disini (APK).",
        "Segera tukarkan poin Telkomsel Anda sebelum hangus malam ini. Klik link untuk menukar dengan pulsa.",
        "Tingkatkan limit kartu kredit BCA Anda hingga 100 juta. Proses 5 menit. Klik disini.",
        "Undangan eksklusif! Gabung di grup VIP Trading kami dan dapatkan profit 10 juta per hari. Slot terbatas.",
        "Mohon maaf, layanan internet Anda akan diputus karena tunggakan. Bayar sekarang via link ini.",
        "Peringatan! Akun WhatsApp Anda telah didaftarkan di perangkat lain. Klik link untuk membatalkan.",
        "Selamat! Anda mendapatkan Cashback Rp 500rb. Klaim sekarang melalui link berikut ini.",
    ]
    ham_data = [
        "Halo bro, nanti malam jadi futsal jam 8 ya. Jangan sampai telat.",
        "Mama minta tolong belikan gula sekalian pas pulang kerja ya.",
        "OTP Anda adalah 482910. Jangan berikan kode ini kepada siapapun.",
        "Terima kasih telah berbelanja di minimarket kami. Total belanja Anda Rp 45.000.",
        "Rapat tim diundur menjadi jam 2 siang di ruang meeting utama.",
        "Pak, berkas laporannya sudah saya taruh di atas meja bapak. Terima kasih.",
        "Selamat pagi, mengingatkan jadwal kontrol dokter gigi besok jam 10 pagi.",
        "Sisa kuota internet Anda adalah 2.5 GB. Berlaku sampai tanggal 20 bulan ini.",
        "Jangan lupa bayar patungan kado buat Budi besok ya, 50rb aja.",
        "Halo, ini dengan kurir Tokopedia. Paket sudah saya titipkan di pos satpam ya.",
        "Baik, nanti saya kabari lagi kalau sudah sampai di lokasi.",
        "Terima kasih atas pembayarannya. Berikut adalah bukti struk digital Anda.",
        "Besok pakai baju seragam warna biru ya, jangan lupa bawa laptop.",
        "Halo, besok libur nasional jadi kantor tutup. Selamat beristirahat.",
        "Tolong cek email ya, saya sudah kirim revisi dokumennya.",
    ]
    return pd.concat([pd.DataFrame({"text": spam_data, "label": 1}),
                      pd.DataFrame({"text": ham_data, "label": 0})], ignore_index=True)


def load_indonesian_dataset(path: str = DATASET_ID_CSV) -> pd.DataFrame:
    """Dataset SMS Indonesia: label 1 (penipuan) -> 1; label 0 (normal) & 2 (promo) -> 0."""
    df = pd.read_csv(path)
    df = df.rename(columns={"Teks": "text"})
    df = df.dropna(subset=["text"]).drop_duplicates(subset="text").reset_index(drop=True)
    df["kategori_asli"] = df["label"].map({0: "normal", 1: "penipuan", 2: "promo"})
    df["label"] = (df["label"] == 1).astype(int)
    return df[["text", "label", "kategori_asli"]]


def load_english_dataset(path: str = DATASET_EN_CSV) -> pd.DataFrame:
    """Dataset Mendeley (opsional): smishing -> 1; ham & spam -> 0 (konsisten: spam promosi bukan penipuan)."""
    df = pd.read_csv(path)
    df["LABEL"] = df["LABEL"].str.lower().str.strip()
    df = df[df["LABEL"].isin(["ham", "spam", "smishing"])].copy()
    df["label"] = (df["LABEL"] == "smishing").astype(int)
    df = df.rename(columns={"TEXT": "text"})[["text", "label"]].dropna(subset=["text"])
    return df


def split_dataset(df: pd.DataFrame):
    """Pembagian hold-out 80:20 berstrata — dipakai bersama oleh pelatihan dan evaluasi."""
    return train_test_split(df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE)


def load_data_2026(include_dev: bool = True) -> pd.DataFrame:
    """SMS 2026 hasil pengumpulan peneliti. Kolom: text, label, jenis, sumber, file.
    SMS berjenis 'ragu' (label belum pasti) tidak dipakai.
    include_dev=False: tanpa 5 SMS judol pengembangan (dipakai saat evaluasi)."""
    files = ["sms_penipuan_2026.csv", "sms_sah_2026.csv", "sms_sah_teman.csv"]
    if include_dev:
        files.append("sms_judol_pengembangan.csv")
    parts = []
    for f in files:
        p = os.path.join(DATA_2026_DIR, f)
        if os.path.exists(p):
            d = pd.read_csv(p).rename(columns={"teks": "text"})
            if "jenis" not in d:
                d["jenis"] = "judol"
            d["file"] = f
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["text", "label", "jenis", "sumber", "file"])
    d = pd.concat(parts, ignore_index=True)
    d = d[d["jenis"] != "ragu"].dropna(subset=["text", "label"]).copy()
    d["label"] = d["label"].astype(int)
    return d.reset_index(drop=True)


def build_training_set(df_train: pd.DataFrame, data_2026: pd.DataFrame = None) -> pd.DataFrame:
    parts = [df_train[["text", "label"]]] + [get_indonesian_examples()] * MANUAL_OVERSAMPLE
    if data_2026 is not None and len(data_2026):
        parts += [data_2026[["text", "label"]]] * DATA_2026_OVERSAMPLE
    if USE_ENGLISH_DATASET and os.path.exists(DATASET_EN_CSV):
        parts.append(load_english_dataset())
    return pd.concat(parts, ignore_index=True)


# ── Model ───────────────────────────────────────────────────────────────────
def fit_model(texts, labels, normalize: bool = None, feature: str = None, alpha: float = None):
    """normalize=True: teks dinormalisasi dulu (text_normalizer) sebelum dihitung TF-IDF.
    Normalisasi dipasang sebagai 'preprocessor' sehingga ikut tersimpan di vectorizer.pkl
    dan otomatis dipakai saat prediksi. feature: "kata" atau "huruf" (lihat FEATURE)."""
    normalize = USE_NORMALIZER if normalize is None else normalize
    feature = feature or FEATURE
    alpha = NB_ALPHA[feature] if alpha is None else alpha
    pre = {"preprocessor": normalize_for_model} if normalize else {"lowercase": True}
    if feature == "huruf":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM, min_df=2,
                                     sublinear_tf=True, **pre)
    else:
        vectorizer = TfidfVectorizer(stop_words=None, max_features=5000, **pre)
    X = vectorizer.fit_transform(pd.Series(texts).astype(str))
    model = MultinomialNB(alpha=alpha)
    model.fit(X, labels)
    return model, vectorizer


def train_and_save_model():
    df = load_indonesian_dataset()
    df_train, df_test = split_dataset(df)
    d26 = load_data_2026() if USE_DATA_2026 else None
    train = build_training_set(df_train, d26)

    print(f"Dataset Indonesia: {len(df)} SMS unik -> latih {len(df_train)}, uji {len(df_test)}")
    if d26 is not None:
        print(f"Data 2026: {len(d26)} SMS ({int(d26.label.sum())} penipuan) x{DATA_2026_OVERSAMPLE}")
    print(f"Data latih akhir (+30 contoh peneliti x{MANUAL_OVERSAMPLE}{' + dataset EN' if USE_ENGLISH_DATASET else ''}): "
          f"{len(train)} SMS ({int(train.label.sum())} penipuan, {int((train.label == 0).sum())} bukan penipuan)"
          f" | fitur: {FEATURE}")

    model, vectorizer = fit_model(train["text"], train["label"])
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    _cache["model"], _cache["vectorizer"] = model, vectorizer
    print(f"Model disimpan: {MODEL_PATH}")
    return model, vectorizer


def _load_model():
    if _cache["model"] is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
            try:
                _cache["model"] = joblib.load(MODEL_PATH)
                _cache["vectorizer"] = joblib.load(VEC_PATH)
            except Exception as e:
                print(f"Gagal memuat model lokal ({e}), melatih ulang...")
                train_and_save_model()
        else:
            train_and_save_model()
    return _cache["model"], _cache["vectorizer"]


def predict_text_phishing(text: str) -> float:
    """Probabilitas (0-1) bahwa teks bergaya bahasa penipuan."""
    if not text or not str(text).strip():
        return 0.0
    model, vectorizer = _load_model()
    X_input = vectorizer.transform([str(text)])
    return float(model.predict_proba(X_input)[0][1])


if __name__ == "__main__":
    train_and_save_model()
