"""
text_normalizer.py v1.0
=======================
Mengembalikan ejaan SMS yang SENGAJA DISAMARKAN ke bentuk aslinya, sebelum
teks dibaca oleh kata kunci (sms_analyzer) dan Naive Bayes (text_classifier).

Kenapa perlu?
  SMS judi online sering memecah kata supaya lolos filter:
      "Daf-tar"  "sa.ldo"  "ju'ta ru'piah"  "R A J A K E R A"  "h0ki"  "Hdiah"
  Komputer membaca "Hdiah" sebagai kata yang berbeda dari "hadiah", sehingga
  kata kunci dan model tidak mengenalinya.

Empat aturan (semuanya dicek terhadap KAMUS kata, supaya teks normal tidak rusak):
  1. Huruf berspasi       "R A J A K E R A"  -> "rajakera"   (>= 4 huruf tunggal berurutan)
  2. Tanda baca di tengah "daf-tar" "sa.ldo" "ju'ta" -> "daftar" "saldo" "juta"
     Hanya digabung bila hasilnya ADA di kamus dan potongannya BUKAN dua kata
     yang sah (jadi "anak-anak", "wr.wb", "top-up" tidak diubah). Singkatan
     satu huruf ("s.d", "a.n") dan bahasa Inggris ("domino's", "i'm") dilewati.
  3. Angka pengganti huruf "h0ki" "s4ldo" -> "hoki" "saldo"   (hanya bila hasilnya ada di kamus)
  4. Huruf yang dihilangkan "hdiah" "sekrang" -> "hadiah" "sekarang"
     Kata yang TIDAK ada di kamus dicoba disisipi satu huruf; bila tepat satu
     kandidat ada di kamus, kata itu diganti.

Kamus = kata yang muncul minimal 2 kali di dataset SMS Indonesia + kata kunci.
Kamus tidak memakai label (penipuan/bukan), hanya daftar kata.

Tautan tidak ikut diubah (dilindungi dulu, lalu dikembalikan).

Keluaran normalize_text(): teks hasil normalisasi + jumlah "peristiwa penyamaran".
Aturan 1-3 dihitung sebagai penyamaran; aturan 4 TIDAK dihitung karena salah ketik
biasa ("knapa", "bsa") juga sering muncul di SMS normal.
"""

import os
import re
from collections import Counter
from functools import lru_cache

from analyzer.url_extractor import URL_REGEX

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ID_CSV = os.path.join(BASE_DIR, "data", "dataset_sms_spam_v1.csv")

# Kata dasar yang wajib ada di kamus (kata kunci sms_analyzer + kata umum judol)
EXTRA_WORDS = (
    "selamat menang hadiah undian jutaan juta rupiah transfer segera darurat tolong pulsa "
    "blokir tutup aktivasi pinjaman dana kurir paket resi tagihan diskon promo klik bantuan "
    "bansos slot gacor maxwin jackpot deposit depo togel cuan hoki withdraw scatter casino "
    "kasino poker judi taruhan saldo daftar sekarang member bonus gratis akun tarik"
).split()

_WORD = re.compile(r"[a-z]+")
_SPACED = re.compile(r"(?<![A-Za-z])((?:[A-Za-z] ){3,}[A-Za-z])(?![A-Za-z])")
_INNER_PUNCT = re.compile(r"\b([a-z]+)[.\-'’`*_]([a-z]+)\b")
_LEET = re.compile(r"\b[a-z]*[a-z][013457][a-z][a-z0-9]*\b")
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_EN_CONTRACTIONS = {"s", "m", "t", "ll", "re", "ve", "d"}


@lru_cache(maxsize=1)
def lexicon() -> frozenset:
    """Kamus kata: muncul >= 2 kali di dataset Indonesia, ditambah EXTRA_WORDS."""
    words = set(EXTRA_WORDS)
    if os.path.exists(DATASET_ID_CSV):
        import pandas as pd
        texts = pd.read_csv(DATASET_ID_CSV)["Teks"].dropna().astype(str)
        counts = Counter(w for t in texts for w in _WORD.findall(t.lower()))
        words |= {w for w, c in counts.items() if c >= 2 and len(w) >= 2}
    return frozenset(words)


def _protect_urls(text: str):
    urls = []

    def repl(m):
        urls.append(m.group(0))
        return f" §{len(urls) - 1}§ "

    return URL_REGEX.sub(repl, text), urls


def _restore_urls(text: str, urls: list) -> str:
    return re.sub(r" ?§(\d+)§ ?", lambda m: " " + urls[int(m.group(1))] + " ", text)


def _repair_missing_letter(word: str, lex: frozenset):
    """'hdiah' -> 'hadiah': coba sisipkan satu huruf; pakai hanya jika kandidatnya tunggal."""
    cands = {word[:i] + c + word[i:] for i in range(len(word) + 1) for c in _ALPHABET}
    hits = [w for w in cands if w in lex]
    return hits[0] if len(hits) == 1 else None


def normalize_text(text: str) -> dict:
    """Kembalikan {'text': teks_normal, 'events': {...}, 'obfuscation_count': n}."""
    lex = lexicon()
    events = {"huruf_berspasi": [], "tanda_baca_di_tengah": [], "angka_pengganti": [], "huruf_hilang": []}
    t, urls = _protect_urls(str(text or ""))

    # 1. Huruf berspasi
    def join_spaced(m):
        events["huruf_berspasi"].append(m.group(1))
        return m.group(1).replace(" ", "")

    t = _SPACED.sub(join_spaced, t).lower()

    # 2. Tanda baca di tengah kata
    def join_punct(m):
        a, b = m.group(1), m.group(2)
        sep = m.group(0)[len(a)]
        if len(a) == 1 and len(b) == 1:                    # singkatan: s.d, a.n
            return m.group(0)
        if sep in "'’" and b in _EN_CONTRACTIONS:          # bahasa Inggris: domino's, i'm
            return m.group(0)
        joined = a + b
        if joined in lex and not (a in lex and b in lex and len(a) > 2 and len(b) > 2):
            events["tanda_baca_di_tengah"].append(m.group(0))
            return joined
        return m.group(0)

    t = _INNER_PUNCT.sub(join_punct, t)

    # 3. Angka pengganti huruf
    def fix_leet(m):
        w = m.group(0)
        cand = w.translate(_LEET_MAP)
        if cand.isalpha() and cand in lex:
            events["angka_pengganti"].append(w)
            return cand
        return w

    t = _LEET.sub(fix_leet, t)

    # 4. Huruf yang hilang (tidak dihitung sebagai penyamaran)
    def fix_missing(m):
        w = m.group(0)
        if len(w) < 5 or w in lex:
            return w
        fixed = _repair_missing_letter(w, lex)
        if fixed:
            events["huruf_hilang"].append(f"{w}->{fixed}")
            return fixed
        return w

    t = _WORD.sub(fix_missing, t)

    t = re.sub(r"\s+", " ", _restore_urls(t, urls)).strip()
    n = sum(len(events[k]) for k in ("huruf_berspasi", "tanda_baca_di_tengah", "angka_pengganti"))
    return {"text": t, "events": events, "obfuscation_count": n}


def normalize_for_model(text: str) -> str:
    """Dipakai sebagai 'preprocessor' TF-IDF, sehingga model latih & prediksi selalu sama."""
    return normalize_text(text)["text"]


if __name__ == "__main__":
    contoh = [
        "#sipalingho-ki Daf-tar sekrang! Hdiah sa.ldo 100rb Car! kam! di * R A J A K E R A *, a.pint4r.com/88462",
        "2626 Cuan terus! Anda telah terpilih untuk mendapatkan hadiah 1 ju'ta ru'piah.",
        "Assalamu'alaikum wr.wb. Anak-anak besok libur, top-up pulsa dulu ya",
    ]
    for c in contoh:
        r = normalize_text(c)
        print(c, "\n  ->", r["text"], "\n  penyamaran:", r["obfuscation_count"], r["events"], "\n")
