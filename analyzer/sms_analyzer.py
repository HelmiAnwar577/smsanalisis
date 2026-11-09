"""
sms_analyzer.py v2.1
====================
Perubahan v2.1:
- Kata kunci dicari pada teks yang sudah dinormalisasi (text_normalizer.py).
- Kata kunci khas judi online (JUDOL_ROOTS).
- Keluaran baru: obfuscation_count, obfuscation_examples, normalized_text.

Ekstraksi entitas penting dari teks SMS untuk analisis indikator.

Perubahan dari v1:
- Flag (?i) pada OTP_REGEX dipindah ke awal pola. Versi lama menulis r'\b(?i)...'
  yang membuat program error saat di-import di Python 3.11 ke atas.
- OTP hanya dikenali bila pesan memuat kata kunci OTP (otp, kode, pin, sandi,
  verifikasi, dll.). Versi lama menganggap SEMUA angka 4-6 digit sebagai OTP
  (tahun "2013", harga "4300", dst.), sehingga banyak SMS penipuan tanpa tautan
  keliru dianggap "transaksional" dan skor AI-nya dipotong.
- Kata kunci dikenali juga dalam bentuk berimbuhan (diblokir, ditutup, memenangkan).
"""

import re

from analyzer.text_normalizer import normalize_text

# ── Pola regex ──────────────────────────────────────────────────────────────
PHONE_REGEX = re.compile(r'(\+62|62|0)[1-9][0-9]{6,11}')
SHORTCODE_REGEX = re.compile(r'\b\d{4,5}\b')

# Kata yang menandakan pesan berisi kode OTP/verifikasi
OTP_KEYWORD_REGEX = re.compile(
    r'(?i)\b(otp|kode|code|pin|sandi|password|verifikasi|verification|token)\b'
)
# Kandidat kode: angka 4-8 digit yang berdiri sendiri
OTP_CANDIDATE_REGEX = re.compile(r'(?<![\d.,])\b([0-9]{4,8})\b(?![.,]?\d)')

MONEY_REGEX = re.compile(r'(?i)(rp|idr)\s?\.?\s?[0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{2})?')

# Daftar kata dasar SAMA dengan v1. Yang berubah hanya cara mencocokkan:
# v2 juga menangkap bentuk berimbuhan (diblokir, ditutup, memenangkan, hadiahnya, ...)
# yang di v1 lolos karena pola \b...\b hanya cocok dengan kata dasar persis.
URGENCY_ROOTS = ("selamat|menang|hadiah|undian|jutaan|transfer|segera|darurat|tolong|pulsa|"
                 "blokir|tutup|aktivasi|pinjaman|dana|kurir|paket|resi|tagihan|diskon|promo|"
                 "klik|bantuan|bansos")
# v2.1: kata khas judi online. Hanya kata yang praktis TIDAK muncul di SMS normal/promo
# pada data latih (dicek: "bonus", "daftar", "saldo", "member" sengaja TIDAK dimasukkan
# karena sering muncul di promo resmi operator dan notifikasi bank).
JUDOL_ROOTS = ("slot|gacor|maxwin|jackpot|deposit|depo|togel|cuan|hoki|withdraw|scatter|"
               "casino|kasino|poker|judi|taruhan")
URGENCY_KEYWORDS = [
    r'(?i)\b(?:di|ter|me|mem|men|meng|ke)?(' + URGENCY_ROOTS + r')(?:kan|an|nya|lah)?\b',
    r'(?i)\b(?:ber)?(' + JUDOL_ROOTS + r')(?:an|nya)?\b',
]


def _extract_otps(text: str, phones: list, amounts: list) -> list:
    """OTP = angka 4-8 digit, HANYA jika pesan memuat kata kunci OTP."""
    if not OTP_KEYWORD_REGEX.search(text):
        return []
    otps = []
    for m in OTP_CANDIDATE_REGEX.finditer(text):
        code = m.group(1)
        if any(code in p for p in phones):      # bagian dari nomor telepon
            continue
        if any(code in a.replace('.', '').replace(',', '') for a in amounts):  # bagian dari nominal
            continue
        otps.append(code)
    return list(set(otps))


def extract_sms_entities(text: str) -> dict:
    """
    Ekstrak informasi penting dari teks SMS untuk analisis Smishing.
    Kunci (key) keluaran sama dengan v1 agar app.py dan llm_evaluator.py tidak perlu diubah.
    """
    # 1. Nomor telepon & shortcode
    phones = [m.group(0) for m in PHONE_REGEX.finditer(text)]
    shortcodes = [m.group(0) for m in SHORTCODE_REGEX.finditer(text)]
    all_numbers = list(set(phones + shortcodes))

    # 2. Nominal uang
    amounts = list(set(m.group(0) for m in MONEY_REGEX.finditer(text)))

    # 3. OTP (wajib ada kata kunci OTP)
    otps = _extract_otps(text, phones, amounts)

    # 4. Kata kunci mendesak/penipuan — dicari pada teks yang SUDAH dinormalisasi,
    #    sehingga "Hdiah", "h-a-d-i-a-h", "h4diah" tetap dikenali sebagai "hadiah".
    norm = normalize_text(text)
    found_keywords = []
    for pat in URGENCY_KEYWORDS:
        for m in re.finditer(pat, norm["text"]):
            found_keywords.append(m.group(1).lower())   # simpan kata dasarnya
    found_keywords = list(set(found_keywords))

    return {
        "phone_numbers": all_numbers,
        "otps": otps,
        "amounts": amounts,
        "urgency_keywords": found_keywords,
        # Transaksional = ada OTP/nominal DAN tidak ada kata kunci penipuan
        "is_transactional": (len(otps) > 0 or len(amounts) > 0) and len(found_keywords) == 0,
        "has_scam_keywords": len(found_keywords) > 0,
        # v2.1: penyamaran ejaan (lihat text_normalizer.py)
        "obfuscation_count": norm["obfuscation_count"],
        "obfuscation_examples": (norm["events"]["huruf_berspasi"]
                                 + norm["events"]["tanda_baca_di_tengah"]
                                 + norm["events"]["angka_pengganti"]),
        "normalized_text": norm["text"],
    }
