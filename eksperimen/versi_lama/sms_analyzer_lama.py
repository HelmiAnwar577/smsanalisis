import re

# Regex Patterns for SMS Analysis
PHONE_REGEX = re.compile(r'(\+62|62|0)[1-9][0-9]{6,11}')
SHORTCODE_REGEX = re.compile(r'\b\d{4,5}\b')
# Catatan: flag (?i) dipindah ke awal pola agar bisa di-import di Python 3.11+.
# Perilakunya sama persis dengan versi asli di Python <= 3.10 (flag berlaku global).
OTP_REGEX = re.compile(r'(?i)\b(otp|pin|kode|code|sandi)[\s:]*([0-9]{4,8})\b')
OTP_STANDALONE_REGEX = re.compile(r'\b[0-9]{4,6}\b') # Just digits in case keyword missing
MONEY_REGEX = re.compile(r'(?i)(rp|idr)\s?\.?\s?[0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{2})?')
URGENCY_KEYWORDS = [
    r'(?i)\b(selamat|menang|hadiah|undian|jutaan|transfer|segera|darurat|tolong|pulsa|blokir|tutup|aktivasi|pinjaman|dana|kurir|paket|resi|tagihan|diskon|promo|klik|bantuan|bansos)\b'
]

def extract_sms_entities(text: str) -> dict:
    """
    Ekstrak informasi penting dari teks SMS untuk analisis heuristik.
    """
    # 1. Phone numbers & Shortcodes
    phones = [m.group(0) for m in PHONE_REGEX.finditer(text)]
    shortcodes = [m.group(0) for m in SHORTCODE_REGEX.finditer(text)]
    
    # Gabungkan semua kemungkinan pengirim/nomor yang disebut
    all_numbers = list(set(phones + shortcodes))
    
    # 2. OTPs
    otps = []
    for m in OTP_REGEX.finditer(text):
        otps.append(m.group(2))
    
    # Fallback to standalone 4-6 digits if specific keywords weren't found but looks like OTP
    if not otps:
        # Avoid grabbing parts of phone numbers, we check words
        words = text.split()
        for w in words:
            # Clean word
            cw = re.sub(r'\W+', '', w)
            if cw.isdigit() and 4 <= len(cw) <= 6:
                if cw not in otps and not any(cw in p for p in phones):
                    otps.append(cw)
    
    otps = list(set(otps))
    
    # 3. Money / Amounts
    amounts = [m.group(0) for m in MONEY_REGEX.finditer(text)]
    
    # 4. Urgency/Scam Keywords
    found_keywords = []
    for pat in URGENCY_KEYWORDS:
        matches = re.finditer(pat, text)
        for m in matches:
            found_keywords.append(m.group(0).lower())
            
    found_keywords = list(set(found_keywords))
    
    return {
        "phone_numbers": all_numbers,
        "otps": otps,
        "amounts": list(set(amounts)),
        "urgency_keywords": found_keywords,
        "is_transactional": (len(otps) > 0 or len(amounts) > 0) and len(found_keywords) == 0,
        "has_scam_keywords": len(found_keywords) > 0
    }
