"""
scoring_engine.py v4.0 (SMS Analysis Logic)
======================
Perbaikan:
- Menghapus logika header email (spoofing, dkim, spf, dmarc).
- Menambahkan skoring berdasarkan entitas SMS (OTP, Keyword Mendesak).
"""

from urllib.parse import urlparse

# ──────────────────────────────────────────────────────────────
# Konstanta
# ──────────────────────────────────────────────────────────────
DANGEROUS_EXTENSIONS = ['.apk', '.exe', '.scr', '.bat', '.cmd', '.vbs', '.ps1', '.msi', '.jar', '.dmg']
MACRO_EXTENSIONS = ['.xlsm', '.docm', '.xlsb', '.pptm', '.xltm', '.dotm', '.xlam', '.ppam']

WEIGHTS = {
    "sms_urgency":       {"score": 20, "label": "Kata kunci mendesak/penipuan",    "category": "S_A"},
    "sms_otp_safe":      {"score":-20, "label": "Pesan bersifat transaksional (OTP)","category": "S_A"},
    "sms_unknown_link":  {"score": 25, "label": "Tautan pada pesan tak dikenal",   "category": "S_A"},
    "punycode":          {"score": 25, "label": "Homograph / Punycode (kritis)",   "category": "S_B"},
    "cyrillic":          {"score": 25, "label": "Karakter Cyrillic (kritis)",      "category": "S_B"},
    "dangerous_ext":     {"score": 30, "label": "File payload berbahaya (.apk dsb)","category": "S_B"},
    "macro_ext":         {"score": 20, "label": "Dokumen bermakro",                "category": "S_B"},
    "shortlink":         {"score": 10, "label": "URL shortener terdeteksi",        "category": "S_B"},
    "very_new_domain":   {"score": 20, "label": "Domain sangat baru (< 30 hari)", "category": "S_C"},
    "new_domain":        {"score": 10, "label": "Domain baru (30–90 hari)",        "category": "S_C"},
    "ip_not_found":      {"score":  5, "label": "IP server tidak ditemukan",       "category": "S_C"},
    "virustotal_hit":    {"score": 15, "label": "Terdeteksi di VirusTotal",        "category": "S_C"},
    "cross_domain_redirect": {"score": 10, "label": "Redirect lintas domain",     "category": "S_D"},
    "url_unreachable":        {"score":  5, "label": "URL tidak dapat dijangkau",  "category": "S_D"},
}

CAPS = {"S_A": 35, "S_B": 40, "S_C": 25, "S_D": 15, "S_E": 35}

def _get_netloc(url: str) -> str:
    try:
        return urlparse(url).netloc or url.split("/")[0]
    except Exception:
        return ""

def _flag(key: str, detail: str = "", override_weight: int = None) -> dict:
    w = WEIGHTS[key]
    return {
        "key":      key,
        "category": w["category"],
        "label":    w["label"] + (f" — {detail}" if detail else ""),
        "weight":   override_weight if override_weight is not None else w["score"],
    }

def _score_sms_entities(sms_entities: dict, has_urls: bool) -> tuple[int, list]:
    flags = []
    s = 0
    
    # Deteksi Urgent/Scam Keywords
    if sms_entities.get("has_scam_keywords"):
        s += WEIGHTS["sms_urgency"]["score"]
        kws = ", ".join(sms_entities["urgency_keywords"])
        flags.append(_flag("sms_urgency", f"Terdeteksi: {kws}"))
        
    # Kombinasi mematikan: ada link + ada keyword penipuan (APK scam dll)
    if has_urls and sms_entities.get("has_scam_keywords"):
        s += WEIGHTS["sms_unknown_link"]["score"]
        flags.append(_flag("sms_unknown_link", "Berisi tautan + bahasa manipulatif"))

    # Jika ini SMS OTP tanpa link aneh, kurangi risiko
    if sms_entities.get("is_transactional") and not has_urls:
        s += WEIGHTS["sms_otp_safe"]["score"]
        flags.append(_flag("sms_otp_safe", "Terdapat OTP/PIN/Nominal tanpa tautan"))

    # Batasi agar skor tidak negatif atau terlalu tinggi
    final_score = max(0, min(s, CAPS["S_A"]))
    return final_score, flags

def _score_urls(url_list: list) -> tuple[int, list]:
    flags, s, seen = [], 0, set()

    for item in url_list:
        url    = item.get("url", "").lower()
        domain = item.get("domain", "").lower()

        if item.get("punycode") and "punycode" not in seen:
            seen.add("punycode")
            s += WEIGHTS["punycode"]["score"]
            flags.append(_flag("punycode", item.get("punycode_decoded", domain)))

        if item.get("cyrillic") and "cyrillic" not in seen:
            seen.add("cyrillic")
            s += WEIGHTS["cyrillic"]["score"]
            flags.append(_flag("cyrillic", item.get("cyrillic_chars", "")))

        if "dangerous_ext" not in seen:
            ext = next((e for e in DANGEROUS_EXTENSIONS if url.endswith(e) or f"{e}?" in url), None)
            if ext:
                seen.add("dangerous_ext")
                s += WEIGHTS["dangerous_ext"]["score"]
                flags.append(_flag("dangerous_ext", ext))

        if "macro_ext" not in seen:
            mac = next((e for e in MACRO_EXTENSIONS if url.endswith(e) or f"{e}?" in url), None)
            if mac:
                seen.add("macro_ext")
                s += WEIGHTS["macro_ext"]["score"]
                flags.append(_flag("macro_ext", mac))

        if item.get("shortlink") and "shortlink" not in seen:
            seen.add("shortlink")
            s += WEIGHTS["shortlink"]["score"]
            flags.append(_flag("shortlink", item.get("shortener_used", "")))

    return min(s, CAPS["S_B"]), flags

def _score_domain(domain_info: dict) -> tuple[int, list]:
    if not domain_info:
        return 0, []

    flags, s = [], 0
    age = domain_info.get("domain_age")
    if age not in ("-", None, ""):
        try:
            age_int = int(age)
            if age_int < 30:
                s += WEIGHTS["very_new_domain"]["score"]
                flags.append(_flag("very_new_domain", f"{age_int} hari"))
            elif age_int < 90:
                s += WEIGHTS["new_domain"]["score"]
                flags.append(_flag("new_domain", f"{age_int} hari"))
        except (ValueError, TypeError):
            pass

    if domain_info.get("ip_address") in ("Tidak Ditemukan", None, "", "N/A"):
        s += WEIGHTS["ip_not_found"]["score"]
        flags.append(_flag("ip_not_found"))

    vt       = domain_info.get("virustotal_positives", 0) or 0
    vt_total = domain_info.get("virustotal_total", 72) or 72
    if vt > 0:
        vt_w = min(round(vt / vt_total * WEIGHTS["virustotal_hit"]["score"]), 15)
        s += vt_w
        flags.append({**_flag("virustotal_hit", f"{vt}/{vt_total} engine"), "weight": vt_w})

    return min(s, CAPS["S_C"]), flags

def _score_redirect(chain: list) -> tuple[int, list]:
    if not chain:
        return 0, []
    flags, s = [], 0
    fd = _get_netloc(chain[0].get("url", ""))
    ld = _get_netloc(chain[-1].get("url", ""))
    if fd and ld and fd != ld:
        s += WEIGHTS["cross_domain_redirect"]["score"]
        flags.append(_flag("cross_domain_redirect", f"{fd} → {ld}"))

    seen_unreachable = False
    for r in chain:
        status = str(r.get("status", "")).lower()
        if ("error" in status or "unreachable" in status or "timeout" in status) and not seen_unreachable:
            seen_unreachable = True
            s += WEIGHTS["url_unreachable"]["score"]
            flags.append(_flag("url_unreachable"))

    return min(s, CAPS["S_D"]), flags

def _score_content(phishing_prob: float, no_urls: bool, is_transactional: bool) -> tuple[int, list]:
    flags = []
    MAX_SE = CAPS["S_E"]
    
    if phishing_prob < 0.50:
        return 0, []
        
    score = round(phishing_prob * MAX_SE)
    score = min(score, MAX_SE)

    if is_transactional and no_urls:
        score = round(score * 0.2)
        if score == 0: return 0, []
        flags.append({
            "key":      "naive_bayes_hit",
            "category": "S_E",
            "label":    f"Gaya penipuan (Diabaikan: Transaksional/OTP - NB: {phishing_prob * 100:.1f}%)",
            "weight":   score,
        })
    elif no_urls:
        score = round(score * 0.4)
        if score == 0: return 0, []
        flags.append({
            "key":      "naive_bayes_hit",
            "category": "S_E",
            "label":    f"Pola AI mencurigakan TANPA TAUTAN (Risiko rendah - NB: {phishing_prob * 100:.1f}%)",
            "weight":   score,
        })
    else:
        flags.append({
            "key":      "naive_bayes_hit",
            "category": "S_E",
            "label":    f"Gaya bahasa penipuan terdeteksi (NB: {phishing_prob * 100:.1f}%)",
            "weight":   score,
        })
        
    return score, flags

def calculate_score(url_analysis: list, domain_info: dict, redirect_chain: list, sms_entities: dict, phishing_probability: float = 0.0) -> dict:
    
    has_urls = len(url_analysis) > 0
    no_urls = len(url_analysis) == 0

    s_a, f_a = _score_sms_entities(sms_entities, has_urls)
    s_b, f_b = _score_urls(url_analysis)
    s_c, f_c = _score_domain(domain_info)
    s_d, f_d = _score_redirect(redirect_chain)
    
    s_e, f_e = _score_content(phishing_probability, no_urls, sms_entities.get("is_transactional", False))

    total = min(max(s_a + s_b + s_c + s_d + s_e, 0), 100)

    # Sistem Status
    if total <= 30:
        status, warna = "AMAN", "green"
    elif total <= 65:
        status, warna = "WASPADA", "orange"
    else:
        status, warna = "BERBAHAYA", "red"

    breakdown = {
        "S_A": {"label": "Analisis Konten (Heuristik)", "score": s_a, "max": CAPS["S_A"], "flags": f_a},
        "S_B": {"label": "Analisis Tautan (URL)",       "score": s_b, "max": CAPS["S_B"], "flags": f_b},
        "S_C": {"label": "Reputasi Domain Target",      "score": s_c, "max": CAPS["S_C"], "flags": f_c},
        "S_D": {"label": "Rantai Redirect",             "score": s_d, "max": CAPS["S_D"], "flags": f_d},
        "S_E": {"label": "Analisis AI (Teks NB)",       "score": s_e, "max": CAPS["S_E"], "flags": f_e},
    }

    return {
        "total_score": total,
        "status":      status,
        "warna":       warna,
        "breakdown":   breakdown,
        "all_flags":   f_a + f_b + f_c + f_d + f_e,
    }