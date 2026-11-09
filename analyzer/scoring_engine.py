"""
scoring_engine.py v6.1 (SMS Analysis Logic)
===========================================
Perubahan v6.1: Naive Bayes tidak dihitung untuk SMS kurang dari 3 kata
(mis. satu kata "menang" sebelumnya langsung berstatus Waspada).

Perubahan v6.0 (aturan fusi hibrida):
- Indikator tautan KRITIS (peniruan merek termasuk nama mirip, punycode/Cyrillic,
  berkas .apk/.exe, dokumen bermakro) sendirian sudah cukup untuk Waspada (S_B >= 31).
  Simetris dengan aturan NB: satu bukti kuat -> Waspada; Berbahaya butuh bukti kedua.
- Pemendek tautan (bit.ly, dll.) TIDAK lagi dihitung sebagai "bukti kedua" untuk bonus
  tautan + kata kunci, karena promo resmi operator juga memakainya
  (penyebab utama salah tuduh pada SMS sah 2026).

Perubahan v5.1:
- Indikator baru S_A "Ejaan sengaja disamarkan" (+20): kata dipecah tanda baca
  ("daf-tar", "ju'ta"), huruf berspasi ("R A J A"), angka pengganti ("h0ki").
  Penyamaran juga dihitung sebagai "bukti kedua" untuk bonus tautan + kata kunci.
  Bobot dan ambangnya dipilih dengan validasi silang di DATA LATIH (evaluasi.py).

Menggabungkan semua sinyal menjadi skor risiko 0-100 dan status
AMAN (0-30) / WASPADA (31-65) / BERBAHAYA (>65).

Lima kategori skor (batas maksimum):
  S_A Entitas SMS (35) | S_B Tautan (40) | S_C Reputasi domain (25)
  S_D Rantai redirect (15) | S_E Naive Bayes (35)

Perubahan dari v4:
- Pengurangan -20 untuk pesan transaksional DIHAPUS dari S_A. Di v4 aturan itu
  tidak pernah berpengaruh: pesan transaksional pasti tanpa kata kunci penipuan,
  jadi S_A-nya 0, dan -20 langsung dipotong lagi menjadi 0 oleh max(0, ...).
  Penurunan risiko pesan transaksional tetap berjalan lewat pengali NB (x0,2).
- Indikator baru S_B "peniruan merek pada tautan" (+25, setara homograph karena
  tujuannya sama: membuat korban mengira tautan itu milik merek resmi). Di v4
  hasil deteksinya hanya dikirim ke LLM, tidak ikut dihitung dalam skor.
- S_E (Naive Bayes): jika NB >= 0,5 maka S_E = maks(31, 35 x p). Artinya model
  yang yakin pesan bergaya penipuan SENDIRIAN sudah cukup untuk status Waspada;
  status Berbahaya tetap butuh bukti tambahan dari Risk Score.
  Pengali 0,4 untuk pesan tanpa tautan DIHAPUS. Dengan pengali itu SMS tanpa
  tautan maksimal hanya 20 + 14 = 34 poin, sehingga penipuan tanpa tautan
  (minta transfer, mama minta pulsa) hampir selalu lolos.
- Bonus "tautan + kata kunci" (+25) kini butuh bukti kedua: NB >= 0,5 ATAU
  tautannya sendiri berisiko (S_B > 0). Tanpa aturan ini, hampir semua SMS promo
  resmi operator (ada tautan + kata "promo/paket/klik") otomatis Waspada.
- Ketiga aturan di atas dipilih dengan validasi silang 5-fold pada DATA LATIH saja
  (lihat eksperimen/evaluasi.py), bukan dengan melihat data uji.
"""

from urllib.parse import urlparse

# ──────────────────────────────────────────────────────────────
# Konstanta
# ──────────────────────────────────────────────────────────────
DANGEROUS_EXTENSIONS = ['.apk', '.exe', '.scr', '.bat', '.cmd', '.vbs', '.ps1', '.msi', '.jar', '.dmg']
MACRO_EXTENSIONS = ['.xlsm', '.docm', '.xlsb', '.pptm', '.xltm', '.dotm', '.xlam', '.ppam']

WEIGHTS = {
    "sms_urgency":       {"score": 20, "label": "Kata kunci mendesak/penipuan",     "category": "S_A"},
    "sms_unknown_link":  {"score": 25, "label": "Tautan pada pesan tak dikenal",    "category": "S_A"},
    "obfuscation":       {"score": 20, "label": "Ejaan sengaja disamarkan",         "category": "S_A"},
    "sms_transactional": {"score":  0, "label": "Pesan transaksional (OTP/nominal) tanpa tautan — bobot AI dikecilkan", "category": "S_A"},
    "punycode":          {"score": 25, "label": "Homograph / Punycode (kritis)",    "category": "S_B"},
    "cyrillic":          {"score": 25, "label": "Karakter Cyrillic (kritis)",       "category": "S_B"},
    "dangerous_ext":     {"score": 30, "label": "File payload berbahaya (.apk dsb)", "category": "S_B"},
    "macro_ext":         {"score": 20, "label": "Dokumen bermakro",                 "category": "S_B"},
    "brand_impersonation": {"score": 25, "label": "Peniruan nama merek pada tautan", "category": "S_B"},
    "shortlink":         {"score": 10, "label": "URL shortener terdeteksi",         "category": "S_B"},
    "very_new_domain":   {"score": 20, "label": "Domain sangat baru (< 30 hari)",   "category": "S_C"},
    "new_domain":        {"score": 10, "label": "Domain baru (30–90 hari)",         "category": "S_C"},
    "ip_not_found":      {"score":  5, "label": "IP server tidak ditemukan",        "category": "S_C"},
    "virustotal_hit":    {"score": 15, "label": "Terdeteksi di VirusTotal",         "category": "S_C"},
    "cross_domain_redirect": {"score": 10, "label": "Redirect lintas domain",      "category": "S_D"},
    "url_unreachable":   {"score":  5, "label": "URL tidak dapat dijangkau",        "category": "S_D"},
}

CAPS = {"S_A": 35, "S_B": 40, "S_C": 25, "S_D": 15, "S_E": 35}

# Parameter fusi (dipilih dengan validasi silang pada data latih — lihat evaluasi.py)
SCORING_PARAMS = {
    "nb_threshold":            0.50,  # di bawah ini NB tidak menyumbang skor
    "nb_min_score":            31,    # NB >= ambang -> minimal 31 poin (= Waspada). None = perilaku v4
    "nb_factor_transactional": 0.20,  # pesan transaksional (OTP/nominal) tanpa tautan
    "nb_factor_no_url":        1.00,  # pesan lain tanpa tautan (v4: 0,40)
    "link_bonus_needs_evidence": True, # bonus tautan+kata kunci butuh NB>=ambang atau S_B>0 (atau penyamaran)
    # v5.1 — penyamaran ejaan (text_normalizer.py)
    "obfuscation_weight":      20,    # 0 = indikator dimatikan (perilaku v5.0)
    "obfuscation_min":         2,     # minimal 2 penyamaran (1 kali bisa kebetulan, mis. "top-up")
    # v6.0 — aturan fusi direvisi setelah evaluasi SMS 2026
    "critical_link_min_score": 31,    # indikator tautan kritis -> S_B minimal 31 (= Waspada). None = v5
    "evidence_needs_critical_link": True,  # pemendek tautan TIDAK lagi dihitung bukti kedua. False = v5
    # v6.1 — teks yang terlalu pendek tidak dinilai gaya bahasanya oleh Naive Bayes
    "nb_min_words":            3,     # < 3 kata -> NB tidak dihitung (mis. satu kata "menang"). 0 = v6.0
}

# Indikator tautan yang hampir tidak pernah muncul di SMS sah
CRITICAL_LINK_FLAGS = {"punycode", "cyrillic", "dangerous_ext", "macro_ext", "brand_impersonation"}
# Parameter yang mengembalikan perilaku fusi sebelum v6 (untuk uji ablasi)
PARAMS_V5 = {"critical_link_min_score": None, "evidence_needs_critical_link": False}

STATUS_THRESHOLDS = {"aman_max": 30, "waspada_max": 65}


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


def _is_obfuscated(sms_entities: dict, params: dict) -> bool:
    return (params.get("obfuscation_weight", 0) > 0
            and sms_entities.get("obfuscation_count", 0) >= params.get("obfuscation_min", 1))


def _score_sms_entities(sms_entities: dict, has_urls: bool, link_evidence: bool = True,
                        params: dict = None) -> tuple:
    flags, s = [], 0
    params = params or SCORING_PARAMS

    if _is_obfuscated(sms_entities, params):
        w = params["obfuscation_weight"]
        s += w
        contoh = ", ".join(sms_entities.get("obfuscation_examples", [])[:4])
        flags.append(_flag("obfuscation", f"Contoh: {contoh}", override_weight=w))

    if sms_entities.get("has_scam_keywords"):
        s += WEIGHTS["sms_urgency"]["score"]
        kws = ", ".join(sorted(sms_entities["urgency_keywords"]))
        flags.append(_flag("sms_urgency", f"Terdeteksi: {kws}"))

    # Kombinasi berbahaya: ada tautan + bahasa manipulatif (+ bukti kedua, lihat SCORING_PARAMS)
    if has_urls and sms_entities.get("has_scam_keywords") and link_evidence:
        s += WEIGHTS["sms_unknown_link"]["score"]
        flags.append(_flag("sms_unknown_link", "Berisi tautan + bahasa manipulatif"))

    # Penanda informatif saja (bobot 0); efeknya ada di pengali NB pada S_E
    if sms_entities.get("is_transactional") and not has_urls:
        flags.append(_flag("sms_transactional", "Terdapat OTP/PIN/nominal tanpa tautan"))

    return max(0, min(s, CAPS["S_A"])), flags


def _score_urls(url_list: list) -> tuple:
    flags, s, seen = [], 0, set()

    for item in url_list:
        url = item.get("url", "").lower()
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

        if item.get("deceptive_subdomain") and "brand_impersonation" not in seen:
            seen.add("brand_impersonation")
            s += WEIGHTS["brand_impersonation"]["score"]
            where = item.get("impersonation_location", "-")
            flags.append(_flag("brand_impersonation", f"{item.get('impersonated_brand', '-')} ({where})"))

        if item.get("shortlink") and "shortlink" not in seen:
            seen.add("shortlink")
            s += WEIGHTS["shortlink"]["score"]
            flags.append(_flag("shortlink", item.get("shortener_used", "")))

    return min(s, CAPS["S_B"]), flags


def _score_domain(domain_info: dict) -> tuple:
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

    vt = domain_info.get("virustotal_positives", 0) or 0
    vt_total = domain_info.get("virustotal_total", 72) or 72
    if vt > 0:
        vt_w = min(round(vt / vt_total * WEIGHTS["virustotal_hit"]["score"]), 15)
        s += vt_w
        flags.append({**_flag("virustotal_hit", f"{vt}/{vt_total} engine"), "weight": vt_w})

    return min(s, CAPS["S_C"]), flags


def domain_risk_score(domain_info: dict) -> int:
    """Skor S_C untuk satu hasil reputasi domain. Dipakai app.py untuk memilih
    domain paling berisiko bila SMS memuat beberapa tautan."""
    return _score_domain(domain_info)[0]


def _score_redirect(chain: list) -> tuple:
    if not chain:
        return 0, []
    flags, s = [], 0
    fd = _get_netloc(chain[0].get("url", ""))
    ld = _get_netloc(chain[-1].get("url", ""))
    if fd and ld and fd != ld:
        s += WEIGHTS["cross_domain_redirect"]["score"]
        flags.append(_flag("cross_domain_redirect", f"{fd} → {ld}"))

    for r in chain:
        status = str(r.get("status", "")).lower()
        if "error" in status or "unreachable" in status or "timeout" in status:
            s += WEIGHTS["url_unreachable"]["score"]
            flags.append(_flag("url_unreachable"))
            break

    return min(s, CAPS["S_D"]), flags


def _score_content(phishing_prob: float, no_urls: bool, is_transactional: bool, params: dict) -> tuple:
    if phishing_prob < params["nb_threshold"]:
        return 0, []

    score = min(round(phishing_prob * CAPS["S_E"]), CAPS["S_E"])
    if params.get("nb_min_score") is not None:
        score = max(score, params["nb_min_score"])

    if is_transactional and no_urls:
        score = round(score * params["nb_factor_transactional"])
        label = f"Gaya penipuan (dikecilkan: pesan transaksional/OTP - NB: {phishing_prob * 100:.1f}%)"
    elif no_urls:
        score = round(score * params["nb_factor_no_url"])
        label = f"Gaya bahasa penipuan, tanpa tautan (NB: {phishing_prob * 100:.1f}%)"
    else:
        label = f"Gaya bahasa penipuan terdeteksi (NB: {phishing_prob * 100:.1f}%)"

    if score == 0:
        return 0, []
    return score, [{"key": "naive_bayes_hit", "category": "S_E", "label": label, "weight": score}]


def status_from_score(total: int) -> tuple:
    if total <= STATUS_THRESHOLDS["aman_max"]:
        return "AMAN", "green"
    if total <= STATUS_THRESHOLDS["waspada_max"]:
        return "WASPADA", "orange"
    return "BERBAHAYA", "red"


def calculate_score(url_analysis: list, domain_info: dict, redirect_chain: list,
                    sms_entities: dict, phishing_probability: float = 0.0,
                    params: dict = None) -> dict:
    """Signature sama dengan v4 (params opsional, dipakai saat eksperimen)."""
    p = {**SCORING_PARAMS, **(params or {})}

    has_urls = len(url_analysis) > 0
    no_urls = not has_urls

    # v6.1: SMS yang terlalu pendek (mis. satu kata) belum punya "gaya bahasa" untuk dinilai
    n_words = len((sms_entities.get("normalized_text") or "").split())
    nb_skipped = p.get("nb_min_words", 0) and n_words < p["nb_min_words"]
    if nb_skipped:
        phishing_probability = 0.0

    s_b, f_b = _score_urls(url_analysis)
    critical = any(f["key"] in CRITICAL_LINK_FLAGS for f in f_b)
    # v6: satu indikator tautan KRITIS sudah cukup untuk Waspada (simetris dengan aturan NB)
    if critical and p.get("critical_link_min_score") is not None:
        s_b = max(s_b, p["critical_link_min_score"])
    # Bukti kedua untuk bonus "tautan + kata kunci"
    link_signal = critical if p.get("evidence_needs_critical_link") else s_b > 0
    link_evidence = (not p["link_bonus_needs_evidence"]
                     or phishing_probability >= p["nb_threshold"] or link_signal
                     or _is_obfuscated(sms_entities, p))
    s_a, f_a = _score_sms_entities(sms_entities, has_urls, link_evidence, p)
    s_c, f_c = _score_domain(domain_info)
    s_d, f_d = _score_redirect(redirect_chain)
    s_e, f_e = _score_content(phishing_probability, no_urls, sms_entities.get("is_transactional", False), p)
    if nb_skipped:
        f_e = [{"key": "nb_skipped", "category": "S_E", "weight": 0,
                "label": f"Teks terlalu pendek ({n_words} kata), gaya bahasa tidak dinilai"}]

    total = min(max(s_a + s_b + s_c + s_d + s_e, 0), 100)
    status, warna = status_from_score(total)

    breakdown = {
        "S_A": {"label": "Analisis Konten ", "score": s_a, "max": CAPS["S_A"], "flags": f_a},
        "S_B": {"label": "Analisis Tautan (URL)",       "score": s_b, "max": CAPS["S_B"], "flags": f_b},
        "S_C": {"label": "Reputasi Domain Target",      "score": s_c, "max": CAPS["S_C"], "flags": f_c},
        "S_D": {"label": "Rantai Redirect",             "score": s_d, "max": CAPS["S_D"], "flags": f_d},
        "S_E": {"label": "Analisis Indikator (Teks NB)",       "score": s_e, "max": CAPS["S_E"], "flags": f_e},
    }

    return {
        "total_score": total,
        "status":      status,
        "warna":       warna,
        "breakdown":   breakdown,
        "all_flags":   f_a + f_b + f_c + f_d + f_e,
    }
