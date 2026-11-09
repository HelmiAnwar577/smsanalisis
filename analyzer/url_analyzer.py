"""
url_analyzer.py v2.2
====================
v2.2: peniruan merek juga dikenali bila nama domain BEDA SATU HURUF dari merek
      (typosquatting): "posindonesla.eu.cc" -> posindonesia, "whctsapp.com" -> whatsapp.
v2.1: pemendek tautan judi online (did.li, dll.).

Analisis struktur setiap tautan: homograph (punycode/Cyrillic), ekstensi berkas
berbahaya, pemendek tautan, encoding, dan peniruan merek.

Perubahan dari v1:
- Pemendek tautan dicocokkan per domain (persis atau subdomain-nya). Versi lama
  memakai pencocokan substring, sehingga "t.co" ikut cocok dengan "telkomsat.co.id"
  dan "s.id" cocok dengan "bpjs.id".
- Peniruan merek kini dicek di tiga tempat: subdomain (bca.login-aman.com),
  nama domain itu sendiri (sicepat-lacak.xyz), dan path tautan pendek
  (bit.ly/mandiri-verif). Domain resmi milik merek tidak dianggap meniru.
- Daftar merek ditambah merek lokal (kurir, e-wallet, operator, instansi).
- Kunci keluaran v1 tetap ada; kunci baru: impersonation_location.
"""

import re
import urllib.parse

import idna

SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "s.id", "cutt.ly", "rb.gy",
              "ow.ly", "short.io", "tiny.cc", "shorturl.at", "is.gd", "v.gd", "t.ly",
              "bitly.ws", "linktr.ee",
              # v2.1: pemendek tautan yang dipakai SMS judi online
              "did.li", "rebrand.ly", "heylink.me", "cutt.us", "tr.ee", "lnk.bio",
              "bit.do", "shorte.st", "urlz.fr", "clck.ru"]
DANGEROUS_EXTENSIONS = ['.apk', '.exe', '.scr', '.bat', '.cmd', '.vbs', '.ps1', '.msi', '.jar', '.dmg']
MACRO_EXTENSIONS = ['.xlsm', '.docm', '.xlsb', '.pptm', '.xltm', '.dotm', '.xlam', '.ppam']

# Merek yang sering ditiru, beserta domain resminya (domain resmi tidak dianggap meniru)
BRANDS = {
    # bank
    "bca": ["bca.co.id", "klikbca.com"],
    "bri": ["bri.co.id"],
    "bni": ["bni.co.id"],
    "mandiri": ["bankmandiri.co.id", "mandiri.co.id"],
    "btn": ["btn.co.id"],
    "danamon": ["danamon.co.id"],
    "cimb": ["cimbniaga.co.id"],
    # e-commerce & ride-hailing
    "shopee": ["shopee.co.id", "shopee.com", "shopeepay.co.id"],
    "tokopedia": ["tokopedia.com"],
    "lazada": ["lazada.co.id"],
    "gojek": ["gojek.com"],
    "grab": ["grab.com", "grab.co"],
    # e-wallet
    "dana": ["dana.id"],
    "ovo": ["ovo.id"],
    "gopay": ["gopay.co.id"],
    "linkaja": ["linkaja.id"],
    # kurir
    "sicepat": ["sicepat.com"],
    "jne": ["jne.co.id"],
    "jnt": ["jet.co.id"],
    "anteraja": ["anteraja.id"],
    "posindonesia": ["posindonesia.co.id"],
    # operator seluler
    "telkomsel": ["telkomsel.com", "tsel.me"],
    "indosat": ["indosatooredoo.com", "indosat.com", "ioh.co.id", "im3.id"],
    "xl": ["xl.co.id"],
    # instansi
    "bpjs": ["bpjs-kesehatan.go.id", "bpjsketenagakerjaan.go.id"],
    "kemensos": ["kemensos.go.id"],
    "pln": ["pln.co.id"],
    # global
    "paypal": ["paypal.com"],
    "apple": ["apple.com", "icloud.com"],
    "google": ["google.com", "google.co.id", "googleapis.com", "googleusercontent.com"],
    "microsoft": ["microsoft.com", "microsoftonline.com"],
    "facebook": ["facebook.com", "facebook.net", "fb.com"],
    "whatsapp": ["whatsapp.com", "whatsapp.net"],
    "instagram": ["instagram.com"],
    "netflix": ["netflix.com"],
    "amazon": ["amazon.com", "amazonaws.com"],
}
# Dipertahankan untuk kompatibilitas dengan kode lain yang mungkin mengimpornya
IMPERSONATED_BRANDS = list(BRANDS.keys())

# Sufiks dua tingkat (agar registered domain "bca.co.id" terbaca utuh)
_SECOND_LEVEL = {"co.id", "go.id", "ac.id", "or.id", "web.id", "my.id", "sch.id",
                 "biz.id", "net.id", "co.uk", "com.sg", "com.my"}


def _registered_domain(host: str) -> str:
    parts = host.lower().strip(".").split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _tokens(s: str) -> list:
    return [t for t in re.split(r'[^a-z0-9]+', s.lower()) if t]


def _token_has_brand(token: str, brand: str) -> bool:
    """Merek pendek (<= 4 huruf) harus cocok persis/varian umum; merek panjang cukup terkandung."""
    if len(brand) <= 4:
        return token in (brand, f"klik{brand}", f"my{brand}", f"m{brand}",
                         f"{brand}online", f"{brand}mobile", f"{brand}id")
    return brand in token


def _find_brand(tokens: list):
    for brand in BRANDS:
        if any(_token_has_brand(t, brand) for t in tokens):
            return brand
    return None


TYPOSQUAT_ENABLED = True   # False = perilaku v2.1 (untuk uji ablasi)


def _edit_distance(a: str, b: str) -> int:
    """Jumlah minimum huruf yang perlu diganti/ditambah/dihapus agar a menjadi b (Levenshtein)."""
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _find_typosquat(tokens: list):
    """v2.2: token yang BEDA SATU HURUF dari nama merek >= 6 huruf ('posindonesla', 'whctsapp').
    Merek pendek tidak dicek karena terlalu banyak kata biasa yang mirip."""
    for t in tokens:
        for brand in BRANDS:
            if len(brand) >= 6 and t != brand and brand not in t and _edit_distance(t, brand) == 1:
                return brand
    return None


def _is_shortener(domain: str):
    d = domain.lower()
    return next((s for s in SHORTENERS if d == s or d.endswith("." + s)), None)


def _detect_file_risk(url: str) -> dict:
    url_lower = url.lower()
    clean_path = url_lower.split("?")[0].split("#")[0]
    dangerous = next((e for e in DANGEROUS_EXTENSIONS if clean_path.endswith(e)), None)
    macro = next((e for e in MACRO_EXTENSIONS if clean_path.endswith(e)), None)
    return {
        "has_dangerous_file": bool(dangerous),
        "dangerous_extension": dangerous or "-",
        "has_macro_file": bool(macro),
        "macro_extension": macro or "-",
    }


def _detect_impersonation(url: str, domain: str, shortener) -> dict:
    """Cari nama merek di subdomain, nama domain, atau path tautan pendek."""
    reg = _registered_domain(domain)
    sub = domain.lower()[: -len(reg)].strip(".") if domain.lower().endswith(reg) else ""
    reg_name = reg.split(".")[0]

    brand, where = None, "-"
    b = _find_brand(_tokens(sub))
    if b:
        brand, where = b, "subdomain"
    else:
        b = _find_brand(_tokens(reg_name))
        if b:
            brand, where = b, "domain"
        elif shortener:
            path = urllib.parse.urlparse(url).path
            b = _find_brand(_tokens(path))
            if b:
                brand, where = b, "path tautan pendek"
    if TYPOSQUAT_ENABLED and not brand and not shortener:
        b = _find_typosquat(_tokens(sub) + _tokens(reg_name))
        if b:
            brand, where = b, "nama mirip (beda 1 huruf)"

    # Domain resmi merek tersebut bukan peniruan
    if brand and (reg in BRANDS[brand] or domain.lower() in BRANDS[brand]):
        brand, where = None, "-"

    return {
        "deceptive_subdomain": bool(brand),   # nama kunci lama dipertahankan (= ada peniruan merek)
        "impersonated_brand": brand or "-",
        "impersonation_location": where,
        "real_domain": reg,
    }


def analyze_urls(urls: list) -> list:
    results = []
    for url in urls:
        domain = url.split("/")[2] if "://" in url else url.split("/")[0]
        domain = domain.split(":")[0].lower()

        all_encodings = re.findall(r'%[0-9a-fA-F]{2}', url)
        encoded = bool(all_encodings)
        decoded = urllib.parse.unquote(url)

        punycode = any(label.startswith("xn--") for label in domain.split("."))
        punycode_decoded = domain
        if punycode:
            try:
                punycode_decoded = idna.decode(domain)
            except Exception:
                pass

        cyrillic_matches = list(set(re.findall(r'[Ѐ-ӿ]', punycode_decoded)))
        cyrillic = bool(cyrillic_matches)
        shortener_used = _is_shortener(domain)

        file_risk = _detect_file_risk(url)
        deceptive = _detect_impersonation(url, domain, shortener_used)

        risk_score = 0
        if punycode or cyrillic: risk_score += 3
        if file_risk["has_dangerous_file"]: risk_score += 3
        if file_risk["has_macro_file"]: risk_score += 2
        if deceptive["deceptive_subdomain"]: risk_score += 2
        if shortener_used: risk_score += 1
        if encoded: risk_score += 1

        if risk_score >= 3: risk_level = "TINGGI"
        elif risk_score >= 1: risk_level = "SEDANG"
        else: risk_level = "RENDAH"

        results.append({
            "url": url,
            "domain": domain,
            "risk_level": risk_level,
            "encoded": encoded,
            "encoded_chars": ", ".join(set(all_encodings)) if all_encodings else "-",
            "decoded": decoded,
            "punycode": punycode,
            "punycode_decoded": punycode_decoded if punycode else "-",
            "cyrillic": cyrillic,
            "cyrillic_chars": ", ".join(cyrillic_matches) if cyrillic_matches else "-",
            "shortlink": bool(shortener_used),
            "shortener_used": shortener_used or "-",
            **file_risk,
            **deceptive,
        })
    return results
