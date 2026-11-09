import urllib.parse
import idna
import re

SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "s.id", "cutt.ly", "rb.gy", "ow.ly", "short.io", "tiny.cc"]
DANGEROUS_EXTENSIONS = ['.apk', '.exe', '.scr', '.bat', '.cmd', '.vbs', '.ps1', '.msi', '.jar', '.dmg']
MACRO_EXTENSIONS = ['.xlsm', '.docm', '.xlsb', '.pptm', '.xltm', '.dotm', '.xlam', '.ppam']
IMPERSONATED_BRANDS = ["shopee", "tokopedia", "gojek", "grab", "bca", "bri", "bni", "mandiri", "paypal", "apple", "google", "microsoft", "facebook", "whatsapp", "instagram", "netflix", "amazon"]

def _detect_file_risk(url: str) -> dict:
    url_lower = url.lower()
    clean_path = url_lower.split("?")[0].split("#")[0]
    dangerous = next((e for e in DANGEROUS_EXTENSIONS if clean_path.endswith(e)), None)
    macro     = next((e for e in MACRO_EXTENSIONS if clean_path.endswith(e)), None)
    return {
        "has_dangerous_file": bool(dangerous),
        "dangerous_extension": dangerous or "-",
        "has_macro_file":      bool(macro),
        "macro_extension":     macro or "-",
    }

def _detect_deceptive_subdomain(domain: str) -> dict:
    parts = domain.lower().split('.')
    if len(parts) < 3:
        return {"deceptive_subdomain": False, "impersonated_brand": "-", "real_domain": domain}
    subdomain_str = ".".join(parts[:-2])
    found_brand   = next((b for b in IMPERSONATED_BRANDS if b in subdomain_str), None)
    return {
        "deceptive_subdomain": bool(found_brand),
        "impersonated_brand":  found_brand or "-",
        "real_domain":         ".".join(parts[-2:]),
    }

def analyze_urls(urls: list) -> list:
    results = []
    for url in urls:
        domain = url.split("/")[2] if "://" in url else url.split("/")[0]
        domain = domain.split(":")[0]

        all_encodings = re.findall(r'%[0-9a-fA-F]{2}', url)
        encoded       = bool(all_encodings)
        decoded       = urllib.parse.unquote(url)

        punycode         = domain.startswith("xn--")
        punycode_decoded = domain
        if punycode:
            try:
                punycode_decoded = idna.decode(domain)
            except Exception:
                pass

        cyrillic_matches = list(set(re.findall(r'[\u0400-\u04FF]', punycode_decoded)))
        cyrillic         = bool(cyrillic_matches)
        shortener_used   = next((s for s in SHORTENERS if s in domain), None)

        file_risk = _detect_file_risk(url)
        deceptive = _detect_deceptive_subdomain(domain)

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
            "url":                url,
            "domain":             domain,
            "risk_level":         risk_level,
            "encoded":            encoded,
            "encoded_chars":      ", ".join(set(all_encodings)) if all_encodings else "-",
            "decoded":            decoded,
            "punycode":           punycode,
            "punycode_decoded":   punycode_decoded if punycode else "-",
            "cyrillic":           cyrillic,
            "cyrillic_chars":     ", ".join(cyrillic_matches) if cyrillic_matches else "-",
            "shortlink":          bool(shortener_used),
            "shortener_used":     shortener_used or "-",
            **file_risk,
            **deceptive,
        })
    return results