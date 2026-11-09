"""
domain_reputation.py v1.1
Perubahan: tanggal WHOIS yang memiliki zona waktu tidak lagi membuat umur domain gagal dihitung.
"""

import os
import socket
import base64
from datetime import datetime
from urllib.parse import urlparse
import requests
import whois

def _extract_domain(url: str) -> str:
    if "://" in url:
        return urlparse(url).netloc.split(":")[0]
    return url.split("/")[0].split(":")[0]

def _safe_date(val) -> str | None:
    if val is None: return None
    if isinstance(val, list): val = val[0]
    if isinstance(val, datetime): return val.strftime("%Y-%m-%d")
    return str(val)

def _check_virustotal(url: str) -> dict:
    api_key = os.environ.get("VT_API_KEY")
    if not api_key: return {}

    try:
        url_id  = base64.urlsafe_b64encode(url.encode()).rstrip(b"=").decode()
        headers = {"x-apikey": api_key}

        resp = requests.get(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            positives = stats.get("malicious", 0) + stats.get("suspicious", 0)
            total = sum(stats.values()) or 1
            return {
                "virustotal_positives": positives,
                "virustotal_total":     total,
                "virustotal_ratio":     round(positives / total * 100, 1),
                "virustotal_checked":   True,
            }

        domain = _extract_domain(url)
        resp2  = requests.get(f"https://www.virustotal.com/api/v3/domains/{domain}", headers=headers, timeout=8)
        if resp2.status_code == 200:
            data2 = resp2.json()
            stats2 = data2.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            malicious2 = stats2.get("malicious", 0)
            total2 = sum(stats2.values()) or 1
            return {
                "virustotal_positives": malicious2,
                "virustotal_total":     total2,
                "virustotal_ratio":     round(malicious2 / total2 * 100, 1),
                "virustotal_checked":   True,
            }
    except Exception:
        pass
    return {}

def get_domain_info(url: str) -> dict:
    domain = _extract_domain(url)
    info = {
        "domain":       domain,
        "registrar":    "-",
        "creation_date":"-",
        "domain_age":   "-",
        "ip_address":   "Tidak Ditemukan",
        "country":      "Tidak Ditemukan",
        "city":         "Tidak Ditemukan",
        "isp":          "Tidak Ditemukan",
        "virustotal_positives": 0,
        "virustotal_total":     0,
        "virustotal_ratio":     0.0,
        "virustotal_checked":   False,
    }

    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list): creation = creation[0]
        if isinstance(creation, datetime) and creation.tzinfo is not None:
            creation = creation.replace(tzinfo=None)   # v2: hindari error tanggal ber-timezone
        age = (datetime.now() - creation).days if isinstance(creation, datetime) else None

        info["registrar"]     = w.registrar or "-"
        info["creation_date"] = _safe_date(creation) or "-"
        info["domain_age"]    = age if age is not None else "-"
    except Exception:
        pass

    try:
        ip = socket.gethostbyname(domain)
        info["ip_address"] = ip
        geo = requests.get(f"http://ip-api.com/json/{ip}", timeout=5).json()
        if geo.get("status") == "success":
            info["country"] = geo.get("country", "-")
            info["city"]    = geo.get("city", "-")
            info["isp"]     = geo.get("isp", "-")
    except Exception:
        pass

    info.update(_check_virustotal(url))
    return info