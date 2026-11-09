"""
redirect_trace.py v2.0
=======================
Tambahan v2:
- Tandai setiap hop dengan flag cross_domain
- Hitung jumlah perpindahan domain
- Header User-Agent realistis agar tidak diblokir server
v2.1:
- stream=True: hanya header yang dibaca, isi berkas (mis. .apk) tidak ikut diunduh
- Peringatan InsecureRequestWarning dimatikan (verify=False memang disengaja)
"""

import requests
import urllib3
from urllib.parse import urlparse

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def _get_netloc(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def trace_redirect(url: str) -> list:
    """
    Lacak seluruh rantai redirect dari URL awal.

    Setiap entry:
    {
        "url"          : str,
        "status"       : int | str,
        "cross_domain" : bool,   # True jika domain berubah dari hop sebelumnya
    }
    """
    chain = []

    try:
        response = requests.get(
            url,
            allow_redirects=True,
            timeout=7,
            headers=_HEADERS,
            verify=False,         # jangan blok karena sertifikat expired
            stream=True,          # jangan unduh isi berkas
        )
        response.close()

        # History = semua redirect intermediate
        all_responses = list(response.history) + [response]
        prev_domain   = None

        for r in all_responses:
            current_domain = _get_netloc(r.url)
            cross          = bool(prev_domain and current_domain and prev_domain != current_domain)

            chain.append({
                "url":          r.url,
                "status":       r.status_code,
                "cross_domain": cross,
            })
            prev_domain = current_domain

    except requests.exceptions.SSLError:
        chain.append({"url": url, "status": "SSL Error", "cross_domain": False})
    except requests.exceptions.ConnectionError:
        chain.append({"url": url, "status": "Error/Unreachable", "cross_domain": False})
    except requests.exceptions.Timeout:
        chain.append({"url": url, "status": "Timeout", "cross_domain": False})
    except requests.exceptions.RequestException as e:
        chain.append({"url": url, "status": f"Error: {str(e)[:60]}", "cross_domain": False})

    return chain
