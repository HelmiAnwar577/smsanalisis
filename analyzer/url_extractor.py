"""
url_extractor.py v1.1
=====================
v1.1: ditambah TLD .at, .li, .bet, .fun, dll. Di v1.0 "shorturl.at/QqqYB" tidak
terbaca sama sekali sebagai tautan.

Mengambil semua tautan dari teks SMS, termasuk tautan TANPA skema
(contoh: "bit.ly/hadiah-bri", "tsel.me/promo", "www.tokopalsu.com").

Tautan tanpa skema dinormalisasi menjadi "http://..." supaya modul lain
(url_analyzer, domain_reputation, redirect_trace) bisa memprosesnya.

Dipakai oleh: app.py (disarankan), evaluasi.py, uji_fungsional.py
"""

import re

# Daftar TLD yang dikenali untuk tautan tanpa skema.
# Sengaja dibatasi agar teks seperti "Rp.50.000" atau "a.n." tidak dianggap tautan.
_TLDS = (
    "com|id|ly|me|gl|co|net|org|top|xyz|info|site|online|cc|io|gy|us|biz|"
    "click|link|live|shop|store|app|win|vip|club|icu|cfd|sbs|buzz|ws|to|tk|ml|ga|cf|gq|"
    # v1.1: TLD yang sering dipakai pemendek tautan & situs judi online
    # (shorturl.at, did.li, dst.). Sudah dicek: tidak menimbulkan salah tangkap di dataset.
    "at|li|pw|fun|bet|asia|one|pro|mobi|tv|lol|cyou|bio|ink|cam|fit|life|today|vin|my"
)

URL_REGEX = re.compile(
    r'((?:https?://|www\.)[^\s,<>"\']+'                       # berskema atau diawali www.
    r'|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:' + _TLDS + r')\b'   # tanpa skema: domain.tld
    r'(?:/[^\s,<>"\']*)?)',                                   # path opsional
    re.IGNORECASE,
)


def extract_urls(text: str) -> list:
    """Kembalikan daftar URL unik (urutan kemunculan dipertahankan), sudah berskema."""
    urls = []
    for raw in URL_REGEX.findall(text or ""):
        u = raw.rstrip('.)!?;:')
        if not u.lower().startswith(("http://", "https://")):
            u = "http://" + u
        if u not in urls:
            urls.append(u)
    return urls
