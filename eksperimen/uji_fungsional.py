"""
uji_fungsional.py — Pengujian fungsional 10 SMS uji (Tabel 5 di artikel).
=========================================================================
Menjalankan 10 SMS uji melewati SELURUH alur sistem, termasuk modul yang
butuh internet (WHOIS, geolokasi IP, VirusTotal, pelacakan redirect).

    python eksperimen/uji_fungsional.py            # mode lengkap (butuh internet)
    python eksperimen/uji_fungsional.py --offline  # tanpa S_C dan S_D

Catatan:
- Reputasi domain dicek untuk SETIAP tautan secara paralel, lalu diambil yang
  paling berisiko (skor S_C tertinggi) — sama persis dengan app.py.
  Redirect dilacak untuk tautan pertama (sesuai artikel).
- Hasil mode lengkap bisa sedikit berbeda dari hari ke hari karena bergantung
  pada layanan luar (umur domain, status server, VirusTotal).
- Isi GEMINI_API_KEY / VT_API_KEY lewat file .env seperti di app.py bila perlu.
"""

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from analyzer import scoring_engine as se  # noqa: E402
from analyzer.sms_analyzer import extract_sms_entities  # noqa: E402
from analyzer.text_classifier import predict_text_phishing  # noqa: E402
from analyzer.url_analyzer import analyze_urls  # noqa: E402
from analyzer.url_extractor import extract_urls  # noqa: E402

SMS_UJI = [
    ("Penipuan: subsidi BPJS", 1, "Info BPJS: Anda terpilih menerima subsidi Rp 1.200.000. Daftar sekarang di http://subsidi-bpjs.top/daftar sebelum kuota habis."),
    ("Penipuan: kurir + APK", 1, "Kurir SiCepat: paket Anda tertahan. Unduh aplikasi pelacak di http://sicepat-lacak.xyz/resi.apk"),
    ("Penipuan: bank Mandiri", 1, "PERINGATAN: akun Mandiri Anda akan ditutup hari ini. Verifikasi data di bit.ly/mandiri-verif"),
    ("Penipuan: anak minta transfer", 1, "Ini saya anak Anda, ganti nomor baru. Tolong transfer Rp 2 juta dulu ke rekening 1234567890, urgent."),
    ("Penipuan: undian Shopee", 1, "Anda memenangkan undian Shopee 25 juta! Hubungi admin WA 081234567890 untuk klaim hadiah."),
    ("Sah: kode verifikasi", 0, "Kode verifikasi Tokopedia Anda: 774512. Berlaku 5 menit, jangan bagikan ke siapa pun."),
    ("Sah: notifikasi pembayaran", 0, "Pembayaran Rp 150.000 ke Indomaret berhasil. Terima kasih."),
    ("Sah: pesan pribadi", 0, "Bro, meeting besok dipindah jam 3 sore ya, ruangan sama."),
    ("Sah: pengingat servis", 0, "Reminder: servis motor Anda dijadwalkan Sabtu pukul 09.00 di bengkel resmi."),
    ("Sah: pemberitahuan kurir", 0, "Paket Anda sudah sampai dan dititipkan ke satpam. Terima kasih."),
]


def analyze_full(text: str, offline: bool = False) -> dict:
    entities = extract_sms_entities(text)
    urls = extract_urls(text)
    url_analysis = analyze_urls(urls)
    prob = predict_text_phishing(text)

    domain_info, chain = {}, []
    if urls and not offline:
        from analyzer.domain_reputation import get_domain_info
        from analyzer.redirect_trace import trace_redirect
        with ThreadPoolExecutor(max_workers=4) as ex:
            infos = list(ex.map(get_domain_info, urls))
        domain_info = max(infos, key=se.domain_risk_score)
        chain = trace_redirect(urls[0])

    result = se.calculate_score(url_analysis, domain_info, chain, entities, prob)
    return {"prob": prob, "urls": urls, "result": result}


def main():
    offline = "--offline" in sys.argv
    rows = []
    for no, (jenis, label, text) in enumerate(SMS_UJI, start=1):
        a = analyze_full(text, offline=offline)
        r = a["result"]
        b = r["breakdown"]
        terdeteksi = r["status"] != "AMAN"
        sesuai = terdeteksi if label == 1 else not terdeteksi
        rows.append({
            "No": no, "Jenis SMS": jenis, "NB": round(a["prob"], 2),
            "S_A": b["S_A"]["score"], "S_B": b["S_B"]["score"], "S_C": b["S_C"]["score"],
            "S_D": b["S_D"]["score"], "S_E": b["S_E"]["score"],
            "Skor": r["total_score"], "Status": r["status"].capitalize(),
            "Hasil": "Sesuai" if sesuai else "Tidak sesuai",
            "Indikator aktif": "; ".join(f["label"] for f in r["all_flags"]),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"Mode: {'OFFLINE (tanpa S_C & S_D)' if offline else 'LENGKAP (dengan internet)'}")
    print(df.drop(columns=["Indikator aktif"]).to_string(index=False))
    tp = int(((df["No"] <= 5) & (df["Hasil"] == "Sesuai")).sum())
    tn = int(((df["No"] > 5) & (df["Hasil"] == "Sesuai")).sum())
    print(f"\nTP={tp} FN={5 - tp} FP={5 - tn} TN={tn}")

    out_dir = os.path.join(ROOT, "eksperimen", "hasil")
    os.makedirs(out_dir, exist_ok=True)
    name = "uji_fungsional_offline.csv" if offline else "uji_fungsional.csv"
    df.to_csv(os.path.join(out_dir, name), index=False)
    print("Disimpan:", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
