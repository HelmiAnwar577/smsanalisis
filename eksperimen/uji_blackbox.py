"""
uji_blackbox.py — Pengujian black-box aplikasi web (Tabel 7 di artikel).
=========================================================================
Menguji perilaku aplikasi dari sisi pengguna (masukan -> keluaran yang diharapkan)
memakai Flask test client, dalam mode offline agar hasilnya dapat diulang.

    python eksperimen/uji_blackbox.py
"""

import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["SMS_DEFENDER_OFFLINE"] = "1"
os.environ.pop("GEMINI_API_KEY", None)          # memaksa narasi cadangan untuk kasus 11

import app as appmod  # noqa: E402

c = appmod.app.test_client()


def post(text):
    t0 = time.perf_counter()
    r = c.post("/analyze", data={"sms_text": text})
    return r, r.get_data(as_text=True), (time.perf_counter() - t0) * 1000


def skor(html):
    m = re.search(r'risk-score">\s*(\d+)', html)
    s = re.search(r'risk-status">([^<]+)', html)
    return (int(m.group(1)) if m else None), (s.group(1).strip() if s else None)


KASUS = []


def kasus(no, skenario, harapan):
    def deco(fn):
        KASUS.append((no, skenario, harapan, fn))
        return fn
    return deco


@kasus(1, "Membuka halaman utama", "Form input SMS tampil")
def _():
    r = c.get("/")
    return r.status_code == 200 and "<textarea" in r.get_data(as_text=True)


@kasus(2, "Menekan Analisis tanpa teks", "Ditolak dengan pesan peringatan")
def _():
    r, h, _ = post("")
    return r.status_code == 400 and "Tidak ada teks SMS" in h


@kasus(3, "Mengirim teks berisi spasi saja", "Ditolak dengan pesan peringatan")
def _():
    r, h, _ = post("     ")
    return r.status_code == 400 and "Tidak ada teks SMS" in h


@kasus(4, "SMS pribadi tanpa tautan", "Skor 0, status Aman")
def _():
    r, h, _ = post("Bro, meeting besok dipindah jam 3 sore ya, ruangan sama.")
    return skor(h) == (0, "AMAN")


@kasus(5, "SMS kode OTP resmi", "Status Aman; isi SMS tampil di bagian Konten SMS Asli")
def _():
    r, h, _ = post("Kode verifikasi Tokopedia Anda: 774512. Berlaku 5 menit, jangan bagikan ke siapa pun.")
    return skor(h)[1] == "AMAN" and "774512" in h


@kasus(6, "SMS hadiah dengan tautan bit.ly tanpa http", "Indikator pemendek tautan tampil di rincian skor; status Berbahaya")
def _():
    r, h, _ = post("Selamat! Nomor Anda terpilih memenangkan hadiah Rp 50 Juta dari Bank BRI. Klik bit.ly/hadiah-bri untuk klaim")
    return "URL shortener terdeteksi" in h and skor(h)[1] == "BERBAHAYA"


@kasus(7, "SMS kurir dengan tautan berkas .apk", "Indikator berkas berbahaya (.apk) tampil di rincian skor; status Berbahaya")
def _():
    r, h, _ = post("Kurir SiCepat: paket Anda tertahan. Unduh aplikasi pelacak di http://sicepat-lacak.xyz/resi.apk")
    return "File payload berbahaya" in h and ".apk" in h and skor(h)[1] == "BERBAHAYA"


@kasus(8, "Tautan dengan domain punycode", "Indikator homograph/punycode aktif")
def _():
    r, h, _ = post("Login segera di http://xn--bc-xja.com/akun agar rekening tidak diblokir")
    return "Punycode" in h and "Homograph" in h


@kasus(9, "SMS dengan tautan meniru merek (beda 1 huruf)", "Indikator peniruan merek aktif; status minimal Waspada")
def _():
    r, h, _ = post("Permintaan hapus akun WhatsApp diproses. Jika bukan Anda, verifikasi di https://whctsapp.com/id-verify")
    return "Peniruan nama merek" in h and skor(h)[1] in ("WASPADA", "BERBAHAYA")


@kasus(10, "SMS sangat panjang (3.500 karakter)", "Tetap diproses; pratinjau dipotong 3.000 karakter")
def _():
    teks = ("Halo apa kabar, besok kita rapat ya. " * 100)[:3500]
    r, h, _ = post(teks)
    m = re.search(r'class="email-preview">(.*?)</div>', h, re.S)
    return r.status_code == 200 and m is not None and len(m.group(1)) == 3000


@kasus(11, "Layanan Gemini tidak tersedia", "Narasi cadangan tampil; aplikasi tetap berjalan")
def _():
    r, h, _ = post("Paket Anda sudah sampai dan dititipkan ke satpam. Terima kasih.")
    return r.status_code == 200 and "[Mode Offline]" in h


@kasus(12, "Aplikasi dijalankan dalam mode offline", "Skor tetap dihitung tanpa reputasi domain dan redirect")
def _():
    r, h, _ = post("PERINGATAN: akun Mandiri Anda akan ditutup hari ini. Verifikasi data di bit.ly/mandiri-verif")
    return skor(h)[0] is not None and r.status_code == 200


@kasus(13, "Membuka endpoint /health", "JSON berisi status ok dan versi aplikasi")
def _():
    j = c.get("/health").get_json()
    return j.get("status") == "ok" and j.get("version") == appmod.VERSION


def main():
    hasil = []
    for no, skenario, harapan, fn in KASUS:
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001
            ok = False
            print("  error:", e)
        hasil.append((no, skenario, harapan, "Berhasil" if ok else "Gagal"))
        print(f"{no:2d}. {skenario:42s} | {harapan:62s} | {'Berhasil' if ok else 'GAGAL'}")

    # Waktu proses rata-rata (mode offline, tanpa Gemini)
    contoh = ["Selamat! Anda memenangkan hadiah. Klik bit.ly/hadiah",
              "Pembayaran Rp 150.000 ke Indomaret berhasil. Terima kasih.",
              "Kurir: unduh http://lacak-paket.xyz/resi.apk"] * 10
    t = [post(s)[2] for s in contoh]
    print(f"\nWaktu proses rata-rata (mode offline, {len(t)} permintaan): {sum(t)/len(t):.1f} ms")
    lulus = sum(1 for h in hasil if h[3] == "Berhasil")
    print(f"Lulus {lulus}/{len(hasil)} kasus")

    import csv
    out = os.path.join(ROOT, "eksperimen", "hasil", "uji_blackbox.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["No", "Skenario", "Hasil yang diharapkan", "Kesimpulan"])
        w.writerows(hasil)
    print("Disimpan:", out)


if __name__ == "__main__":
    main()
