# -*- coding: utf-8 -*-
"""
uji_fungsional_baru.py — Pengujian fungsional sistem dengan 40 SMS BARU
(yang belum pernah dipakai untuk perbaikan sistem), sesuai permintaan umum
reviewer. Hanya modul offline yang dijalankan (NB + heuristik / skoring);
modul WHOIS/VirusTotal/Gemini/redirect tidak diakses internet (opsional).
Hasil dinilai: penipuan benar jika status Waspada/Berbahaya; sah benar jika Aman.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer.url_analyzer import analyze_urls
from analyzer.sms_analyzer import extract_sms_entities
from analyzer.scoring_engine import calculate_score
from analyzer.text_classifier import predict_text_phishing

# ---------- 40 SMS BARU ----------
PENIPUAN = [
    "Selamat! Anda memenangkan undian Telkomsel Rp 25 Juta. Klaim sekarang di bit.ly/undian-telkomsel",
    "DANA: Akun Anda akan dibekukan hari ini. Verifikasi segera di danaid-verif.xyz",
    "Paket Anda gagal dikirim siCepat. Bayar ongkir Rp 35.000 di sicepat-bayar.com/bayar",
    "Kartu kredit BCA Anda diblokir. Buka resmi-bca.id/unblock untuk aktivasi kembali",
    "Kemenangan TF Rp 100 Juta dari undian Lazada! Transfer pajak Rp 250rb ke rekening ini",
    "Unduhan HP Anda terpilih mendapat 4GB data gratis. Isi data di telkomsel-poin.top",
    "Peringatan data PRU rutin: KYC Anda bermasalah. Update sekarang jne-lacak.id/KYC",
    "Halo, saya Pak Anton BCA. Ada transaksi Rp 4.900.000 ya? Semua info ke +6281234xxx",
    "BRIVA: Metro Fax BRI tinggal 1 jam. Cek sisadana di bri-cek.us?Klaim",
    "Menangin name 777 slot! Deposit 50rb dapat 100rb. Hubungi admin WA +628999...",
    "Notifikasi Amazon: akun Anda segera dihapus! Ketik kata sandi di amzn-id.top/login",
    "Ahli waris! Ada harta peninggalan Rp 2 miliar. Hubungi advokat 08512345678",
    "Kartu ATM tertahan di mesin BCA. Selesaikan di atm-bcaklad.com/aktivasi",
    "Bansos PKH tahap 7 sudah cair. Cek penerima di(space) bansos-pkh.id/daftar",
    "Anda ditangkap utk kasus penipuan. Bayar denda Rp 5 juta via briva; hubungi ibu amt 0812xxx",
    "Pimpinan baru Telkomsel bagi-bagi prabayar 20GB. Cek di telkomselku-4g.top/claim",
    "Taplus BNI Anda masuk daftar hitam! Isi ulang data: bni-taplus.com/update",
    "Bonvote UNDURBAL pakai referensi 9384512F; klaim hadiah 5jt di cutt.ly/bonvote123",
    "GoPay: Refund Rp 890.000 terkunci. Buka gopay-refund.com/safe, buka protokol",
    "Ups, ovo Anda hangus 12 jam lagi. Top up sekarang di ovo-kendali.top",
    "Survei BLK jawab pertanyaan bebas link 3jt: link fisik denda 07 di bit.ly/blk-3jt",
    "GRATIS pulsa 100rb utk pelanggan setia Indosat. Klaim sekarang di im3-gratis.top",
    "Pos J&T: cekfungsi pngmsran paket 40kg buruh: cek dok registri di jntChecked.com",
    "Rp juta token listrik PLN distrack: sisa token 1,2 kWh. Beli cepat pln-token.top",
    "Selamat hari tax: &oot anggota: klaim gaz promoo A17 3jt di shopee-11-11.com",
    "Kartu kredit cwhite expired; batas kartu diperpanjang 500jt! App ini bank-mgg.id",
    "Agen pulsa\240 dibayar komisi 10%. Deposit 500rb via mutasi REK 123456789 BRI",
    "Rapat online DocuSign: 7 kurir berkas ktp Anda; klik docheck.co.id/KTP2028",
    "Mama!\240 jadi butuh uang segera 1jt??? topup Diamond ML boleh pakai pulsa, chips murah!",
    "Network drop sebesar 4.9999 Anda Memenangkan tiket buka puasa ke Umroh Mei 2025!",
    "BPS\240 tanya: survey nelayan jawab sevices; dp bit.ly/survei_130; cauwa: ajin 7jt",
    "Pesanan gaya: reduk komisi 85.000 pakai e-wallet sabit di tokopayonline.info",
    "Jaket popok SL-221\240 nationwide: sekali transfer (ita max 19 dapta dgz5), 1 !",
    "Informasi panen PT Dinasti: CPO fermant Anda saldo keter produsen, klaim 5jt df",
    "Kawem memilih garansi: ads TOOL mana jab Welch garansi hafal ID peng(indices)",
    "Ahoy bung boanean eksplor FB marka zuck: beck Tin link bit.ly/fb-marka",
    "Rahasia dagang: sistem dapat 3jt/hari tanpa kerja! Modul khusus bit.ly/kerja3jt",
    "Keberuntungan satu-satunya minggu ini! BCA pilih Anda utk pardorama 16 GB",
    "Pakamal: sisa bayar 349.000; bayar nomor rekening ZIP: pinjaman andaBit OK",
    "OJK mencatat pinjol Anda macet. Pelunasan resmi di ojk-pelunasan.id (wajib hari ini)",
]

SAH = [
    "OTP BCA Anda 550291. Jangan berikan kode ini kepada siapa pun, termasuk pihak BCA.",
    "Transaksi No SA-77213: Pembayaran QRIS Rp 45.000 di Kopi Kenangan berhasil.",
    "Halo besok libur ya, jadi kabarin kalau jadi main futsal sore.",
    "Jadwal booking servis resmi sudah terkonfirmasi: Sabtu, 09.00 di bengkel berarsip.",
    "Terima kasih sudah bertransaksi 250rb di Alfamart Tunjungan baru kembalian 250.000.",
    "Paket Shopee sudah diantar. Titip di pos satpam ya, terima kasih banyak!",
    "Kuota Anda tersisa 4,5 GB berlaku sampai 27, jangan lupa cek paket MyTelkomsel.",
    "Rapat unit diundur ke besok jam 10 di ruang rapat lantai 2 ya.",
    "Saldo GoPay Anda Rp 187.500. Transaksi terakhir: makan siang kantin Rp 18.000.",
    "Pemberitahuan: rekening Anda aman, tidak ada transaksi mencurigakan hari ini.",
    "Info kereta: KA Argo Bromo berangkat pukul 07.30, gerbang 4, peron 1.",
    "Kartu kredit Anda telah aktif dan siap digunakan sejak hari ini. Selamat!",
    "Tagihan listrik bulan ini Rp 312.500 dengan tunggakan nol. Terima kasih.",
    "Kalimat dari ibu: nanti mampir ke rumah ya, ada sate buatan tetangga.",
    "Reminder klinik gigi besok 13:30, dokter Zulkifli ruang B paket gigi normal ini.",
    "Status pengiriman JNE: paket tiba di cabang Medan; silakan ambil sebelum 5 hari.",
    "Password WiFi dicatat: rumah-dua-lima-tiga-empat, tunggu dulu jangan bagikan.",
    "Briefing tim besok via Zoom pukul 09.00, link di calendar. Terima kasih.",
    "Meteran air Juli: 4 bulan 22 m3; tagihan Rp 96.800, bayar sampai 25 di kantor.",
    "Anggota koperasi baru diukk 5 juta bulanan. Resiko kecukupan sesuai ketentuan KSP.",
    "Info VM: ruang 24/7 (ada turun ) di mini market sama di bawah kantor kuru.",
    "Pizza havualan hari ini pakai update topping baru saborergic; pesan online ya!",
    "Hujan lampau di gambut kemarin ditingkatkan solusi nan patut dicermati 750.",
    "Bukti transaksi: topup e-money Anda Rp 100.000 via m-banking berhasil pukul 14:05.",
    "Undangan rapat yadi: kapan venue gajah mada dekat FB check past, jadi berbagi.",
    "Absen bulanan tim konten: total 12 dari 14 hari kerja, sisanya izin 2 hari.",
    "Guru Les privat hari Senin tunda jadi hari Rabu jam 16.00 karena rapat sekolah.",
    "Survei kepuasan nasabah bca men_top: jawab prima pro data! Baik, sangat baik!",
    "Paket ganti kartu ATM已经 sudah masuk output hari ini; eci hati-hati di jalan.",
    "Berangkat ke kantor pukul 06.30, macet pagi di jalan irian ya!",
    "Seni film yuk malam ini ada yang bagus di bioskop 21 pakai promo tiptop 50%.",
    "Pak Kartono pesan batubara 5 ton untuk stok boiler; invoice mengikuti besok.",
    "Ronda RT 04 malam ini dijaga regu 2: Bapak Hartono & Bapak Simanullang ya.",
    "Aku di jalan 15 menit lagi, jalanan macet banget dekat terminal.",
    "Sekarang tiket diskusi terakhir sudah konfirmasi; kamu boleh datang besok.",
    "Kurs BCA hari ini: beli USD 15.570, jual USD 15.780. Informasi setiap jam.",
    "Ambil dompet hitam hilang di Purwokerto, ada kartu pelajar & uang Rp 50rb.",
    "Congrats kelas Python selesai! Nilai akhir Anda 92, sertifikat terkirim via email.",
    "Kuis mingguan fotografi dimulai jam 20 di grup WA, peserta wajib hadir ya.",
    "Parkir elektronik Grand Asia Afrika: saldo Rp 12.000, entri truk sudah tutup."
]

# SMS nomor 36 & 44 ada string janggal sengaja "berani" komentar:
# angka 36: string Python tanpa koma sengaja digabung -> perbaiki manual di bawah.
SAH[34] = "Aku di jalan 15 menit lagi, jalanan macet banget dekat terminal"
SAH[39] = "Parkir elektronik Grand Asia Afrika: saldo Rp 12.000, gerbang sudah tutup"

assert len(PENIPUAN) == 40, len(PENIPUAN)
assert len(SAH) == 40, len(SAH)


def analisis(teks):
    ent = extract_sms_entities(teks)
    urls = [u for u in _ekstrak_url(teks)]
    url_an = analyze_urls(urls)
    # domain reputasi & redirect & VT tidak diakses (kosong) supaya uji murni offline
    prob = predict_text_phishing(teks)
    skor = calculate_score(
        url_analysis=url_an,
        domain_info={},
        redirect_chain=[],
        sms_entities=ent,
        phishing_probability=prob,
    )
    return skor


def _ekstrak_url(teks):
    import re as _re
    rx = _re.compile(
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
        r'|\b(?:[a-z0-9-]+\.)+(?:com|net|org|id|co\.id|ly|xyz|top|cc|tk|ml|ga|cf|gq|pw|info|site|online|link|me|io|gl|co|app|click|us)\b(?:/[^\s<>"\']*)?',
        _re.IGNORECASE,
    )
    hasil = []
    for u in rx.findall(teks):
        u = u.rstrip('.,;:!?)')
        if not u.lower().startswith(("http://", "https://")):
            u = "http://" + u
        hasil.append(u)
    return list(dict.fromkeys(hasil))


def fmt(x):
    return f"{x:.2f}".replace(".", ",")


def main():
    baris = []
    benar = 0
    for i, sms in enumerate(PENIPUAN, 1):
        s = analisis(sms)
        status = s["status"]
        hasil = "Sesuai" if status in ("Waspada", "Berbahaya") else "Gagal"
        if hasil == "Sesuai":
            benar += 1
        baris.append({"no": i, "jenis": "Penipuan", "skor": s["total_score"],
                      "status": status, "hasil": hasil})

    for i, sms in enumerate(SAH, 41):
        s = analisis(sms)
        status = s["status"]
        hasil = "Sesuai" if status == "Aman" else "Gagal"
        if hasil == "Sesuai":
            benar += 1
        baris.append({"no": i, "jenis": "Sah", "skor": s["total_score"],
                      "status": status, "hasil": hasil})

    # statistik
    tp = sum(1 for b in baris if b["jenis"] == "Penipuan" and b["status"] != "Aman")
    fn = sum(1 for b in baris if b["jenis"] == "Penipuan" and b["status"] == "Aman")
    tn = sum(1 for b in baris if b["jenis"] == "Sah" and b["status"] == "Aman")
    fp = sum(1 for b in baris if b["jenis"] == "Sah" and b["status"] != "Aman")
    akurasi = (tp + tn) / len(baris) * 100
    pres = tp / max(tp + fp, 1) * 100
    rec = tp / max(tp + fn, 1) * 100
    f1 = 2 * pres * rec / max(pres + rec, 1e-9) * 100

    skoor_penipuan = [b["skor"] for b in baris if b["jenis"] == "Penipuan"]
    skoor_sah = [b["skor"] for b in baris if b["jenis"] == "Sah"]

    print(f"TP {tp} | FN {fn} | FP {fp} | TN {tn}")
    print(f"Akurasi {fmt(akurasi)}% | Presisi {fmt(pres)}% | Recall {fmt(rec)}% | F1 {fmt(f1)}%")
    print(f"Skor penipuan: min {min(skoor_penipuan)} - maks {max(skoor_penipuan)}")
    print(f"Skor sah     : min {min(skoor_sah)} - maks {max(skoor_sah)}")
    print()
    for b in baris:
        print(f"no={b['no']:2d} | {b['jenis']:8s} | skor {b['skor']:3d} | {b['status']:9s} | {b['hasil']}")

    with open("hasil_uji_40.json", "w", encoding="utf-8") as f:
        json.dump({"confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
                   "akurasi": akurasi, "presisi": pres, "recall": rec, "f1": f1,
                   "baris": baris}, f, ensure_ascii=False, indent=1)
    print("\nDetail disimpan ke hasil_uji_40.json")


if __name__ == "__main__":
    main()
