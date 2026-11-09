# SMS Defender v6.1 — Catatan Perubahan

## Penyesuaian judul baru (31 Oktober 2026, malam)

Judul baru: *Sistem Deteksi Smishing Menggunakan Kombinasi Metode Naive Bayes Classifier dan
Skoring Risiko dengan Penjelasan Berbasis Generative AI*. **Hanya teks tampilan yang diubah;
logika deteksi, bobot, dan hasil pengujian tidak berubah** (uji black-box tetap 13/13).

| File | Perubahan |
|---|---|
| `templates/index.html` | Subjudul menjadi "Sistem Deteksi Smishing (Naive Bayes + Skoring Risiko + Generative AI)". Footer tidak lagi tertulis "v4.0", sekarang mengikuti versi aplikasi. |
| `templates/result.html` | Footer disamakan dengan judul baru. |
| `app.py` | Halaman utama mengirim nomor versi ke tampilan; keterangan di awal file disesuaikan. |

## Yang baru di v6.1 (31 Oktober 2026)

Berdasarkan versi teman (label "Heuristik" diganti "Risk Scoring"/"Indikator", bagian
Data Forensik dihapus dari halaman hasil). Logika deteksi tidak diubah, kecuali:

| File | Perubahan |
|---|---|
| `analyzer/scoring_engine.py` | Naive Bayes **tidak dihitung untuk SMS kurang dari 3 kata** (`nb_min_words = 3`). Sebelumnya satu kata seperti "menang" atau kata kasar langsung berstatus Waspada. Tidak ada SMS penipuan di data (lama maupun 2026) yang kurang dari 3 kata. |
| `eksperimen/uji_blackbox.py` | Skenario 5, 6, 7, 9, 10, 12 disesuaikan dengan tampilan baru (tanpa Data Forensik). Skenario 9 kini menguji tautan tiruan merek (whctsapp). Hasil: 13/13. |
| `eksperimen/` | Dikembalikan ke dalam proyek supaya semua angka artikel bisa direproduksi. |

**Hasil setelah perubahan (tidak berubah dari v6.0):** F1 data 2026 92,23%; salah tuduh 0,6/172;
uji fungsional 10/10; uji black-box 13/13; batas Aman 30 tetap terbaik pada uji sensitivitas.

---

# SMS Defender v6.0 — Catatan Perubahan

## Yang baru di v6.0 (28 Oktober 2026): siap menghadapi SMS 2026

Judul tetap: *Multinomial Naive Bayes + Analisis Heuristik Tautan (hibrida)*.

| File | Perubahan |
|---|---|
| `analyzer/text_classifier.py` | TF-IDF **per potongan 2–4 huruf** (char n-gram) + **Multinomial Naive Bayes** (alpha 0,01). Dipilih dengan validasi silang di data latih lama saja. Data SMS 2026 ikut dilatih (digandakan 3×, sama dengan contoh peneliti). |
| `analyzer/url_analyzer.py` | Deteksi **tautan mirip merek** (beda 1 huruf): `posindonesla` → posindonesia, `whctsapp` → whatsapp. Domain resmi yang belum terdaftar ditambahkan (indosat.com, ioh.co.id, shopeepay.co.id, grab.co). |
| `analyzer/scoring_engine.py` | **Aturan fusi direvisi:** (1) satu indikator tautan kritis (merek tiruan, punycode, .apk, makro) → minimal Waspada; (2) pemendek tautan (bit.ly) tidak lagi dihitung sebagai bukti kedua. |
| `analyzer/data/data_2026/` | **Baru.** 32 SMS penipuan (28 judol, 4 smishing) + 172 SMS sah dari 2 HP, 5 SMS judol pengembangan. Nomor dan nama disensor. 4 SMS berlabel `ragu` tidak dipakai. |
| `eksperimen/evaluasi_2026.py` | **Baru.** Uji ablasi A–E, validasi silang 5-fold ×10 pada data 2026. |

### Hasil uji ablasi (`python eksperimen/evaluasi_2026.py`)

| Konfigurasi | F1 data 2026 | Salah tuduh 2026 (dari 172) | F1 data uji lama |
|---|---|---|---|
| A. Sebelum perbaikan | 13,79% | 22,0 | 85,47% |
| B. + NB per potongan huruf | 34,21% | 31,0 | 91,34% |
| C. + data 2026 di data latih | 75,07% | 12,6 | 86,11% |
| D. + deteksi tautan mirip | 75,24% | 12,6 | 86,11% |
| **E. + aturan fusi direvisi (v6)** | **92,23%** | **0,6** | 85,21% |
| NB saja (konfigurasi E) | 89,68% | 0,6 | 87,00% |

Catatan jujur untuk artikel:
- Aturan fusi E disusun **setelah** melihat kesalahan konfigurasi D pada data 2026, sehingga angka E
  berpotensi sedikit optimis. Sebutkan ini di keterbatasan penelitian.
- Pada data uji lama, sistem hibrida (85,21%) sedikit di bawah NB saja (87,00%); keunggulan hibrida
  terlihat pada data 2026.
- Data 2026 masih kecil (32 penipuan, dari 2 HP).

---

# SMS Defender v5.1 — Catatan Perubahan

## Yang baru di v5.1 (Oktober 2026): tahan terhadap ejaan samaran SMS judi online

Dipicu oleh 5 SMS judol dari HP peneliti (`analyzer/data/data_2026/sms_judol_pengembangan.csv`).
Di v5.0, 3 dari 5 lolos dengan status Aman (skor 0).

| File | Perubahan |
|---|---|
| `analyzer/text_normalizer.py` | **Baru.** Mengembalikan ejaan samaran: "R A J A" → "raja", "daf-tar"/"sa.ldo"/"ju'ta" → "daftar"/"saldo"/"juta", "h0ki" → "hoki", "Hdiah" → "hadiah". Setiap perubahan dicek ke kamus kata, jadi "anak-anak", "s.d", "wr.wb", "top-up" tidak diubah. Tautan tidak disentuh. |
| `analyzer/sms_analyzer.py` | Kata kunci dicari di teks yang sudah dinormalisasi. Ditambah kata khas judol (slot, gacor, maxwin, togel, cuan, hoki, dll.). "bonus", "daftar", "saldo", "member" sengaja tidak dimasukkan karena sering muncul di promo resmi dan notifikasi bank. |
| `analyzer/scoring_engine.py` | Indikator baru S_A "Ejaan sengaja disamarkan" (+20) bila ada ≥ 2 penyamaran. Bobot dan ambang dipilih dengan validasi silang di data latih. |
| `analyzer/text_classifier.py` | Naive Bayes membaca teks yang sudah dinormalisasi (`USE_NORMALIZER = True`). |
| `analyzer/url_extractor.py` | TLD `.at`, `.li`, `.bet`, `.fun`, dll. ditambahkan (dulu `shorturl.at/...` tidak terbaca sama sekali). |
| `analyzer/url_analyzer.py` | Pemendek tautan baru: `did.li`, `rebrand.ly`, `heylink.me`, dll. |
| `templates/result.html` | Baris baru "Ejaan Disamarkan" beserta teks yang sudah dibaca ulang. |
| `model_nb.pkl`, `vectorizer.pkl` | Dilatih ulang. |

**Hasil di data uji lama (229 SMS, tidak memburuk):** F1 Naive Bayes 85,47% → 86,44%;
F1 sistem gabungan 83,76% → 84,75%. Uji black-box tetap 13/13.

**Hasil pada 5 SMS judol:** tertangkap 2/5 → 3/5. SMS 3 ("Member baru dapat…") dan
SMS 4 ("Kakak akun Anda 1 juta…") masih lolos: tidak ada penyamaran dan kata-katanya
belum pernah muncul di dataset 2013–2016. Ini butuh data latih baru, bukan aturan tambahan.

**Penting untuk artikel:** 5 SMS ini dipakai untuk MERANCANG perbaikan, jadi tidak boleh
dipakai sebagai data UJI. Angka uji judol di artikel harus dari SMS lain yang dikumpulkan
kemudian.

---

# SMS Defender v5.0 — Catatan Perubahan

Ini proyek `emailanalisis2` milikmu yang sudah dipasangi kode v5, siap dijalankan.

## Cara menjalankan

```bash
pip install -r requirements.txt
python app.py
```

Buat file `.env` di folder ini (jangan pernah diunggah ke mana pun):

```
GEMINI_API_KEY=isi_kunci_gemini
VT_API_KEY=isi_kunci_virustotal   # opsional
```

**Mode offline** (untuk demo saat internet ruangan tidak stabil; reputasi domain dan
redirect dimatikan, sama seperti saat evaluasi di artikel):

```bash
set SMS_DEFENDER_OFFLINE=1          # Windows CMD
$env:SMS_DEFENDER_OFFLINE="1"       # PowerShell
python app.py
```

## Menjalankan evaluasi (semua angka artikel)

```bash
python eksperimen/evaluasi.py                  # Tabel 3, 4, 5 + validasi silang + kestabilan + uji data Inggris
python eksperimen/uji_fungsional.py --offline  # Tabel 6
python eksperimen/uji_blackbox.py              # Tabel 7 (13 skenario black-box)
```

Hasil tersimpan di `eksperimen/hasil/`. Angkanya selalu sama setiap dijalankan.

## Yang berubah dari versi yang kamu kirim

| Bagian | Perubahan |
|---|---|
| `analyzer/` | Modul v5 (lihat docstring di tiap file). `llm_evaluator.py` tidak berubah. |
| `analyzer/data/` | Folder `Data` diganti `data` (huruf kecil; Linux membedakan huruf besar-kecil). Berisi `dataset_sms_spam_v1.csv` (dataset Indonesia, dipakai) dan `Dataset_5971.csv` (Inggris, hanya untuk perbandingan). |
| `app.py` | Memakai `analyzer/url_extractor.py`; bila ada beberapa tautan, dipilih reputasi domain paling berisiko; mode offline; `/health` menampilkan versi 5.0. |
| `templates/result.html` | Kolom "File Berbahaya" dulu selalu "Tidak" karena membaca kunci yang salah (`dangerous_ext`); sekarang benar. Kolom baru "Peniruan Merek". Jumlah trigger hanya menghitung indikator berbobot; penanda info ditulis "(info)", bukan "(+0)". |
| `requirements.txt` | Ditambah `python-dotenv` (dipakai `app.py` tapi belum tercantum). Paket yang tidak dipakai dihapus (`pygal`, `IPy`, `geoip2`, `genai`, `pdfplumber`, `kagglehub`). |
| `eksperimen/` | Baru: `evaluasi.py`, `uji_fungsional.py`, `uji_blackbox.py`, hasilnya, dan `versi_lama/` (kode + model sebelum perbaikan sebagai pembanding). |
| Skrip lama | `evaluasi_nb.py`, `evaluasi_nb_v2.py`, `test_nb.py`, `uji_fungsional_baru.py`, `hasil_uji_40.json` dipindah ke `eksperimen/versi_lama/skrip_lama/`. Skrip ini memakai fungsi lama yang sudah tidak ada, jadi tidak bisa dijalankan dengan kode v5. |

## Bug di skrip lama yang perlu kamu tahu

- `uji_fungsional_baru.py` membandingkan status dengan `"Waspada"`, padahal sistem
  mengeluarkan `"WASPADA"` (huruf besar), sehingga semua SMS tertulis "Gagal".
  Rumus F1-nya juga dikali 100 dua kali, sehingga `hasil_uji_40.json` menulis F1 = 6666,7%.
- Beberapa teks di 80 SMS uji itu rusak (kata acak, karakter Mandarin, spasi aneh),
  jadi tidak layak dipakai di artikel tanpa diperbaiki.
