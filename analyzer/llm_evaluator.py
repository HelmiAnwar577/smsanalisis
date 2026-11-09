import os
import json
import google.generativeai as genai

_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

def generate_narrative(score_result: dict, sms_entities: dict, url_analysis: list, domain_info: dict) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback_narrative("API Key Gemini tidak ditemukan di file .env")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_GEMINI_MODEL)

        active_flags = [
            f"{f['category']} | {f['label']} (+{f['weight']})"
            for f in score_result.get("all_flags", [])
        ]

        breakdown = score_result.get("breakdown", {})
        breakdown_summary = {k: f"{v['score']}/{v['max']}" for k, v in breakdown.items()}

        context = {
            "skor_total":     score_result["total_score"],
            "status":         score_result["status"],
            "breakdown_skor": breakdown_summary,
            "flag_aktif":     active_flags,
            "entitas_sms": {
                "nomor_telepon": sms_entities.get("phone_numbers", []),
                "otp_pin": sms_entities.get("otps", []),
                "nominal_uang": sms_entities.get("amounts", []),
                "kata_kunci_darurat": sms_entities.get("urgency_keywords", []),
            },
            "domain_reputasi": {
                "umur_hari":  domain_info.get("domain_age", "-"),
                "registrar":  domain_info.get("registrar", "-"),
                "isp":        domain_info.get("isp", "-"),
                "country":    domain_info.get("country", "-"),
                "virustotal": domain_info.get("virustotal_positives", 0),
            },
            "url_berisiko": [
                {
                    "url":                u.get("url", ""),
                    "risiko":             u.get("risk_level", "-"),
                    "ekstensi_berbahaya": u.get("dangerous_extension", "-"),
                    "brand_ditiru":       u.get("impersonated_brand", "-"),
                }
                for u in url_analysis if u.get("risk_level") in ("TINGGI", "SEDANG")
            ][:5],
        }

        prompt = f"""
Anda adalah Analis Keamanan Siber senior dalam sistem riset berjudul:
"Deteksi Tingkat Risiko SMS Penipuan / Phishing".

Data forensik SMS berikut SUDAH dievaluasi otomatis.
KONTEKS HASIL SCORING:
{json.dumps(context, ensure_ascii=False, indent=2)}

TUGAS ANDA:
Berdasarkan data di atas, hasilkan HANYA narasi penjelasan berformat JSON murni. JANGAN markdown.
ATURAN OUTPUT:
1. "saran": Satu paragraf ringkas (maks 60 kata) untuk orang awam. Berikan saran praktis jika teks ini penipuan.
2. "rincian": Array 3 string kalimat analitis (Konten/Kata Kunci SMS, URL Target, dan Risiko Total AI).

{{
  "saran": "...",
  "rincian": ["...", "...", "..."]
}}
"""
        response = model.generate_content(prompt)
        text = response.text.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0].strip()

        parsed = json.loads(text)
        return {
            "saran":   parsed.get("saran",   "Tidak ada saran."),
            "rincian": parsed.get("rincian", []),
        }

    except json.JSONDecodeError:
        return _fallback_narrative("Format respons AI tidak valid (bukan JSON).")
    except Exception as e:
        return _fallback_narrative(f"Gagal menghubungi AI: {str(e)[:160]}")

def _fallback_narrative(pesan_error: str) -> dict:
    return {
        "saran": f"[Mode Offline] {pesan_error} — Gunakan data teknis untuk analisis manual.",
        "rincian": [
            "Sistem AI sedang offline — periksa kata kunci manipulatif pada teks.",
            "Periksa tabel URL Analyzer untuk temuan URL mencurigakan.",
            "Periksa apakah ada OTP atau nomor asing.",
        ],
    }