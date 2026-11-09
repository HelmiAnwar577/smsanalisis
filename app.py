"""
app.py — SMS Defender v5.0
==========================
Aplikasi web deteksi smishing: Naive Bayes Classifier + skoring risiko + penjelasan berbasis Generative AI.

Jalankan:
    python app.py                      # mode lengkap (butuh internet)
    set SMS_DEFENDER_OFFLINE=1         # Windows CMD  -> mode offline
    $env:SMS_DEFENDER_OFFLINE="1"      # PowerShell   -> mode offline
    python app.py

Mode offline mematikan reputasi domain (WHOIS, geolokasi, VirusTotal) dan
pelacakan redirect, sama seperti saat evaluasi di artikel. Berguna untuk demo
bila koneksi internet ruangan tidak stabil. Narasi Gemini tetap dicoba; bila
gagal, sistem menampilkan narasi cadangan.

Perubahan dari v4:
- Pengambilan tautan memakai analyzer/url_extractor.py (sama dengan evaluasi).
- Bila SMS memuat beberapa tautan, dipilih reputasi domain dengan skor S_C
  tertinggi (sebelumnya: VirusTotal terbanyak / domain termuda).
- Mode offline lewat variabel lingkungan SMS_DEFENDER_OFFLINE.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from analyzer.domain_reputation import get_domain_info  # noqa: E402
from analyzer.llm_evaluator import generate_narrative  # noqa: E402
from analyzer.redirect_trace import trace_redirect  # noqa: E402
from analyzer.scoring_engine import calculate_score, domain_risk_score  # noqa: E402
from analyzer.sms_analyzer import extract_sms_entities  # noqa: E402
from analyzer.url_analyzer import analyze_urls  # noqa: E402
from analyzer.url_extractor import extract_urls  # noqa: E402

# Naive Bayes (opsional — graceful fallback)
try:
    from analyzer.text_classifier import predict_text_phishing
    _NB_OK = True
except Exception:
    _NB_OK = False

    def predict_text_phishing(_text):
        return 0.0

app = Flask(__name__)

VERSION = "6.1"
MAX_PREVIEW = 3000
OFFLINE = os.environ.get("SMS_DEFENDER_OFFLINE", "").strip().lower() in ("1", "true", "ya", "yes")


def build_sms_preview(sms_text: str, source: str = "raw") -> dict:
    return {
        "source": source,
        "body_snippet": sms_text[:MAX_PREVIEW],
        "full_snippet": sms_text[:MAX_PREVIEW],
    }


def _fetch_domain_info_all(urls: list) -> dict:
    """Cek reputasi setiap domain secara paralel, kembalikan yang paling berisiko."""
    if not urls:
        return {}

    results = []
    with ThreadPoolExecutor(max_workers=min(len(urls), 5)) as ex:
        futures = {ex.submit(get_domain_info, url): url for url in urls}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                pass

    if not results:
        return {}
    return max(results, key=domain_risk_score)


def analyze_sms(sms_text: str) -> dict:
    """Seluruh alur analisis satu SMS (dipakai route /analyze)."""
    sms_entities = extract_sms_entities(sms_text)
    urls = extract_urls(sms_text)
    url_analysis = analyze_urls(urls)

    if urls and not OFFLINE:
        redirect_chain = trace_redirect(urls[0])
        domain_info = _fetch_domain_info_all(urls)
    else:
        redirect_chain, domain_info = [], {}

    phishing_probability = predict_text_phishing(sms_text) if _NB_OK else 0.0

    score_result = calculate_score(
        url_analysis=url_analysis,
        domain_info=domain_info,
        redirect_chain=redirect_chain,
        sms_entities=sms_entities,
        phishing_probability=phishing_probability,
    )
    return {
        "sms_entities": sms_entities,
        "urls": urls,
        "url_analysis": url_analysis,
        "redirect_chain": redirect_chain,
        "domain_info": domain_info,
        "phishing_probability": phishing_probability,
        "score_result": score_result,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", version=VERSION)


@app.route("/analyze", methods=["POST"])
def analyze():
    sms_text = request.form.get("sms_text", "")

    if not sms_text.strip():
        return render_template("index.html", version=VERSION, error="Tidak ada teks SMS yang dapat dianalisis."), 400

    a = analyze_sms(sms_text)
    score_result = a["score_result"]

    narrative = generate_narrative(
        score_result=score_result,
        sms_entities=a["sms_entities"],
        url_analysis=a["url_analysis"],
        domain_info=a["domain_info"],
    )

    risk_details = {
        "score": score_result["total_score"],
        "status": score_result["status"],
        "warna": score_result["warna"],
        "saran": narrative["saran"],
        "rincian": narrative["rincian"],
        "breakdown": score_result["breakdown"],
        "all_flags": score_result["all_flags"],
    }

    return render_template(
        "result.html",
        urls=a["urls"],
        url_analysis=a["url_analysis"],
        redirect_chain=a["redirect_chain"],
        domain_info=a["domain_info"],
        sms_entities=a["sms_entities"],
        risk_details=risk_details,
        sms_preview=build_sms_preview(sms_text),
        offline=OFFLINE,
        version=VERSION,
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "naive_bayes": _NB_OK,
        "offline_mode": OFFLINE,
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"Server SMS Defender v{VERSION} berjalan"
          f"{' (MODE OFFLINE: reputasi domain & redirect dimatikan)' if OFFLINE else ''}")
    print("Buka: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True)
