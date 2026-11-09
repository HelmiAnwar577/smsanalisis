import sys
import joblib

MODEL_PATH = "analyzer/model_nb.pkl"
VEC_PATH = "analyzer/vectorizer.pkl"

nb_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

sms = "Selamat! Nomor Anda terpilih memenangkan hadiah Rp 100 Juta dari Bank BNI. Klik bit.ly/hadiah untuk klaim."
X_input = vectorizer.transform([sms])
prob = nb_model.predict_proba(X_input)[0][1]
print(f"Probabilitas Phishing (Email Model): {prob}")
