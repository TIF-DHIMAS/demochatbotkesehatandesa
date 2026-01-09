from flask import Flask, render_template, request, jsonify, send_file
import torch
import pandas as pd
import joblib
import tempfile, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from io import BytesIO

app = Flask(__name__)

# =====================
# LOAD DATASET
# =====================
df = pd.read_csv("health_qa2.csv")
questions = df["question"].astype(str).tolist()
answers = df["answer"].astype(str).tolist()
intents = df["intent"].astype(str).tolist()

# =====================
# LOAD INTENT MODEL
# =====================
MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label_encoder = joblib.load("label_encoder.pkl")
num_labels = len(label_encoder.classes_)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)
model.load_state_dict(
    torch.load("indobert_finetuned_intent.pt", map_location="cpu")
)
model.eval()

# =====================
# TF-IDF
# =====================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# =====================
# LOAD WHISPER (TINY)
# =====================
whisper_model = whisper.load_model("tiny")

# =====================
# FUNCTIONS
# =====================
def predict_intent(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    conf, idx = torch.max(probs, dim=1)

    intent = label_encoder.inverse_transform([idx.item()])[0]
    confidence = round(conf.item() * 100, 2)

    return intent, confidence


def get_answer(user_text, intent):
    idxs = [i for i, v in enumerate(intents) if v == intent]
    if not idxs:
        return "Maaf, saya belum punya informasi yang sesuai."

    user_vec = vectorizer.transform([user_text])
    sims = cosine_similarity(user_vec, X[idxs])[0]

    best_idx = sims.argmax()
    score = sims[best_idx]

    if score < 0.3:
        return "Maaf, saya belum yakin dengan jawabannya. Silakan tanyakan dengan kalimat lain."

    return answers[idxs[best_idx]]

# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat_api", methods=["POST"])
def chat_api():
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "Silakan masukkan pesan."})

    intent, confidence = predict_intent(user_msg)
    answer = get_answer(user_msg, intent)

    return jsonify({
        "reply": answer,
        "intent": intent,
        "confidence": confidence
    })


@app.route("/voice", methods=["POST"])
def voice():
    text = request.json.get("text", "")
    tts = gTTS(text, lang="id")

    audio = BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)

    return send_file(audio, mimetype="audio/mpeg")


@app.route("/stt", methods=["POST"])
def stt():
    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        path = tmp.name

    result = whisper_model.transcribe(path, language="id")
    os.remove(path)

    return jsonify({"text": result["text"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
