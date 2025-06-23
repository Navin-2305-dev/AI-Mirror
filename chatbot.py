from flask import Flask, redirect, request, jsonify, render_template, url_for
from flask_cors import CORS
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
import markdown


import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemma-3n-e4b-it")

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

def log_emotion(user_input, response):
    with open("emotion_log.txt", "a") as f:
        f.write(f"{datetime.now()} | USER: {user_input}\nBOT: {response}\n\n")

def analyze_mood_log():
    mood_counts = {}
    mood_keywords = {
        "happy": "Happy",
        "joy": "Happy",
        "sad": "Sad",
        "depressed": "Sad",
        "angry": "Angry",
        "mad": "Angry",
        "anxious": "Anxious",
        "stressed": "Anxious",
        "excited": "Excited",
        "overwhelmed": "Overwhelmed",
        "calm": "Calm",
        "lonely": "Lonely"
    }

    try:
        with open("emotion_log.txt", "r") as f:
            content = f.read().lower()

        for keyword, label in mood_keywords.items():
            count = content.count(keyword)
            if count > 0:
                mood_counts[label] = mood_counts.get(label, 0) + count

    except FileNotFoundError:
        mood_counts = {}

    return mood_counts

def generate_mood_chart(mood_data):
    if not mood_data:
        return None

    labels = list(mood_data.keys())
    values = list(mood_data.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color='skyblue')
    plt.title("Mood Distribution Over Time")
    plt.xlabel("Mood")
    plt.ylabel("Frequency")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    chart_data = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()

    return chart_data

@app.route("/mood")
def mood_chart():
    mood_data = analyze_mood_log()
    chart = generate_mood_chart(mood_data)
    return render_template("mood.html", chart=chart, mood_data=mood_data)
        
# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    prompt = (
        "You are a compassionate emotional support chatbot designed to listen and respond supportively. "
        "Your role is to analyze the user's emotional tone from their messages and offer kind, respectful, and helpful responses. "
        "Always maintain a professional, warm, and empathetic tone. "
        "Avoid using overly personal or casual terms like 'honey', 'sweetie', or 'dear'. "
        "Do not give medical advice. Do not pretend to be human. "
        "Use inclusive, neutral language. "
        "\n\nUser message:\n"
        f"{user_input}"
    )

    try:
        response = model.generate_content(prompt)
        reply = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text"):
                reply += part.text
        
        log_emotion(user_input, reply)
        reply_html = markdown.markdown(reply)

        return jsonify({"reply": reply_html})
    
    except Exception as e:
        return jsonify({"reply": f"❌ An error occurred: {e}"})

if __name__ == "__main__":
    app.run(debug=True)
