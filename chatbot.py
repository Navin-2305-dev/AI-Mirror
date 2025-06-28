from flask import Flask, request, jsonify, render_template, url_for, session
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
import re

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemma-3n-e4b-it")

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

@app.route("/")
def home():
    if 'emotion_log' not in session:
        session['emotion_log'] = []
    if 'chat_messages' not in session:
        session['chat_messages'] = []
    return render_template("index.html", chat_messages=session.get('chat_messages', []))

def log_emotion(user_input, response):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_input': user_input,
        'bot_response': response
    }
    session['emotion_log'].append(log_entry)
    session.modified = True

def log_chat_message(sender, message):
    chat_entry = {
        'sender': sender,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    session['chat_messages'].append(chat_entry)
    session.modified = True

def analyze_mood_log():
    mood_counts = {}
    mood_instances = {}
    mood_keywords = {
        r"\bhappy\b": "Happy",
        r"\bjoy\b": "Happy",
        r"\bsad\b": "Sad",
        r"\bdepressed\b": "Sad",
        r"\bangry\b": "Angry",
        r"\bmad\b": "Angry",
        r"\banxious\b": "Anxious",
        r"\bstressed\b": "Anxious",
        r"\bexcited\b": "Excited",
        r"\boverwhelmed\b": "Overwhelmed",
        r"\bcalm\b": "Calm",
        r"\blonely\b": "Lonely"
    }
    mood_suggestions = {
        "Happy": "You're radiating positivity! Keep nurturing this by engaging in activities you love, like spending time with friends or pursuing hobbies.",
        "Sad": "It’s okay to feel down sometimes. Try journaling your thoughts, listening to uplifting music, or reaching out to a trusted friend for support.",
        "Angry": "Feeling angry is natural. Consider deep breathing exercises, a short walk, or writing down what’s bothering you to process it calmly.",
        "Anxious": "Anxiety can feel overwhelming. Practice mindfulness techniques like meditation or focus on small, manageable tasks to feel more grounded.",
        "Excited": "Your enthusiasm is infectious! Channel this energy into creative projects or share your excitement with others to amplify the joy.",
        "Overwhelmed": "When things feel too much, take a moment to breathe deeply and prioritize tasks. Breaking things down can make them more manageable.",
        "Calm": "Your calm state is wonderful! Maintain it with relaxation techniques like yoga or a quiet walk in nature.",
        "Lonely": "Feeling lonely can be tough. Reach out to someone you care about, join a community activity, or engage in self-care to feel more connected."
    }

    emotion_log = session.get('emotion_log', [])
    for entry in emotion_log:
        user_input = entry['user_input'].lower()
        timestamp = entry['timestamp']
        for pattern, label in mood_keywords.items():
            if re.search(pattern, user_input):
                mood_counts[label] = mood_counts.get(label, 0) + 1
                if label not in mood_instances:
                    mood_instances[label] = []
                mood_instances[label].append({
                    'timestamp': timestamp,
                    'message': entry['user_input']
                })
                break  # Count only one mood per message

    return mood_counts, mood_instances, mood_suggestions

def generate_mood_chart(mood_data):
    if not mood_data:
        return None

    labels = list(mood_data.keys())
    values = list(mood_data.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['#60a5fa', '#34d399', '#f87171', '#fbbf24', '#a78bfa', '#6ee7b7'])
    plt.title("Mood Distribution", fontsize=14, pad=15)
    plt.xlabel("Mood", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    chart_data = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()

    return chart_data

@app.route("/mood")
def mood_chart():
    mood_data, mood_instances, mood_suggestions = analyze_mood_log()
    return render_template("mood.html", chart=generate_mood_chart(mood_data), mood_data=mood_data, mood_instances=mood_instances, mood_suggestions=mood_suggestions)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Please enter a message."})

    # Build conversation history
    emotion_log = session.get('emotion_log', [])
    conversation_history = "\n\nConversation History:\n"
    for entry in emotion_log:
        conversation_history += f"User: {entry['user_input']}\nBot: {entry['bot_response']}\n"

    prompt = (
        "You are a compassionate emotional support chatbot designed to listen and respond supportively. "
        "Your role is to analyze the user's emotional tone from their messages and offer brief, kind, respectful, and helpful responses. "
        "Always maintain a professional, warm, and empathetic tone. "
        "Avoid using overly personal or casual terms like 'honey', 'sweetie', or 'dear'. "
        "Do not give medical advice. Do not pretend to be human. "
        "Use inclusive, neutral language. "
        "Review the entire conversation history provided below to understand the user's emotional context and respond accordingly. "
        f"{conversation_history}"
        "\n\nCurrent User Message:\n"
        f"{user_input}"
    )

    try:
        response = model.generate_content(prompt)
        reply = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text"):
                reply += part.text
        
        log_emotion(user_input, reply)
        log_chat_message("user", user_input)
        reply_html = markdown.markdown(reply)
        log_chat_message("bot", reply_html)

        return jsonify({"reply": reply_html})
    
    except Exception as e:
        return jsonify({"reply": f"An error occurred: {e}"})

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session['chat_messages'] = []
    session['emotion_log'] = []
    session.modified = True
    return jsonify({"status": "Chat cleared"})

if __name__ == "__main__":
    app.run(debug=True)