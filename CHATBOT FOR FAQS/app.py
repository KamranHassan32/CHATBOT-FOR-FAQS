from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
import nltk
import os
import re

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - depends on package availability
    OpenAI = None

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

BASE_DIR = Path(__file__).resolve().parent
FAQS_PATH = BASE_DIR / "faq_data.json"


def load_faqs():
    if FAQS_PATH.exists():
        with FAQS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
    return [
        {
            "question": "What is the return policy?",
            "answer": "You can return most items within 30 days of delivery if they are unused and in original packaging."
        }
    ]


FAQS = load_faqs()

client = None
if OpenAI is not None:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)


def preprocess_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    return " ".join(tokens)


processed_questions = [preprocess_text(item["question"]) for item in FAQS]
vectorizer = TfidfVectorizer()
faq_matrix = vectorizer.fit_transform(processed_questions)


def find_best_match(user_question: str):
    processed_query = preprocess_text(user_question)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, faq_matrix).flatten()
    best_index = int(similarities.argmax())
    score = float(similarities[best_index])

    if score < 0.2:
        return {
            "answer": "I could not find a close match. Please try a question like shipping, returns, payments, or support.",
            "matched_question": None,
            "score": score,
        }

    return {
        "answer": FAQS[best_index]["answer"],
        "matched_question": FAQS[best_index]["question"],
        "score": score,
    }


def get_chatgpt_reply(user_message: str):
    if client is None:
        return None

    faq_context = "\n".join(
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in FAQS
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful support assistant for a company FAQ. Use the provided FAQ context when relevant and answer clearly.",
                },
                {
                    "role": "user",
                    "content": f"Here is the FAQ knowledge base:\n{faq_context}\n\nUser question: {user_message}\nAnswer the question directly.",
                },
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()

    if not user_message:
        return jsonify({"reply": "Please ask a question about our product or service."})

    result = find_best_match(user_message)
    reply = result["answer"]
    matched_question = result["matched_question"]
    score = round(result["score"], 3)

    if result["matched_question"] is None and result["score"] < 0.2:
        gpt_reply = get_chatgpt_reply(user_message)
        if gpt_reply:
            reply = gpt_reply
            matched_question = None
            score = 0.0

    return jsonify({
        "reply": reply,
        "matched_question": matched_question,
        "score": score,
    })


if __name__ == "__main__":
    app.run(debug=True)
