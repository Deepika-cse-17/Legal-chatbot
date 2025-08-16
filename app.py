# app.py
# A minimal legal information chatbot (NOT legal advice)
# Run: 1) pip install flask scikit-learn
#      2) python app.py
# Open http://127.0.0.1:5000

from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import html

app = Flask(__name__)

# -----------------------------
# Knowledge base (sample)
# Replace/extend with your own jurisdiction-specific content.
# Keep answers informational & general.
# -----------------------------
KB = [
    {
        "q": "What is a contract?",
        "a": "A contract is an agreement between two or more parties that is intended to be legally binding. Typically it requires offer, acceptance, consideration, and intention to create legal relations."
    },
    {
        "q": "When is a contract enforceable?",
        "a": "In general, a contract is enforceable when the essential elements are present (offer, acceptance, consideration, capacity, lawful purpose) and any required formalities are met (e.g., writing when mandated by law)."
    },
    {
        "q": "What is consideration in a contract?",
        "a": "Consideration is something of value exchanged between parties (e.g., money, goods, services, a promise) that supports a contract."
    },
    {
        "q": "What is negligence?",
        "a": "Negligence is a failure to exercise reasonable care, resulting in damage or injury to another. A negligence claim typically requires duty, breach, causation, and damages."
    },
    {
        "q": "What should I do if I am arrested?",
        "a": "Stay calm. Ask if you are free to leave. If not, you can state your right to remain silent and request a lawyer. Do not resist. Laws vary by jurisdiction; contact a licensed attorney."
    },
    {
        "q": "What is GDPR?",
        "a": "The General Data Protection Regulation (GDPR) is an EU law on data protection and privacy. It sets rules on how personal data is processed and gives individuals certain rights."
    },
    {
        "q": "What is intellectual property?",
        "a": "Intellectual property (IP) covers creations of the mind such as inventions, literary and artistic works, designs, symbols, names and images. Main types include patents, trademarks, and copyrights."
    },
    {
        "q": "How do I register a trademark?",
        "a": "Trademark registration steps vary by country, but often include: searching for existing marks, filing an application with the trademarks office, responding to examinations/oppositions, and paying fees."
    },
    {
        "q": "What is a non-disclosure agreement (NDA)?",
        "a": "An NDA is a contract that restricts one or more parties from disclosing confidential information to others. It can be mutual or one-way and should define scope, permitted use, term, and remedies."
    },
    {
        "q": "How do I make a simple will?",
        "a": "Laws vary widely. Generally you identify yourself, revoke prior wills, name an executor, describe how to distribute assets, sign with required witnesses, and follow formalities. Consult a local lawyer."
    }
]

# Preprocess & vectorize KB
docs = [item["q"] + " " + item["a"] for item in KB]
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
kb_matrix = vectorizer.fit_transform(docs)

# -----------------------------
# Helpers
# -----------------------------
EMERGENCY_WORDS = {"suicide", "self harm", "self-harm", "violence", "immediate danger", "emergency", "threat"}

def is_greeting(text: str) -> bool:
    return bool(re.search(r"\b(hi|hello|hey|hola|namaste)\b", text, re.I))

def is_emergency(text: str) -> bool:
    text_l = text.lower()
    return any(w in text_l for w in EMERGENCY_WORDS)

def wants_lawyer(text: str) -> bool:
    return bool(re.search(r"\b(lawyer|attorney|advocate|legal counsel|legal aid)\b", text, re.I))

def best_match(user_text: str):
    """Return (score, KB item) for the most relevant entry."""
    user_vec = vectorizer.transform([user_text])
    sims = cosine_similarity(user_vec, kb_matrix)[0]
    idx = sims.argmax()
    return sims[idx], KB[idx]

def safe_answer(user_text: str):
    # Intent: emergency or crisis
    if is_emergency(user_text):
        return (
            "If you or someone else is in immediate danger, please contact local emergency services right now. "
            "If this is about self-harm or crisis, reach out to your local suicide prevention or mental health helpline. "
            "You are not alone, and help is available."
        )

    # Greetings
    if is_greeting(user_text):
        return "Hello! I’m a legal information bot. Ask me general questions (e.g., contracts, IP, privacy). I can’t provide legal advice."

    # Lawyer intent
    if wants_lawyer(user_text):
        return (
            "I can only provide general legal information. For advice on your specific situation, consider consulting a licensed lawyer in your jurisdiction "
            "(e.g., your local bar association’s referral service or accredited legal aid clinics)."
        )

    # Retrieval over KB
    score, item = best_match(user_text)
    if score < 0.15:
        return ("I’m not confident I have an answer to that. Laws differ by jurisdiction. "
                "Consider contacting a licensed attorney or a local legal aid clinic.")
    return item["a"]

# -----------------------------
# Web UI
# -----------------------------
TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Legal Info Chatbot (Demo)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    body { margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 900px; margin: 0 auto; padding: 24px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 20px; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    .disclaimer { font-size: 12px; color: #94a3b8; margin-bottom: 16px; }
    .chat { height: 60vh; overflow-y: auto; padding: 12px; background: #0b1020; border-radius: 12px; border: 1px solid #1f2937; }
    .msg { margin: 10px 0; max-width: 80%; padding: 12px 14px; border-radius: 12px; line-height: 1.4; }
    .user { background: #1d4ed8; color: white; margin-left: auto; border-top-right-radius: 4px; }
    .bot { background: #111827; color: #e5e7eb; margin-right: auto; border-top-left-radius: 4px; border: 1px solid #1f2937; }
    form { display: flex; gap: 8px; margin-top: 12px; }
    input[type=text] { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #334155; background: #0b1020; color: #e2e8f0; }
    button { padding: 12px 16px; border: 0; border-radius: 10px; background: #22c55e; color: #052e16; font-weight: 600; cursor: pointer; }
    .footer { margin-top: 10px; font-size: 11px; color: #94a3b8; }
    .kb { font-size: 12px; color: #cbd5e1; margin-top: 16px; }
    .kb summary { cursor: pointer; }
    .small { font-size: 12px; color: #9ca3af; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>⚖️ Legal Information Chatbot (Demo)</h1>
      <div class="disclaimer">
        This tool provides general legal information for educational purposes only and is <b>not legal advice</b>.
        Laws vary by jurisdiction. For advice on your situation, consult a licensed attorney.
      </div>

      <div id="chat" class="chat">
        <div class="msg bot">Hello! Ask me general legal questions like “What is negligence?” or “How do I register a trademark?”</div>
      </div>

      <form id="form">
        <input id="input" type="text" placeholder="Type your question..." autocomplete="off" required />
        <button type="submit">Send</button>
      </form>

      <div class="footer">
        Tip: Type “lawyer” to get referral guidance. Type “help” or “emergency” if you are in danger to see crisis info.
      </div>

      <details class="kb">
        <summary>Show sample topics I know</summary>
        <ul>
          {% for item in kb %}
            <li><span class="small">{{ item.q }}</span></li>
          {% endfor %}
        </ul>
      </details>
    </div>
  </div>

  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const input = document.getElementById('input');

    function addMessage(text, who) {
      const div = document.createElement('div');
      div.className = 'msg ' + (who === 'user' ? 'user' : 'bot');
      div.innerText = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      addMessage(text, 'user');
      input.value = '';

      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ question: text })
        });
        const data = await res.json();
        addMessage(data.answer, 'bot');
      } catch (err) {
        addMessage('Sorry, something went wrong. Please try again.', 'bot');
      }
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE, kb=KB)

@app.route("/chat")
def chat():
    return render_template_string("chat.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True, silent=True) or {}
    question = payload.get("question", "").strip()
    question = html.unescape(question)
    if not question:
        return jsonify({"answer": "Please type a question."})

    answer = safe_answer(question)
    return jsonify({"answer": answer})

from pyngrok import ngrok

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print("🌍 Public URL:", public_url)

    # Run Flask app
    app.run(port=5000)

