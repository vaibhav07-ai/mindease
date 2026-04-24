import os
import nltk

nltk.data.path.append('/app/nltk_data')
nltk.download('vader_lexicon', download_dir='/app/nltk_data')
nltk.download('punkt', download_dir='/app/nltk_data')
nltk.download('wordnet', download_dir='/app/nltk_data')
nltk.download('punkt_tab', download_dir='/app/nltk_data')

import json
import random
import joblib
import nltk

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

from groq import Groq
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

model_ml = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)

import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CRISIS_WORDS = [
    'suicide', 'kill myself', 'want to die',
    'hurt myself', 'end my life', 'khud ko hurt',
    'marna chahta', 'jeena nahi chahta'
]

def clean(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

def get_reply(user_message):

    # STEP 1 — Crisis check sabse pehle
    for word in CRISIS_WORDS:
        if word in user_message.lower():
            return ("Mujhe aapki bahut chinta ho rahi hai. "
                    "Kripya abhi iCall ko call karein: 9152987821. "
                    "Aap akele nahi hain, madad available hai.")

    # STEP 2 — Sentiment check
    score = sia.polarity_scores(user_message)
    if score['compound'] <= -0.5:
        mood = "negative"
    elif score['compound'] >= 0.5:
        mood = "positive"
    else:
        mood = "neutral"

    # STEP 3 — Apna ML model try karo
    cleaned = clean(user_message)
    predicted_number = model_ml.predict([cleaned])[0]
    intent_tag = encoder.inverse_transform([predicted_number])[0]
    confidence = max(model_ml.predict_proba([cleaned])[0])

    # STEP 4 — ML confident hai toh ML ka jawab do
    if confidence >= 0.55:
        for intent in data['intents']:
            if intent['tag'] == intent_tag:
                reply = random.choice(intent['responses'])
                if mood == "negative" and intent_tag not in ['crisis', 'depression']:
                    reply = "I can sense things feel hard right now. " + reply
                return reply

    # STEP 5 — Groq AI se poocho
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are MindEase, a caring desi friend who listens and supports.
- Reply in Hinglish if user writes Hindi/Hinglish, English if they write English
- Be like a close yaar — warm, casual, NOT formal or robotic
- Keep replies short — max 2-3 sentences
- Validate feelings naturally, like a friend would
- NEVER mention any helpline number unless user explicitly says they want to hurt themselves
- Never diagnose or give medical advice
- No repetitive phrases, every reply should feel fresh"""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=200
        )
        return response.choices[0].message.content

    except Exception as e:
        print("Groq error:", e)
        return "Yaar main sun raha hun. Thoda aur batao kya ho raha hai?"