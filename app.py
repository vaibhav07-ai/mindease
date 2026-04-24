from flask import Flask, render_template, request, jsonify
from chatbot import get_reply

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'reply': 'Please type something!'})
    reply = get_reply(user_message)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)