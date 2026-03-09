from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot
app = Flask(__name__, template_folder="../templates")
bot = FAQChatbot("../data/faqs.json")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data["message"]
    response = bot.get_response(user_message)
    return jsonify({"response": response})
if __name__ == "__main__":
    app.run(debug=True)