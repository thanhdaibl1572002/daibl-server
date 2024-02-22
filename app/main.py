from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import ( is_vietnamese, predict_sentiment_svm )

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict_svm():
    try:
        comment = request.json.get("comment", "")
        if is_vietnamese(comment):
            return jsonify(predict_sentiment_svm(comment))
        else:
            return jsonify(-2)

    except Exception as e:
        return jsonify({"error": str(e)})
