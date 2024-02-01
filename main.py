from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import (
    is_vietnamese,
    predict_sentiment_svm,
    predict_sentiment_phobert,
)

app = Flask(__name__)
CORS(app)

@app.route("/predict/svm", methods=["POST"])
def predict_svm():
    try:
        comment = request.json.get("comment", "")
        if is_vietnamese(comment):
            return jsonify(predict_sentiment_svm(comment))
        else:
            return jsonify(-2)

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict/phobert", methods=["POST"])
def predict_phobert():
    try:
        comment = request.json.get("comment", "")
        if is_vietnamese(comment):
            return jsonify(predict_sentiment_phobert(comment))
        else:
            return jsonify(-2)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
