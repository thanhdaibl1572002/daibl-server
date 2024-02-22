import re
import string
import pickle
from unidecode import unidecode

clear_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u200d"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\u3030"
    "\ufe0f"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text):
    text = re.sub(clear_pattern, " ", text)
    text = re.sub(r"([a-z]+?)\1+", r"\1", text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])\s*(\w)", r"\1 \2 \3", text)
    text = re.sub(r"(\w)\s*([" + string.punctuation + "])", r"\1 \2", text)
    text = re.sub(f"([{string.punctuation}])([{string.punctuation}])+", r"\1", text)
    text = text.strip()
    while text.endswith(tuple(string.punctuation + string.whitespace)):
        text = text[:-1]
    while text.startswith(tuple(string.punctuation + string.whitespace)):
        text = text[1:]
    text = re.sub(r"\s+", " ", text)
    return text


def remove_digits(text):
    return "".join([char for char in str(text) if not char.isdigit()])


def remove_special_characters(text):
    text = re.sub(
        r"[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựíìỉĩịýỳỷỹỵđĐ ]",
        "",
        text,
    )
    return text


def remove_vietnamese_accent(text):
    text = re.sub(r"[áàảãạăắằẳẵặâấầẩẫậ]", "a", text)
    text = re.sub(r"[éèẻẽẹêếềểễệ]", "e", text)
    text = re.sub(r"[óòỏõọôốồổỗộơớờởỡợ]", "o", text)
    text = re.sub(r"[íìỉĩị]", "i", text)
    text = re.sub(r"[úùủũụưứừửữự]", "u", text)
    text = re.sub(r"[ýỳỷỹỵ]", "y", text)
    text = re.sub(r"[đ]", "d", text)
    return text


def is_vietnamese(text):
    return unidecode(text) != text


def predict_sentiment_svm(text):
    text = text.lower()
    text = clean_text(text)
    text = remove_digits(text)
    text = remove_special_characters(text)
    text = remove_vietnamese_accent(text)
    text = text.strip()
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction.tolist()[0]