import re
import string
import pickle
import pandas as pd
from unidecode import unidecode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

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


def predict_sentiment(text):
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


def predict_credit(code_gender, flag_own_car, flag_own_realty, cnt_children, amt_income_total, name_income_type, name_education_type, name_family_status, name_housing_type, days_birth, days_employed, flag_work_phone, flag_phone, flag_email, occupation_type, cnt_fam_members, months_balance):
    with open("random_forest_model.pkl", "rb") as f:
        random_forest_model = pickle.load(f)

    input_data = pd.DataFrame({
        "CODE_GENDER": [code_gender],
        "FLAG_OWN_CAR": [flag_own_car],
        "FLAG_OWN_REALTY": [flag_own_realty],
        "CNT_CHILDREN": [cnt_children],
        "AMT_INCOME_TOTAL": [amt_income_total],
        "NAME_INCOME_TYPE": [name_income_type],
        "NAME_EDUCATION_TYPE": [name_education_type],
        "NAME_FAMILY_STATUS": [name_family_status],
        "NAME_HOUSING_TYPE": [name_housing_type],
        "DAYS_BIRTH": [days_birth],
        "DAYS_EMPLOYED": [days_employed],
        "FLAG_WORK_PHONE": [flag_work_phone],
        "FLAG_PHONE": [flag_phone],
        "FLAG_EMAIL": [flag_email],
        "OCCUPATION_TYPE": [occupation_type],
        "CNT_FAM_MEMBERS": [cnt_fam_members],
        "MONTHS_BALANCE": [months_balance]
    })

    scaler = MinMaxScaler()
    input_data[["AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "DAYS_BIRTH"]] = scaler.fit_transform(input_data[["AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "DAYS_BIRTH"]])

    columns_to_encode = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
    one_hot_encoder = OneHotEncoder(drop=None, sparse=False)
    encoded_data = one_hot_encoder.fit_transform(input_data[columns_to_encode])
    input_data_encoded = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(columns_to_encode))
    input_data = pd.concat([input_data, input_data_encoded], axis=1)
    input_data = input_data.drop(columns=columns_to_encode)

    label_encoder = LabelEncoder()
    for column in ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"]:
        input_data[column] = label_encoder.fit_transform(input_data[column])

    prediction = random_forest_model.predict(input_data)

    return prediction[0]