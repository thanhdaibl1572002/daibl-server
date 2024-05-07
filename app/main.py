from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import is_vietnamese, predict_sentiment, predict_credit

app = Flask(__name__)
CORS(app)

@app.route('/sentiment', methods=['POST'])
def sentiment():
    try:
        comment = request.json.get('comment', '')
        if is_vietnamese(comment):
            return jsonify(predict_sentiment(comment))
        else:
            return jsonify(-2)

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/credit', methods=['POST'])
def credit():
    try:
        code_gender = request.json.get('CODE_GENDER')
        flag_own_car = request.json.get('FLAG_OWN_CAR')
        flag_own_realty = request.json.get('FLAG_OWN_REALTY')
        cnt_children = request.json.get('CNT_CHILDREN')
        amt_income_total = request.json.get('AMT_INCOME_TOTAL')
        name_income_type = request.json.get('NAME_INCOME_TYPE')
        name_education_type = request.json.get('NAME_EDUCATION_TYPE')
        name_family_status = request.json.get('NAME_FAMILY_STATUS')
        name_housing_type = request.json.get('NAME_HOUSING_TYPE')
        days_birth = request.json.get('DAYS_BIRTH')
        days_employed = request.json.get('DAYS_EMPLOYED')
        flag_work_phone = request.json.get('FLAG_WORK_PHONE')
        flag_phone = request.json.get('FLAG_PHONE')
        flag_email = request.json.get('FLAG_EMAIL')
        occupation_type = request.json.get('OCCUPATION_TYPE')
        cnt_fam_members = request.json.get('CNT_FAM_MEMBERS')
        months_balance = request.json.get('MONTHS_BALANCE')

        prediction = predict_credit(code_gender, flag_own_car, flag_own_realty, cnt_children, amt_income_total, name_income_type, name_education_type, name_family_status, name_housing_type, days_birth, days_employed, flag_work_phone, flag_phone, flag_email, occupation_type, cnt_fam_members, months_balance)
        
        return jsonify(prediction) 
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)