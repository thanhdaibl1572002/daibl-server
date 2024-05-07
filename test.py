import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('data.csv')

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 537667 entries, 0 to 537666
# Data columns (total 20 columns):
#  #   Column               Non-Null Count   Dtype  
# ---  ------               --------------   -----  
#  0   ID                   537667 non-null  int64  
#  1   CODE_GENDER          537667 non-null  object 
#  2   FLAG_OWN_CAR         537667 non-null  object 
#  3   FLAG_OWN_REALTY      537667 non-null  object 
#  4   CNT_CHILDREN         537667 non-null  int64  
#  5   AMT_INCOME_TOTAL     537667 non-null  float64
#  6   NAME_INCOME_TYPE     537667 non-null  object 
#  7   NAME_EDUCATION_TYPE  537667 non-null  object 
#  8   NAME_FAMILY_STATUS   537667 non-null  object 
#  9   NAME_HOUSING_TYPE    537667 non-null  object 
#  10  DAYS_BIRTH           537667 non-null  int64  
#  11  DAYS_EMPLOYED        537667 non-null  int64  
#  12  FLAG_MOBIL           537667 non-null  int64  
#  13  FLAG_WORK_PHONE      537667 non-null  int64  
#  14  FLAG_PHONE           537667 non-null  int64  
#  15  FLAG_EMAIL           537667 non-null  int64  
#  16  OCCUPATION_TYPE      537667 non-null  object 
#  17  CNT_FAM_MEMBERS      537667 non-null  float64
#  18  MONTHS_BALANCE       537667 non-null  int64  
#  19  STATUS               537667 non-null  int64  
# dtypes: float64(2), int64(10), object(8)
# memory usage: 82.0+ MB

df.drop(columns=['ID', 'FLAG_MOBIL'], inplace = True)

scaler = MinMaxScaler()
df[['AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH']] = scaler.fit_transform(df[['AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_BIRTH']])
with open('min_max_scaler.pkl', 'wb') as f:
  pickle.dump(scaler, f)

columns_to_encode = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
one_hot_encoder = OneHotEncoder(drop=None, sparse_output=False)
encoded_data = one_hot_encoder.fit_transform(df[columns_to_encode])
df_encoded = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(columns_to_encode))
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'])
with open('one_hot_encoder.pkl', 'wb') as f:
  pickle.dump(one_hot_encoder, f)

label_encoder_income_type = LabelEncoder()
label_encoder_education_type = LabelEncoder()
label_encoder_family_status = LabelEncoder()
label_encoder_housing_type = LabelEncoder()
label_encoder_occupation_type = LabelEncoder()
df['NAME_INCOME_TYPE'] = label_encoder_income_type.fit_transform(df['NAME_INCOME_TYPE'])
df['NAME_EDUCATION_TYPE'] = label_encoder_education_type.fit_transform(df['NAME_EDUCATION_TYPE'])
df['NAME_FAMILY_STATUS'] = label_encoder_family_status.fit_transform(df['NAME_FAMILY_STATUS'])
df['NAME_HOUSING_TYPE'] = label_encoder_housing_type.fit_transform(df['NAME_HOUSING_TYPE'])
df['OCCUPATION_TYPE'] = label_encoder_occupation_type.fit_transform(df['OCCUPATION_TYPE'])
with open('label_encoder_income_type.pkl', 'wb') as f:
  pickle.dump(label_encoder_income_type, f)
with open('label_encoder_education_type.pkl', 'wb') as f:
  pickle.dump(label_encoder_education_type, f)
with open('label_encoder_family_status.pkl', 'wb') as f:
  pickle.dump(label_encoder_family_status, f)
with open('label_encoder_housing_type.pkl', 'wb') as f:
  pickle.dump(label_encoder_housing_type, f)
with open('label_encoder_occupation_type.pkl', 'wb') as f:
  pickle.dump(label_encoder_occupation_type, f)
  
X = df.drop(['STATUS'], axis=1)
y = df['STATUS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest_model = RandomForestClassifier(random_state = 0)
random_forest_model.fit(X_train, y_train)
with open('random_forest_model.pkl', 'wb') as f:
  pickle.dump(random_forest_model, f)