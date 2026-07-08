import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Logistic_Regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns =joblib.load('columns.pkl')

st.title("Heart Disease Prediction Model ")
st.markdown("Provide the following details")

age = st.slider("age",18,100,40)
sex = st.selectbox("sex",['Male','Female'])
chest_pain = st.selectbox("cheest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting blood pressure (mm Hg)",80,200,120)
cholesterol = st.number_input("Serum cholesterol (mg/dl)",100,600,200)
fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl",[0,1])
resting_ecg = st.selectbox("Resting electrocardiographic results",["Normal","ST","LVH"])
max_hr = st.slider("Max heart rate",60,220,150)
exercise_angina = st.selectbox("Exercise induced angina",['Yes','No'])
oldpeak = st.slider("Oldpeak (ST depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST slope",["Up","Flat","Down"])

if st.button("Predict"):
    raw_input = {
        'age': age,
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_bs': fasting_bs,
        'max_hr': max_hr,
        'oldpeak': oldpeak,
        'sex' + sex: 1,
        'chest_pain' + chest_pain: 1,
        'resting_ecg' + resting_ecg: 1,
        'exercise_angina' + exercise_angina: 1,
        'st_slope' + st_slope: 1
    }
     
    input_df = pd.DataFrame([raw_input])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error("High risk of heart disease")
    else:
        st.success("Low risk of heart disease")