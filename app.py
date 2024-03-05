import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import matplotlib as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import pickle

from src.utils import load_object

from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


#TITLE
st.title('Stroke Risk Prediction App')

# Input from the user 
gender = st.text_input('Enter your gender: ')
age = st.number_input('Enter your age: ')
hypertension = st.number_input('Hypertension (1 for Yes, 0 for No)')
heart_disease = st.number_input('Heart Disease (1 for Yes, 0 for No)')
ever_married = st.text_input('Ever Married (Yes or No)')
work_type = st.text_input('Work Type (children, Self-employed, Govt_job, Never_worked, Private)')
Residence_type = st.text_input('Residence Type (Urban or Rural)')
avg_glucose_level = st.number_input('Average Glucose level')
bmi = st.number_input('BMI(Body Mass Index)')
smoking_status = st.text_input('Smoking Status (never smoked, formerly smoked, smokes)')

# giving the data to the predict pipeline for prediction
data = CustomData(gender=gender, age=age, hypertension=hypertension, heart_disease=heart_disease,
                        ever_married=ever_married, work_type=work_type, Residence_type=Residence_type, avg_glucose_level=avg_glucose_level,
                        bmi=bmi, smoking_status=smoking_status)

# Predict the output
if st.button('Predict'):
    pred_df = data.get_data_as_data_frame()
        
    predict_pipeline = PredictPipeline()

    results = predict_pipeline.predict(pred_df)

    def prediction(results):
        if results == 0:
            return 'You are risk free of getting a Stroke'
        if results == 1:
            return 'You have risk of getting a Stroke'

    st.title(str(prediction(results)))
