import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import matplotlib as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from src.utils import load_object

from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Dataset
df = pd.read_csv(r'notebook\data\stroke.csv')
df = df[df['gender'] != 'Other']

# User menu
st.sidebar.title('Stroke Risk Prediction App')
user_menu = st.sidebar.radio('Select an Option',
                             ('Overview', 'Dataset', 'EDA', 'App'))

if user_menu == 'Overview':
    pass

if user_menu == 'Dataset':
    # DATASET
    st.title('Dataset')
    st.dataframe(df)

    # df.shape
    st.header('Dataset Infomation')
    st.code('The dataset consists of 5109 rows and 11 columns')

    # DESCRIPTION OF THE DATA
    st.header('Description of the data')
    st.table(df.describe())

if user_menu == 'EDA':
    # TITLE 
    st.title('Explorartory Data Analysis (EDA)')

    # VARIABLES DECLARATION
    age = df['age'].nunique()
    hypertension = df['hypertension'].nunique()
    heart_disease = df['heart_disease'].nunique()
    avg_glucose_level = df['avg_glucose_level'].nunique()
    bmi = df['bmi'].nunique()
    stroke = df['stroke'].nunique()
    gender = df['gender'].nunique()
    ever_married = df['ever_married'].nunique()
    work_type = df['work_type'].nunique()
    residence = df['Residence_type'].nunique()
    smoking = df['smoking_status'].nunique()

    st.header('Question 1')
    st.write('What are the unique values in each features in the data?')
    st.text('----------------------------------------------------------------------------')

    # NUMERICAL FEATURES
    st.title("Numerical Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Age")
        st.title(age)
    with col2:
        st.header("Hypertension")
        st.title(hypertension)
    with col3:
        st.header("Heart Disease")
        st.title(heart_disease)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("BMI")
        st.title(bmi)
    with col2:
        st.header("Stroke")
        st.title(stroke)
    with col3:
        st.header("Glucose Level")
        st.title(avg_glucose_level)
    
    st.text('----------------------------------------------------------------------------')

    # CATEGORICAL FEATURES
    st.title("Categorical Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Gender")
        st.title(gender)
    with col2:
        st.header("Married")
        st.title(ever_married)
    with col3:
        st.header("Work Type")
        st.title(work_type)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Residence Type")
        st.title(residence)
    with col2:
        st.header("Smoking Status")
        st.title(smoking)

    st.text('----------------------------------------------------------------------------')
    st.header('Question 2')
    st.write('What are the value counts of each features in the data?')
    
    # NUMERICAL COUNTPLOT VISUALISATION
    st.title('Categorical Features Visualisation')

    categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
    countplot_name = st.selectbox('Features for Value Counts', categorical_features)

    fig = px.bar(df, x=countplot_name, title=f'{countplot_name} countplot visualisation')
    st.plotly_chart(fig)

    fig = px.bar(df, x=countplot_name, color='stroke', title=f'{countplot_name} with stroke')
    st.plotly_chart(fig)

    # CONTINUOUS FEATURES VISUALISATION
    st.text('----------------------------------------------------------------------------')
    st.header('Continuous features Visualisation')

    fig = px.histogram(df, x='age', title='Age counplot visualisation')
    st.plotly_chart(fig)

    fig = px.histogram(df, x='bmi', title='BMI(Body Mass Index) counplot visualisation')
    st.plotly_chart(fig)
    
    fig = px.histogram(df, x='avg_glucose_level', title='Average Glucose Level counplot visualisation')
    st.plotly_chart(fig)

    st.text('----------------------------------------------------------------------------')

    # OUTLIERS DETECTION VISUALISATION
    st.header('Question 3')
    st.write('What if there are any Outliers in the dataset?')

    numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
    # discrete_features = [feature for feature in numerical_features if len(df[feature].unique()) < 25]
    continuous_features = [feature for feature in numerical_features if len(df[feature].unique()) > 25]

    outlier_feature_name = st.selectbox('Features names: ', continuous_features)
    fig = px.box(df, y=outlier_feature_name, color='stroke', title='Outlier Detection (Boxplot)')
    st.plotly_chart(fig)

    fig = px.violin(df, x=outlier_feature_name, color='stroke', title='Outlier Detection (Violinplot)')
    st.plotly_chart(fig)

    st.text('----------------------------------------------------------------------------')

    # PAIRPLOT
    st.header('Question 4')
    st.write('How if we can visualize all the features together?')
    st.write("Why not, here's pairplot")
    st.header('Pairplot Visualisation')
    fig = sns.pairplot(data=df, hue='stroke')
    st.pyplot(fig)

if user_menu == 'App':
    #TITLE
    st.title('Stroke Risk Prediction App')

    # Input from the user 
    gender = st.selectbox('Gender', df['gender'].unique())
    age = st.selectbox('Age', df['age'].unique())
    hypertension = st.selectbox('Hypertension (1 for Yes, 0 for No)', df['hypertension'].unique())
    heart_disease = st.selectbox('Heart Disease (1 for Yes, 0 for No)', df['heart_disease'].unique())
    ever_married = st.selectbox('Ever Married', df['ever_married'].unique())
    work_type = st.selectbox('Work Type', df['work_type'].unique())
    Residence_type = st.selectbox('Residence Type', df['Residence_type'].unique())
    avg_glucose_level = st.selectbox('Average Glucose level', df['avg_glucose_level'].unique())
    bmi = st.selectbox('BMI(Body Mass Index)', df['bmi'].unique())
    smoking_status = st.selectbox('Smoking Status', df['smoking_status'].unique())

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
