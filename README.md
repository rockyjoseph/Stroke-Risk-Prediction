# Stroke Risk Prediction

## 1. Overview
---
This design doc outlines the development of a web application for stroke risk prediction using a kaggle dataset. The application will utilize machine learning models that:

- Evaluates whether the person will having a risk of getting stroke based on process parameters, including other features such as BMI(Body Mass Index), heart disease, smoking status, etc.

- Identifies if you are likely to stroke in near future by getting the data of your habits.

## 2. Success Metrics
---
The success of the project will be measured based on the following metrics:

- Precsion, recall, and F1 score of the machine learning models.
- Responsiveness and ease of use of the web application.

## 3. Requirements & Constraints
---
### 3.1 Functional Requirements

The web application should provide the following functionality:

- Users can provide the process parameters to the model and receive a prediction of whether they are having risk of getting any stroke or not.
- Users can visualize the data and gain insights into the behavior of the features.

### 3.2 Non-functional Requirements

The web application should meet the following non-functional requirements:

- The model should have high precision, recall, and F1 score.
- The web application should be responsive and easy to use.
- The web application should be secure and protect user data.

### 3.3 Constraints

- The application should be built using Streamlit and deployed using Docker and Netlify.
- The cost of deployment should be minimal or free.

## 4. Methodology
---
### 4.1. Problem Statement

The problem is to develop a machine learning model that predicts risk of getting a stroke or not.

### 4.2. Data

The dataset consists of 5000+ data points stored as rows with 11 features in columns. The features include process parameters such as gender and process age, bmi(body mass index), average glucose level, and smoking status. The target variable is a binary label indicating whether the person is having a risk of getting stroke or not.

### 4.3. Techniques
We will utilize a binary classification model to get the person details and give the prediction of getting a stroke or not respectively. The following machine learning techniques will be used:

- Data preprocessing and cleaning
- Feature engineering and selection
- Model selection and training
- Hyperparameter tuning
- Model evaluation and testing

## 6. Architecture
---
The web application architecture will consist of the following components:

- A frontend web application built using Streamlit
- A machine learning model for stroke risk prediction
- Docker containers to run the model
- Cloud infrastructure to host the application
- CI/CD pipeline using GitHub Actions for automated deployment

The Streamlit will interact with the github to request predictions, model training, and data storage. The backend server will manage user authentication, data storage, and model training. The machine learning model will be trained and deployed using Docker containers. The application will be hosted on Netlify. The CI/CD pipeline will be used to automate the deployment process.

## 7. Pipeline
---

### Project Pipeline

The pipeline follows the following sequence of steps:

`Data`: The pipeline starts with the input data, which is sourced from a specified location. It can be in the form of a CSV file or any other supported format.

`Preprocessing`: The data undergoes preprocessing steps to clean, transform, and prepare it for model training. This stage handles tasks such as missing value imputation, feature scaling, and categorical variable encoding.

`Model Training`: The preprocessed data is used to train machine learning models. The pipeline supports building multiple models, allowing for experimentation and comparison of different algorithms or hyperparameters.

`Model Evaluation`: The trained models are evaluated using appropriate evaluation metrics to assess their performance. This stage helps in selecting the best-performing model for deployment.

`Netlity`: The project is deployed on Netlify. Netlify provides a cloud hosting solution that allows for scalability, reliability, and easy management of the web application.

`Web App`: The web application is accessible via a web browser, providing a user-friendly interface for interacting with the prediction functionality. Users can input new data and obtain predictions from the deployed model.

`Prediction`: The deployed model uses the input data from the web application to generate predictions. These predictions are then displayed to the user via the web interface.

`Data`: The predicted data is captured and stored, providing a record of the predictions made by the web application. This data can be used for analysis, monitoring, or further processing as needed.

`CI/CD Pipeline`: The pipeline is automated using GitHub Actions, which allows for continuous integration and deployment of the application. This automation ensures that the application is always up-to-date and provides a consistent experience for users.


## 8. Conclusion

[Click here](http://64.227.150.232:8501/)  to use the web application.

This design doc outlines the development of a web application for stroke risk prediction using a kaggle dataset. The application will utilize a machine learning model that identifies equipment failures based on process parameters, including gender and process age, bmi(body mass index), average glucose level, smoking status. The web application will be built using Streamlit and deployed using Docker and Netlify.