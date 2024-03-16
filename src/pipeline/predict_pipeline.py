import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        gender: str,
        age: int,
        hypertension: int,
        heart_disease: int,
        ever_married: str,
        work_type: str,
        Residence_type: str,
        avg_glucose_level: float,
        bmi: float,
        smoking_status: str,
    ):

        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.ever_married = ever_married
        self.work_type = work_type
        self.Residence_type = Residence_type
        self.avg_glucose_level = avg_glucose_level
        self.bmi = bmi
        self.smoking_status = smoking_status

    def get_data_as_data_frame(self):
        try:
            custom_data_input = {
                'gender':[self.gender],
                'age':[self.age],
                'hypertension':[self.hypertension],
                'heart_disease':[self.heart_disease],
                'ever_married':[self.ever_married],
                'work_type':[self.work_type],
                'Residence_type':[self.Residence_type],
                'avg_glucose_level':[self.avg_glucose_level],
                'bmi':[self.bmi],
                'smoking_status':[self.smoking_status],
            }

            return pd.DataFrame(custom_data_input)

        except Exception as e:
            raise CustomException(e, sys)
