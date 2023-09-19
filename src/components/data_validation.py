import os, sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.over_sampling import SMOTE

from src.logger import logging
from src.utils import save_object

from src.exception import CustomException

@dataclass
class DataValidationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()

    def get_data_validation_object(self):
        try:
            numerical_columns = ['age','avg_glucose_level','bmi']
            categorical_columns = ['gender','ever_married','work_type','Residence_type','smoking_status']

            num_pipeline = Pipeline(
                steps = [
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('ohe', OneHotEncoder())
                ]
            )

            logging.info(f'Categorical columns {categorical_columns}')
            logging.info(f'Numerical columns {numerical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the Train-Test data")
            logging.info("Obtaining preprocessing object")

            sample = SMOTE()
            preprocessing_object = self.get_data_validation_object()

            test_df = test_df[test_df['gender'] != 'Other']

            target_column_name = 'stroke'
            categorical_columns = ['gender','ever_married','work_type','Residence_type','smoking_status']

            q1 = train_df['bmi'].quantile(.25)
            q3 = train_df['bmi'].quantile(.75)

            IQR = q3 - q1

            train_df['bmi'] = train_df['bmi'].apply(lambda x:np.nan if x > q3 + 1.5 * IQR or x < q1 - 1.5 * IQR else x)
            test_df['bmi'] = test_df['bmi'].apply(lambda x:np.nan if x > q3 + 1.5 * IQR or x < q1 - 1.5 * IQR else x)

            upper_limit = q3 + 1.5 * IQR
            lower_limit = q1 - 1.5 * IQR

            train_df = train_df[(train_df['bmi'] <= upper_limit) & (train_df['bmi'] >= lower_limit)]
            test_df = test_df[(test_df['bmi'] <= upper_limit) & (test_df['bmi'] >= lower_limit)]

            # train_encoder = pd.get_dummies(train_df[categorical_columns], drop_first=True)
            # train_df[train_encoder.columns] = train_encoder            

            # test_encoder = pd.get_dummies(test_df[categorical_columns], drop_first=True)
            # test_df[test_encoder.columns] = test_encoder


            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_df.drop(categorical_columns, axis=1, inplace=True)
            test_df.drop(categorical_columns, axis=1, inplace=True)

            feature_train_df, target_train_df = sample.fit_resample(input_feature_train_arr, target_feature_train_df)
            feature_test_df, target_test_df = sample.fit_resample(input_feature_test_arr, target_feature_test_df)

            train_arr = np.c_[
                feature_train_df, target_train_df
            ]

            test_arr = np.c_[
                feature_test_df, target_test_df
            ]

            logging.info(f'Saved Preprocessing object')

            save_object(
                file_path = self.data_validation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )

            return train_arr, test_arr, self.data_validation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)