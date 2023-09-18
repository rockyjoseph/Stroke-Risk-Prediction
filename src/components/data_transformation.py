import sys, os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    data_transformation_file_path = os.path.join('artifacts','model.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_arr, test_arr):
        try:
            logging.info('Splitting train-test input data')

            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1],
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)