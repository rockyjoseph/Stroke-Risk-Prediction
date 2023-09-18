import sys, os
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self):
        try:
            df = pd.read_csv('artifacts\data.csv')
            logging.info('Splitting train-test input data')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.data_transformation_config.test_data_path, index=False, header=True)

            logging.info('Data Transformation completed!')

            return (
                self.data_transformation_config.train_data_path,
                self.data_transformation_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)